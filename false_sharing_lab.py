
import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess

import numpy as np
from numba import njit, uint64, types
from numba.core import cgutils
from numba.extending import intrinsic
from llvmlite import ir

try:
    import resource
except ImportError:
    resource = None


GIT_USERNAME = "230103136-web"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)


# TASK 1 — Hardware topology and identity

def detect_cache_line_size():
    linux_path = (
        "/sys/devices/system/cpu/cpu0/cache/"
        "index0/coherency_line_size"
    )

    if os.path.exists(linux_path):
        with open(linux_path, "r") as f:
            return int(f.read().strip())

    if sys.platform == "darwin":
        return int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.cachelinesize"],
                text=True
            ).strip()
        )

    return 64


def get_cpu_model():
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                value = subprocess.check_output(
                    ["sysctl", "-n", key],
                    text=True,
                    stderr=subprocess.DEVNULL
                ).strip()
                if value:
                    return value
            except Exception:
                pass

    return platform.processor() or platform.machine()


def init_system_context(git_user):
    assert git_user.strip()
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME"

    logical_cores = os.cpu_count() or 1

    if logical_cores < NUM_THREADS:
        raise RuntimeError("At least 4 logical cores required")

    line_size = detect_cache_line_size()
    cpu_model = get_cpu_model()

    cpu_sig = (
        f"{platform.machine()}::"
        f"{platform.processor()}::{logical_cores}"
    )

    digest = hashlib.sha256(
        f"{git_user.strip()}::{cpu_sig}".encode()
    ).hexdigest()

    seed = int(digest[:16], 16)

    affinity_supported = hasattr(os, "sched_setaffinity")

    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))

    return {
        "git_username": git_user,
        "cpu_model": cpu_model,
        "cpu_signature": cpu_sig,
        "logical_cores": logical_cores,
        "cache_line_bytes": line_size,
        "instance_seed": seed,
        "affinity_supported": affinity_supported,
        "available_affinity": affinity,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__
    }


# Allocate a 64-byte-aligned uint64 buffer

def aligned_uint64(length, alignment=64):
    raw = np.zeros(
        length + alignment // 8,
        dtype=np.uint64
    )

    offset = (
        -raw.ctypes.data
    ) % alignment // 8

    view = raw[offset:offset + length]

    assert view.ctypes.data % alignment == 0
    return view


# TASK 2 — GIL-free native counter kernel

@intrinsic
def native_memory_add(typingctx, arr, index, increment):
    if not isinstance(arr, types.Array):
        return None
    if arr.dtype != types.uint64:
        return None
    if not isinstance(index, types.Integer):
        return None
    if increment != types.uint64:
        return None

    signature = types.void(arr, index, increment)

    def codegen(context, builder, signature, args):
        array = context.make_array(
            signature.args[0]
        )(context, builder, value=args[0])

        ptr = cgutils.get_item_pointer(
            context,
            builder,
            signature.args[0],
            array,
            [args[1]],
            wraparound=False
        )

        # Compiler memory barriers prevent LLVM from
        # promoting the repeated loads/stores into registers.
        barrier = ir.InlineAsm(
            ir.FunctionType(ir.VoidType(), []),
            "",
            "~{memory}",
            side_effect=True
        )

        builder.call(barrier, [])
        current = builder.load(ptr)
        updated = builder.add(current, args[2])
        builder.store(updated, ptr)
        builder.call(barrier, [])

        return context.get_dummy_value()

    return signature, codegen


@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    increment = np.uint64((seed & 7) + 1)

    for i in range(iters):
        native_memory_add(buffer, slot, increment)


# TASK 4 — Register-privatized baseline

@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    increment = np.uint64((seed & 7) + 1)
    accumulator = np.uint64(0)

    for i in range(iters):
        accumulator += increment

    buffer[slot] = accumulator


def get_context_switches():
    if resource is None:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF)

    return {
        "voluntary": usage.ru_nvcsw,
        "involuntary": usage.ru_nivcsw
    }


# TASK 3 — Threaded stride benchmark

def run_stride_benchmark(
    stride, context, private=False
):
    length = (NUM_THREADS - 1) * stride + 1

    buffer = aligned_uint64(length, alignment=context["cache_line_bytes"])

    assert buffer.ctypes.data % context["cache_line_bytes"] == 0

    seed = context["instance_seed"]

    kernel = (
        worker_private_accumulate
        if private
        else worker_strided_accumulate
    )

    # Warm up JIT compilation before timing.
    warm = aligned_uint64(1)
    kernel(warm, 0, 1, 1)

    barrier = threading.Barrier(NUM_THREADS + 1)
    errors = []

    def thread_job(tid):
        try:
            slot = tid * stride
            thread_seed = (seed + tid) & 0xFFFFFFFF

            barrier.wait()

            kernel(
                buffer,
                slot,
                ITERATIONS_PER_THREAD,
                thread_seed
            )
        except Exception as exc:
            errors.append(str(exc))

    threads = [
        threading.Thread(
            target=thread_job,
            args=(tid,)
        )
        for tid in range(NUM_THREADS)
    ]

    before = get_context_switches()

    for t in threads:
        t.start()

    start_ns = time.perf_counter_ns()
    barrier.wait()

    for t in threads:
        t.join()

    elapsed_ns = time.perf_counter_ns() - start_ns

    after = get_context_switches()

    if errors:
        raise RuntimeError(str(errors))

    values = [
        int(buffer[tid * stride])
        for tid in range(NUM_THREADS)
    ]

    expected = [
        ITERATIONS_PER_THREAD
        * (((seed + tid) & 0xFFFFFFFF & 7) + 1)
        for tid in range(NUM_THREADS)
    ]

    parity = values == expected

    switches = None

    if before is not None and after is not None:
        switches = {
            "voluntary": (
                after["voluntary"] - before["voluntary"]
            ),
            "involuntary": (
                after["involuntary"] - before["involuntary"]
            )
        }

    return {
        "layout": "private" if private else f"stride_{stride}",
        "stride": stride,
        "spacing_bytes": stride * 8,
        "elapsed_ns": int(elapsed_ns),
        "ns_per_update": (
            elapsed_ns
            / (NUM_THREADS * ITERATIONS_PER_THREAD)
        ),
        "context_switches": switches,
        "final_values": values,
        "expected_values": expected,
        "parity_passed": parity
    }


# TASK 5 — LLVM audit and parity


def audit_llvm_and_parity(results):
    import re

    signature = worker_strided_accumulate.signatures[0]
    llvm = worker_strided_accumulate.inspect_llvm(signature)

    native_function = llvm.split(
        "define ", 2
    )[1].split("\\ndefine ", 1)[0]

    store_lines = [
        line.strip()
        for line in native_function.splitlines()
        if "store i64" in line
    ]

    load_lines = [
        line.strip()
        for line in native_function.splitlines()
        if "load i64" in line
    ]

    barrier_lines = [
        line.strip()
        for line in native_function.splitlines()
        if "asm sideeffect" in line
        and "~{memory}" in line
    ]

    loop_verified = False

    blocks = re.split(
        r"(?m)^(?=[A-Za-z0-9_.$-]+:)",
        native_function
    )

    for block in blocks:
        match = re.match(
            r"([A-Za-z0-9_.$-]+):",
            block
        )

        if not match:
            continue

        label = match.group(1)

        has_load = "load i64" in block
        has_store = "store i64" in block
        has_barriers = block.count("asm sideeffect") >= 2

        has_backedge = bool(
            re.search(
                r"br i1[^\\n]*label %"
                + re.escape(label)
                + r"(?=[,\\s]|$)",
                block
            )
        )

        if (
            has_load
            and has_store
            and has_barriers
            and has_backedge
        ):
            loop_verified = True
            print("LLVM loop verified:", label)
            break

    parity = all(
        result["parity_passed"]
        for result in results.values()
    )

    reference = results["stride_1"]["final_values"]

    same_values = all(
        result["final_values"] == reference
        for result in results.values()
    )

    return {
        "llvm_store_i64_count": len(store_lines),
        "llvm_load_i64_count": len(load_lines),
        "llvm_volatile_store_count": sum(
            "store volatile i64" in line
            for line in store_lines
        ),
        "llvm_memory_barrier_count": len(barrier_lines),
        "llvm_store_examples": store_lines[:10],
        "all_parity_passed": parity,
        "all_layouts_identical": same_values,
        "llvm_has_i64_store": bool(store_lines),
        "memory_store_per_iteration_verified": loop_verified
    }


def main():
    context = init_system_context(GIT_USERNAME)

    print("=" * 65)
    print("CACHE LINE AND FALSE SHARING LAB")
    print("=" * 65)

    print("Git username:", GIT_USERNAME)
    print("CPU:", context["cpu_model"])
    print("Logical cores:", context["logical_cores"])
    print("Cache line:", context["cache_line_bytes"], "bytes")
    print("Instance seed:", context["instance_seed"])
    print("Affinity supported:", context["affinity_supported"])

    print("\nRunning benchmarks...")

    import statistics

    results = {}
    repetitions = 3

    for stride in STRIDES:
        runs = [
            run_stride_benchmark(stride, context)
            for _ in range(repetitions)
        ]

        times = [r["elapsed_ns"] for r in runs]
        median_ns = int(statistics.median(times))

        result = min(
            runs,
            key=lambda r: abs(r["elapsed_ns"] - median_ns)
        ).copy()

        result["elapsed_ns"] = median_ns
        result["ns_per_update"] = (
            median_ns
            / (NUM_THREADS * ITERATIONS_PER_THREAD)
        )
        result["repetitions"] = repetitions
        result["all_elapsed_ns"] = times

        results[f"stride_{stride}"] = result

        print(
            f"stride={stride:2d} "
            f"median={median_ns} ns "
            f"ns/update={result['ns_per_update']:.6f} "
            f"parity={all(r['parity_passed'] for r in runs)}",
            flush=True
        )

        result["parity_passed"] = all(
            r["parity_passed"] for r in runs
        )

    private_runs = [
        run_stride_benchmark(8, context, private=True)
        for _ in range(repetitions)
    ]

    private_times = [
        r["elapsed_ns"] for r in private_runs
    ]

    private_median = int(
        statistics.median(private_times)
    )

    private_result = min(
        private_runs,
        key=lambda r: abs(
            r["elapsed_ns"] - private_median
        )
    ).copy()

    private_result["elapsed_ns"] = private_median
    private_result["ns_per_update"] = (
        private_median
        / (NUM_THREADS * ITERATIONS_PER_THREAD)
    )
    private_result["repetitions"] = repetitions
    private_result["all_elapsed_ns"] = private_times
    private_result["parity_passed"] = all(
        r["parity_passed"] for r in private_runs
    )

    results["private"] = private_result

    print(
        f"private median={private_median} ns "
        f"ns/update={private_result['ns_per_update']:.6f} "
        f"parity={private_result['parity_passed']}"
    )

    audit = audit_llvm_and_parity(results)

    slowdown = (
        results["stride_1"]["elapsed_ns"]
        / results["stride_8"]["elapsed_ns"]
    )

    print("\nSlowdown stride 1 / stride 8:", slowdown)
    print("LLVM i64 stores:", audit["llvm_store_i64_count"])
    print("LLVM volatile stores:", audit["llvm_volatile_store_count"])
    print("LLVM examples:", audit["llvm_store_examples"])

    # A native store in LLVM is not sufficient to
    # prove one memory store per loop iteration.
    verification_passed = (
        audit["all_parity_passed"]
        and audit["all_layouts_identical"]
        and audit["llvm_has_i64_store"]
        and audit["memory_store_per_iteration_verified"]
    )

    manifest = {
        "system": context,
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "benchmarks": results,
        "slowdown_stride1_vs_stride8": slowdown,
        "llvm_audit": audit,
        "verification_passed": verification_passed,
        "memory_store_per_iteration_verified": audit["memory_store_per_iteration_verified"]
    }

    with open(
        "cache_lab_manifest.json", "w",
        encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    print("\nManifest: cache_lab_manifest.json")
    print("Verification passed:", verification_passed)

    if not audit["memory_store_per_iteration_verified"]:
        print(
            "WARNING: Compiler may have optimized "
            "repeated memory stores. "
            "False-sharing timing is not yet validated."
        )


if __name__ == "__main__":
    main()
