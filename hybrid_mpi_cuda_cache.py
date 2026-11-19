import os
import sys
import json
import time
import math
import hashlib
import platform
import subprocess
import numpy as np

from mpi4py import MPI
from numba import cuda, njit, float64

TILE = 16
N = 256
STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15

@cuda.jit
def pack_halos_kernel(u, top, bottom, left, right):
    i = cuda.grid(1)
    if i < N:
        top[i] = u[1, i + 1]
        bottom[i] = u[N, i + 1]
        left[i] = u[i + 1, 1]
        right[i] = u[i + 1, N]

@cuda.jit
def unpack_halos_kernel(u, top, bottom, left, right):
    i = cuda.grid(1)
    if i < N:
        u[0, i + 1] = top[i]
        u[N + 1, i + 1] = bottom[i]
        u[i + 1, 0] = left[i]
        u[i + 1, N + 1] = right[i]

@cuda.jit
def tiled_stencil_kernel(u, out, kappa, source, dt, dx, mode):
    tile = cuda.shared.array((18, 18), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    x = bx * TILE + tx + 1
    y = by * TILE + ty + 1

    tid = ty * TILE + tx

    for q in range(tid, 324, 256):
        sy = q // 18
        sx = q % 18
        gy = by * TILE + sy
        gx = bx * TILE + sx
        tile[sy, sx] = u[gy, gx]

    cuda.syncthreads()

    if x <= N and y <= N:
        interior = (x >= 2 and x <= N - 1 and
                    y >= 2 and y <= N - 1)

        if (mode == 0 or
            (mode == 1 and interior) or
            (mode == 2 and not interior)):

            c = tile[ty + 1, tx + 1]

            lap = (
                tile[ty, tx + 1] +
                tile[ty + 2, tx + 1] +
                tile[ty + 1, tx] +
                tile[ty + 1, tx + 2] -
                4.0 * c
            ) / (dx * dx)

            out[y, x] = (
                c + dt * (
                    kappa[y, x] * lap +
                    source[y, x] -
                    LAMBDA_DAMP * c
                )
            )

def host_shm_accumulate(shm_view, index, iterations):
    import ctypes

    lib = ctypes.CDLL("./volatile_benchmark.so")

    lib.volatile_increment.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
        ctypes.c_uint64
    ]
    lib.volatile_increment.restype = None

    ptr = shm_view.ctypes.data_as(
        ctypes.POINTER(ctypes.c_uint64)
    )

    lib.volatile_increment(
        ptr, int(index), int(iterations)
    )

def gpu_uuid():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid",
         "--format=csv,noheader"],
        text=True
    ).strip().splitlines()[0]

def init_hybrid_context(username):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if comm.Get_size() != 4:
        raise RuntimeError("Exactly 4 MPI ranks required")

    cart = comm.Create_cart(
        dims=[2, 2],
        periods=[False, False],
        reorder=False
    )

    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)

    cuda.select_device(0)

    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    if node.Get_size() != 4:
        raise RuntimeError(
            "All four ranks must share one node"
        )

    # Extra bytes allow explicit 64-byte alignment.
    win = MPI.Win.Allocate_shared(
        64 * 8 + 64 if node.Get_rank() == 0 else 0,
        1,
        comm=node
    )

    memory, _ = win.Shared_query(0)
    raw = np.ndarray(
        buffer=memory,
        dtype=np.uint8,
        shape=(64 * 8 + 64,)
    )

    offset = (-raw.ctypes.data) % 64
    shm = np.ndarray(
        buffer=memory,
        dtype=np.uint64,
        shape=(64,),
        offset=int(offset)
    )

    uuid = comm.bcast(
        gpu_uuid() if rank == 0 else None,
        root=0
    )

    cpu_signature = (
        f"{platform.machine()}::{os.cpu_count()}"
    )

    digest = hashlib.sha256(
        f"{username}::{cpu_signature}::{uuid}".encode()
    ).digest()

    seed = int.from_bytes(digest[:8], "big")

    if node.Get_rank() == 0:
        shm[:] = 0
    node.Barrier()

    return {
        "comm": comm,
        "cart": cart,
        "node": node,
        "win": win,
        "shm": shm,
        "rank": rank,
        "neighbors": [up, down, left, right],
        "uuid": uuid,
        "seed": seed,
        "username": username
    }

def initialize_fields(ctx):
    rank = ctx["rank"]
    coords = ctx["cart"].Get_coords(rank)

    u = np.zeros((258, 258), dtype=np.float64)
    kappa = np.zeros_like(u)
    source = np.zeros_like(u)

    dx = 1.0 / 513.0

    for iy in range(1, 257):
        for ix in range(1, 257):
            gy = coords[0] * 256 + iy
            gx = coords[1] * 256 + ix

            x = gx * dx
            y = gy * dx

            u[iy, ix] = (
                math.sin(math.pi * x) *
                math.sin(math.pi * y)
            )

            kappa[iy, ix] = (
                0.15 +
                0.08 * math.sin(math.pi * x) ** 2 *
                math.cos(math.pi * y) ** 2
            )

            source[iy, ix] = 0.0

    return u, kappa, source

def make_buffers():
    names = ["top", "bottom", "left", "right"]

    return {
        name: {
            "send": cuda.pinned_array(N, dtype=np.float64),
            "recv": cuda.pinned_array(N, dtype=np.float64),
            "device_send": cuda.device_array(
                N, dtype=np.float64
            ),
            "device_recv": cuda.device_array(
                N, dtype=np.float64
            )
        }
        for name in names
    }

def exchange_halos(ctx, u, buffers, stream, interior=None):
    cart = ctx["cart"]
    neighbors = ctx["neighbors"]
    names = ["top", "bottom", "left", "right"]

    pack_halos_kernel[1, 256, stream](
        u,
        *[buffers[n]["device_send"] for n in names]
    )

    for name in names:
        b = buffers[name]
        b["device_send"].copy_to_host(
            b["send"], stream=stream
        )

    stream.synchronize()

    recv_tags = [11, 10, 13, 12]
    send_tags = [10, 11, 12, 13]

    requests = []

    for i, name in enumerate(names):
        neighbor = neighbors[i]

        if neighbor != MPI.PROC_NULL:
            requests.append(
                cart.Irecv(
                    buffers[name]["recv"],
                    source=neighbor,
                    tag=recv_tags[i]
                )
            )

    for i, name in enumerate(names):
        neighbor = neighbors[i]

        if neighbor != MPI.PROC_NULL:
            requests.append(
                cart.Isend(
                    buffers[name]["send"],
                    dest=neighbor,
                    tag=send_tags[i]
                )
            )

    # Interior GPU work can execute during MPI exchange.
    if interior is not None:
        interior()

    MPI.Request.Waitall(requests)

    for i, name in enumerate(names):
        if neighbors[i] == MPI.PROC_NULL:
            buffers[name]["recv"].fill(0.0)

        buffers[name]["device_recv"].copy_to_device(
            buffers[name]["recv"],
            stream=stream
        )

    unpack_halos_kernel[1, 256, stream](
        u,
        *[buffers[n]["device_recv"] for n in names]
    )

def run_solver(ctx, mode, dt, dx, kmax):
    comm = ctx["comm"]
    rank = ctx["rank"]

    host_u, host_k, host_s = initialize_fields(ctx)

    u = cuda.to_device(host_u)
    out = cuda.to_device(host_u)
    k = cuda.to_device(host_k)
    s = cuda.to_device(host_s)

    stream_interior = cuda.stream()
    stream_boundary = cuda.stream()
    buffers = make_buffers()

    block = (16, 16)
    grid = (16, 16)

    energies = {}
    residuals = {}

    initial = host_u[1:257, 1:257].copy()

    comm.Barrier()
    start = MPI.Wtime()

    for step in range(STEPS + 1):
        if step in (0, 100, 200, 300):
            current = u.copy_to_host()
            local_energy = np.sum(
                current[1:257, 1:257] ** 2
            ) * dx * dx

            energy = comm.allreduce(
                local_energy, op=MPI.SUM
            )

            local_residual = np.max(
                np.abs(
                    current[1:257, 1:257] - initial
                )
            )

            residual = comm.allreduce(
                local_residual, op=MPI.MAX
            )

            energies[str(step)] = float(energy)
            residuals[str(step)] = float(residual)

            if rank == 0:
                print(
                    f"{mode} step={step} "
                    f"energy={energy:.15e} "
                    f"residual={residual:.15e}",
                    flush=True
                )

        if step == STEPS:
            break

        if mode == "overlapped":
            def interior_work():
                tiled_stencil_kernel[
                    grid, block, stream_interior
                ](u, out, k, s, dt, dx, 1)

            exchange_halos(
                ctx, u, buffers,
                stream_boundary, interior_work
            )

            stream_boundary.synchronize()

            tiled_stencil_kernel[
                grid, block, stream_boundary
            ](u, out, k, s, dt, dx, 2)

            stream_interior.synchronize()
            stream_boundary.synchronize()

        else:
            exchange_halos(
                ctx, u, buffers, stream_boundary
            )
            stream_boundary.synchronize()

            tiled_stencil_kernel[
                grid, block, stream_boundary
            ](u, out, k, s, dt, dx, 0)

            stream_boundary.synchronize()

        u, out = out, u

    cuda.synchronize()

    elapsed = MPI.Wtime() - start
    elapsed = comm.allreduce(elapsed, op=MPI.MAX)

    result = u.copy_to_host()

    return result, elapsed * 1000, energies, residuals

def benchmark_shm_strides(ctx):
    node = ctx["node"]
    shm = ctx["shm"]
    rank = node.Get_rank()

    results = {}

    # Compile before timing.
    host_shm_accumulate(shm, rank * 16, 1)
    node.Barrier()

    for stride in (1, 8, 16):
        if rank == 0:
            shm[:] = 0

        node.Barrier()
        ctx["win"].Sync()
        node.Barrier()

        start = MPI.Wtime()

        host_shm_accumulate(
            shm, rank * stride, SHM_ITERS
        )

        node.Barrier()
        elapsed = MPI.Wtime() - start

        elapsed = node.allreduce(
            elapsed, op=MPI.MAX
        )

        ctx["win"].Sync()
        node.Barrier()

        expected = np.uint64(SHM_ITERS)
        local_ok = shm[rank * stride] == expected
        all_ok = node.allreduce(
            bool(local_ok), op=MPI.LAND
        )

        if not all_ok:
            raise RuntimeError(
                f"Shared memory counter failed: {stride}"
            )

        results[str(stride)] = elapsed * 1000

    return results

def audit_ptx():
    asm = tiled_stencil_kernel.inspect_asm()

    if isinstance(asm, dict):
        asm = "\n".join(asm.values())

    import re

    counts = {
        "st.shared.f64": len(
            re.findall(r"\bst\.shared\.f64\b", asm)
        ),
        "ld.shared.f64": len(
            re.findall(r"\bld\.shared\.f64\b", asm)
        ),
        "bar.sync": len(
            re.findall(r"\bbar\.sync\b", asm)
        )
    }

    counts["passed"] = all(
        counts[k] > 0
        for k in (
            "st.shared.f64",
            "ld.shared.f64",
            "bar.sync"
        )
    )

    return counts






def audit_llvm(*args, **kwargs):
    import subprocess

    asm = subprocess.check_output(
        ["objdump", "-d", "-M", "intel",
         "./volatile_benchmark.so"],
        text=True
    )

    marker = "<volatile_increment>:"
    if marker not in asm:
        raise RuntimeError("C function not found")

    body = asm.split(marker, 1)[1]

    # Stop at the next function.
    lines = []
    for line in body.splitlines():
        if line.strip().endswith(">:"):
            break
        lines.append(line)

    instructions = []

    for line in lines:
        # objdump separates address, bytes and instruction
        # using tabs.
        parts = line.strip().split(chr(9))
        if len(parts) >= 3:
            instructions.append(parts[-1].strip())

    read_found = any(
        "mov" in ins
        and "[rsi]" in ins
        and ins.split("mov", 1)[1].strip().startswith("rax,")
        for ins in instructions
    )

    write_found = any(
        "mov" in ins
        and "[rsi]" in ins
        and ins.split("mov", 1)[1].strip().startswith("QWORD PTR [rsi],")
        for ins in instructions
    )

    loop_found = any(
        ins.startswith("jne ")
        for ins in instructions
    )

    if not (read_found and write_found and loop_found):
        raise RuntimeError(
            f"Audit failed: read={read_found}, "
            f"write={write_found}, loop={loop_found}, "
            f"instructions={instructions}"
        )

    return {
        "volatile_load": 1,
        "volatile_store": 1,
        "loop_present": True,
        "passed": True
    }

def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python hybrid_mpi_cuda_cache.py "
            "GITHUB_USERNAME"
        )

    username = sys.argv[1]
    ctx = init_hybrid_context(username)

    comm = ctx["comm"]
    rank = ctx["rank"]

    dx = 1.0 / 513.0

    _, host_k, _ = initialize_fields(ctx)

    local_kmax = float(np.max(host_k))
    kmax = comm.allreduce(
        local_kmax, op=MPI.MAX
    )

    dt_max = dx * dx / (4.0 * kmax)
    dt = 0.8 * dt_max

    if rank == 0:
        print(
            f"mpi_ranks=4 topology=2x2 "
            f"GPU_UUID={ctx['uuid']} "
            f"INSTANCE_SEED={ctx['seed']}",
            flush=True
        )

        print(
            f"kappa_max={kmax:.15e} "
            f"dx={dx:.15e} "
            f"dt_max={dt_max:.15e} "
            f"dt={dt:.15e}",
            flush=True
        )

    serial, serial_ms, se, sr = run_solver(
        ctx, "serialized", dt, dx, kmax
    )

    overlap, overlap_ms, oe, ore = run_solver(
        ctx, "overlapped", dt, dx, kmax
    )

    local_diff = np.max(
        np.abs(serial - overlap)
    )

    max_diff = comm.allreduce(
        float(local_diff), op=MPI.MAX
    )

    if rank == 0:
        ptx = audit_ptx()
        print("PTX_AUDIT", json.dumps(ptx))
    else:
        ptx = None

    ptx = comm.bcast(ptx, root=0)

    llvm = audit_llvm(ctx)
    shm_times = benchmark_shm_strides(ctx)

    slowdown = (
        shm_times["1"] / shm_times["8"]
        if shm_times["8"] > 0 else None
    )

    manifest = {
        "github_username": username,
        "instance_seed": ctx["seed"],
        "gpu_uuid": ctx["uuid"],
        "cpu_signature": (
            f"{platform.machine()}::{os.cpu_count()}"
        ),
        "mpi_ranks": 4,
        "topology": [2, 2],
        "global_grid": [512, 512],
        "local_grid": [258, 258],
        "steps": STEPS,
        "kappa_max": kmax,
        "dx": dx,
        "dt_max": dt_max,
        "dt": dt,
        "gpu_serialized_ms": serial_ms,
        "gpu_overlapped_ms": overlap_ms,
        "max_diff": max_diff,
        "parity_passed": bool(max_diff < 1e-12),
        "serialized_energy": se,
        "overlapped_energy": oe,
        "serialized_residual": sr,
        "overlapped_residual": ore,
        "ptx_audit": ptx,
        "llvm_audit": llvm,
        "shm_stride_ms": shm_times,
        "false_sharing_slowdown": slowdown
    }

    if rank == 0:
        with open(
            "hybrid_lab_manifest.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(manifest, f, indent=2)

        print(
            "RESULTS",
            json.dumps(manifest, indent=2),
            flush=True
        )

        print(
            "PARITY:",
            "PASS" if max_diff < 1e-12 else "FAIL",
            flush=True
        )

        print(
            "Manifest saved: hybrid_lab_manifest.json",
            flush=True
        )

    comm.Barrier()
    ctx["win"].Free()
    ctx["node"].Free()
    ctx["cart"].Free()

if __name__ == "__main__":
    main()
