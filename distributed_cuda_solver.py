import os
import sys
import json
import math
import hashlib
import subprocess
import re
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = 18
GLOBAL_NX = GLOBAL_NY = 512
NY = NX = 256
STEPS = 400
LAMBDA_DAMP = 0.15
DX = DY = 1.0 / 513.0

DIRECTIONS = ("up", "down", "left", "right")
OPPOSITE = {
    "up": "down", "down": "up",
    "left": "right", "right": "left"
}
TAGS = {"up": 10, "down": 11, "left": 12, "right": 13}


# TASK 1 — MPI topology and hardware identity

def get_gpu_uuid():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid",
         "--format=csv,noheader"],
        text=True
    ).strip().splitlines()[0].strip()


def derive_student_seed(github_user, gpu_uuid):
    raw = f"{github_user.strip()}::{gpu_uuid}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def init_distributed_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if comm.Get_size() != 4:
        raise RuntimeError("Exactly 4 MPI ranks required")

    cuda.select_device(0)

    uuid = comm.bcast(
        get_gpu_uuid() if rank == 0 else None,
        root=0
    )

    seed = derive_student_seed(github_user, uuid)

    cart = comm.Create_cart(
        [2, 2],
        periods=[False, False],
        reorder=False
    )

    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)

    return {
        "comm": cart,
        "rank": cart.Get_rank(),
        "coords": cart.Get_coords(cart.Get_rank()),
        "neighbors": {
            "up": up, "down": down,
            "left": left, "right": right
        },
        "seed": seed,
        "uuid": uuid,
        "user": github_user
    }


# TASK 2 — Local subdomain initialization

def initialize_subdomains(ctx):
    rr, cc = ctx["coords"]

    y = (rr * NY + np.arange(NY + 2)) * DY
    x = (cc * NX + np.arange(NX + 2)) * DX
    xx, yy = np.meshgrid(x, y)

    phase = (ctx["seed"] % 10000) / 10000.0

    kappa = (
        0.16
        + 0.04 * np.sin(2 * np.pi * xx + phase)
        * np.cos(2 * np.pi * yy)
        + 0.04 * np.sin(4 * np.pi * yy + phase)
    )
    kappa = np.clip(kappa, 0.08, 0.24)

    cx = 0.25 + 0.10 * phase
    cy = 0.35 + 0.10 * (1 - phase)

    u = (
        (0.65 + 0.2 * phase)
        * np.exp(-((xx-cx)**2 + (yy-cy)**2)/0.012)
    )

    u += 0.35 * np.exp(
        -((xx-0.70)**2 + (yy-0.72)**2)/0.018
    )

    source = 0.002 * np.exp(
        -((xx-0.5)**2 + (yy-0.5)**2)/0.008
    )

    nbr = ctx["neighbors"]

    if nbr["up"] == MPI.PROC_NULL:
        u[0, :] = 0.0
    if nbr["down"] == MPI.PROC_NULL:
        u[-1, :] = 0.0
    if nbr["left"] == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if nbr["right"] == MPI.PROC_NULL:
        u[:, -1] = 0.0

    return (
        cuda.to_device(u.astype(np.float64)),
        cuda.to_device(kappa.astype(np.float64)),
        cuda.to_device(source.astype(np.float64)),
        float(np.max(kappa[1:-1, 1:-1]))
    )


# TASK 3 — GPU halo packing and unpacking

@cuda.jit
def pack_halos_kernel(u, out, direction, n):
    q = cuda.grid(1)

    if q < n:
        if direction == 0:
            out[q] = u[1, q+1]
        elif direction == 1:
            out[q] = u[NY, q+1]
        elif direction == 2:
            out[q] = u[q+1, 1]
        else:
            out[q] = u[q+1, NX]


@cuda.jit
def unpack_halos_kernel(u, inp, direction, n):
    q = cuda.grid(1)

    if q < n:
        if direction == 0:
            u[0, q+1] = inp[q]
        elif direction == 1:
            u[NY+1, q+1] = inp[q]
        elif direction == 2:
            u[q+1, 0] = inp[q]
        else:
            u[q+1, NX+1] = inp[q]


def allocate_exchange():
    buffers = {}

    for d in DIRECTIONS:
        n = NX if d in ("up", "down") else NY

        buffers[d] = {
            "ds": cuda.device_array(n, dtype=np.float64),
            "dr": cuda.device_array(n, dtype=np.float64),
            "hs": cuda.pinned_array(n, dtype=np.float64),
            "hr": cuda.pinned_array(n, dtype=np.float64)
        }

    return buffers


def exchange_halos_async(ctx, u, buffers, stream):
    comm = ctx["comm"]
    nbr = ctx["neighbors"]

    for idx, d in enumerate(DIRECTIONS):
        if nbr[d] != MPI.PROC_NULL:
            n = NX if d in ("up", "down") else NY

            pack_halos_kernel[
                (n+127)//128, 128, stream
            ](u, buffers[d]["ds"], idx, n)

            buffers[d]["ds"].copy_to_host(
                buffers[d]["hs"], stream=stream
            )

    stream.synchronize()

    requests = []

    for d in DIRECTIONS:
        if nbr[d] != MPI.PROC_NULL:
            requests.append(
                comm.Irecv(
                    buffers[d]["hr"],
                    source=nbr[d],
                    tag=TAGS[OPPOSITE[d]]
                )
            )

    for d in DIRECTIONS:
        if nbr[d] != MPI.PROC_NULL:
            requests.append(
                comm.Isend(
                    buffers[d]["hs"],
                    dest=nbr[d],
                    tag=TAGS[d]
                )
            )

    return requests


def finish_exchange(ctx, u, buffers, stream, requests):
    MPI.Request.Waitall(requests)

    for idx, d in enumerate(DIRECTIONS):
        if ctx["neighbors"][d] != MPI.PROC_NULL:
            n = NX if d in ("up", "down") else NY

            buffers[d]["dr"].copy_to_device(
                buffers[d]["hr"], stream=stream
            )

            unpack_halos_kernel[
                (n+127)//128, 128, stream
            ](u, buffers[d]["dr"], idx, n)

    stream.synchronize()


# TASK 4 — Shared memory tiled CUDA stencil

@cuda.jit
def tiled_stencil_kernel(
    u, k, s, out, dt,
    rs, re, cs, ce, ring_only
):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    r = rs + cuda.blockIdx.y * 16 + ty
    c = cs + cuda.blockIdx.x * 16 + tx

    sm = cuda.shared.array((18, 18), dtype=float64)

    valid = (
        r >= 1 and r <= NY
        and c >= 1 and c <= NX
    )

    if valid:
        sm[ty+1, tx+1] = u[r, c]
    else:
        sm[ty+1, tx+1] = 0.0

    if tx == 0:
        if valid:
            sm[ty+1, 0] = u[r, c-1]
        else:
            sm[ty+1, 0] = 0.0

    if tx == 15:
        if valid:
            sm[ty+1, 17] = u[r, c+1]
        else:
            sm[ty+1, 17] = 0.0

    if ty == 0:
        if valid:
            sm[0, tx+1] = u[r-1, c]
        else:
            sm[0, tx+1] = 0.0

    if ty == 15:
        if valid:
            sm[17, tx+1] = u[r+1, c]
        else:
            sm[17, tx+1] = 0.0

    cuda.syncthreads()

    if r <= re and c <= ce and valid:
        if (
            ring_only == 0
            or r == 1 or r == NY
            or c == 1 or c == NX
        ):
            z = sm[ty+1, tx+1]
            ku = k[r, c]

            flux_x = (
                0.5 * (ku+k[r, c+1])
                * (sm[ty+1, tx+2]-z)
                - 0.5 * (ku+k[r, c-1])
                * (z-sm[ty+1, tx])
            ) / (DX*DX)

            flux_y = (
                0.5 * (ku+k[r+1, c])
                * (sm[ty+2, tx+1]-z)
                - 0.5 * (ku+k[r-1, c])
                * (z-sm[ty, tx+1])
            ) / (DY*DY)

            out[r, c] = z + dt * (
                flux_x + flux_y
                - LAMBDA_DAMP*z*z*z
                + s[r, c]
            )


def launch_region(
    u, k, s, out, dt, stream,
    rs, re, cs, ce, ring
):
    gx = (ce-cs+16)//16
    gy = (re-rs+16)//16

    tiled_stencil_kernel[
        (gx, gy), (16, 16), stream
    ](
        u, k, s, out, dt,
        rs, re, cs, ce, int(ring)
    )


def step_overlapped(
    ctx, u, k, s, out, dt,
    buffers, sb, si, overlap=True
):
    requests = exchange_halos_async(
        ctx, u, buffers, sb
    )

    if overlap:
        launch_region(
            u, k, s, out, dt, si,
            2, NY-1, 2, NX-1, False
        )

        finish_exchange(
            ctx, u, buffers, sb, requests
        )

        launch_region(
            u, k, s, out, dt, sb,
            1, NY, 1, NX, True
        )

        si.synchronize()
        sb.synchronize()

    else:
        finish_exchange(
            ctx, u, buffers, sb, requests
        )

        launch_region(
            u, k, s, out, dt, sb,
            1, NY, 1, NX, False
        )

        sb.synchronize()

    return out, u


# TASK 5 — Reductions, benchmarking and PTX audit

def stats(ctx, u, previous):
    a = u.copy_to_host()[1:-1, 1:-1]

    local_energy = np.array(
        [np.sum(a)*DX*DY], dtype=np.float64
    )
    global_energy = np.zeros_like(local_energy)

    ctx["comm"].Allreduce(
        local_energy, global_energy, op=MPI.SUM
    )

    diff = (
        np.sum((a-previous)**2)*DX*DY
        if previous is not None else 0.0
    )

    local = np.array([diff], dtype=np.float64)
    total = np.zeros_like(local)

    ctx["comm"].Allreduce(
        local, total, op=MPI.SUM
    )

    return (
        float(global_energy[0]),
        float(np.sqrt(total[0])),
        a
    )


def run_mode(ctx, mode, dt):
    u, k, s, kmax = initialize_subdomains(ctx)

    out = cuda.to_device(
        np.zeros((NY+2, NX+2), dtype=np.float64)
    )

    buffers = allocate_exchange()
    sb = cuda.stream()
    si = cuda.stream()

    trace = []
    previous = None

    ctx["comm"].Barrier()
    start = MPI.Wtime()

    for step in range(STEPS+1):
        if step % 100 == 0:
            energy, residual, previous = stats(
                ctx, u, previous
            )

            trace.append({
                "step": step,
                "energy": energy,
                "residual": residual
            })

            if ctx["rank"] == 0:
                print(
                    f"{mode} step={step} "
                    f"energy={energy:.15e} "
                    f"residual={residual:.15e}",
                    flush=True
                )

        if step == STEPS:
            break

        u, out = step_overlapped(
            ctx, u, k, s, out, dt,
            buffers, sb, si,
            mode == "overlapped"
        )

    ctx["comm"].Barrier()
    elapsed = MPI.Wtime()-start

    wall = ctx["comm"].allreduce(
        elapsed, op=MPI.MAX
    )

    return u, trace, wall*1000, kmax


def audit_ptx(ctx, u, k, s, dt):
    out = cuda.device_array_like(u)

    launch_region(
        u, k, s, out, dt,
        cuda.default_stream(),
        1, NY, 1, NX, False
    )

    cuda.synchronize()

    sig = list(tiled_stencil_kernel.signatures)[-1]
    asm = tiled_stencil_kernel.inspect_asm(sig)

    counts = {}

    for name in (
        "st.shared.f64",
        "ld.shared.f64",
        "bar.sync"
    ):
        counts[name] = len(
            re.findall(
                r"\b"+re.escape(name)+r"\b", asm
            )
        )

    for prefix in ("st", "ld"):
        counts[prefix+".shared_total"] = len(
            re.findall(
                r"\b"+prefix+r"\.shared(?:\.[\w]+)*",
                asm
            )
        )

    counts["bar.sync_total"] = len(
        re.findall(r"\bbar\.sync\b", asm)
    )

    counts["passed"] = (
        counts["st.shared_total"] > 0
        and counts["ld.shared_total"] > 0
        and counts["bar.sync_total"] > 0
    )

    return counts


def run_solver(username):
    ctx = init_distributed_context(username)

    if ctx["rank"] == 0:
        print(
            "mpi_ranks=4 topology=2x2 "
            "GPU_UUID="+ctx["uuid"]+" "
            "INSTANCE_SEED="+str(ctx["seed"]),
            flush=True
        )

    u, k, s, kmax = initialize_subdomains(ctx)

    kmax = ctx["comm"].allreduce(
        kmax, op=MPI.MAX
    )

    dtmax = DX*DX/(4*kmax)
    dt = 0.8*dtmax

    if ctx["rank"] == 0:
        print(
            f"kappa_max={kmax:.15e} "
            f"dx={DX:.15e} "
            f"dt_max={dtmax:.15e} "
            f"dt={dt:.15e}",
            flush=True
        )

    ptx = audit_ptx(ctx, u, k, s, dt)

    if ctx["rank"] == 0:
        print(
            "PTX_AUDIT "+json.dumps(ptx),
            flush=True
        )

    us, ts, ms, _ = run_mode(
        ctx, "serialized", dt
    )

    uo, to, mo, _ = run_mode(
        ctx, "overlapped", dt
    )

    aa = us.copy_to_host()[1:-1, 1:-1]
    bb = uo.copy_to_host()[1:-1, 1:-1]

    local = float(np.max(np.abs(aa-bb)))

    parity = ctx["comm"].allreduce(
        local, op=MPI.MAX
    )

    if ctx["rank"] == 0:
        manifest = {
            "github_username": username,
            "gpu_uuid": ctx["uuid"],
            "instance_seed": ctx["seed"],
            "mpi_ranks": 4,
            "cartesian_dims": [2, 2],
            "grid": [512, 512],
            "local_padded_shape": [258, 258],
            "steps": STEPS,
            "kappa_max": kmax,
            "dx": DX,
            "dt": dt,
            "dt_max": dtmax,
            "serialized_ms": ms,
            "overlapped_ms": mo,
            "parity_max_diff": parity,
            "ptx_audit": ptx,
            "ptx_audit_passed": ptx["passed"],
            "serialized_energy_trace": ts,
            "overlapped_energy_trace": to
        }

        with open(
            "submission_manifest.json", "w"
        ) as f:
            json.dump(manifest, f, indent=2)

        print(
            "FINAL_MANIFEST "+json.dumps(manifest),
            flush=True
        )

        passed = (
            parity < 1e-12 and ptx["passed"]
        )

        print(
            "RESULT "+(
                "PASS" if passed
                else "CHECK_REQUIRED"
            ),
            flush=True
        )

    ctx["comm"].Barrier()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python distributed_cuda_solver.py "
            "GITHUB_USERNAME"
        )

    run_solver(sys.argv[1])
