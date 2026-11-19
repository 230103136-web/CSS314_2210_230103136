# Q1-Q6: Lab Discussion

Q1. MPI decomposition
The 512x512 grid is decomposed across four MPI ranks
in a 2x2 Cartesian topology. Halo cells support
communication between neighboring subdomains.

Q2. CUDA implementation
GPU kernels compute the stencil update using
double-precision values. Shared memory and
synchronization are inspected through PTX.

Q3. Communication overlap
The serialized implementation takes 4321.11 ms,
while the overlapped implementation takes 1260.22 ms.
The measured speedup is approximately 3.43x.
These timings may include initialization overhead.

Q4. Numerical correctness
Both implementations produce identical energy
and residual values at steps 0, 100, 200 and 300.
The maximum difference is 0.0.

Q5. False sharing
Stride 1 takes 634.08 ms, compared with
162.57 ms for stride 8 and 196.84 ms for stride 16.
Closely spaced counters can cause cache-line contention.

Q6. Compiler verification
The original Numba implementation optimized repeated
memory updates into one operation. A C volatile
implementation preserves memory loads and stores
inside the loop. Assembly inspection confirms
the loop structure. LLVM IR verification remains
a separate requirement if explicitly requested.
