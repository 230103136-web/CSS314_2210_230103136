# Hybrid MPI-CUDA Distributed Stencil

Student ID: 230103136
GitHub: 230103136-web

## Configuration
- MPI ranks: 4
- Topology: 2x2
- Global grid: 512x512
- Simulation steps: 300
- GPU: NVIDIA Tesla T4 (Colab)

## Results
- Serialized: 4321.11 ms
- Overlapped: 1260.22 ms
- Maximum difference: 0.0
- GPU parity: PASS
- PTX audit: PASS
- CPU assembly audit: PASS
- False-sharing slowdown: 3.90x

## CPU benchmark
The C volatile function performs 40 million
read-modify-write operations per rank.

Stride 1: 634.08 ms
Stride 8: 162.57 ms
Stride 16: 196.84 ms

Note: CPU audit verifies assembly, not LLVM IR.
