# Cache Line and False Sharing Mechanics

**GitHub username:** 230103136-web  
**CPU:** Apple M4  
**Logical cores:** 10  
**Cache line size:** 128 bytes  
**Python:** 3.14.3  
**Threads:** 4  
**Iterations per thread:** 60,000,000  

## Q1. MESI Protocol and False Sharing

False sharing happens when independent threads update different variables located in the same cache line.

With stride 1, the four counters are only 8 bytes apart. All four counters fit inside one 128-byte cache line on the Apple M4.

When one core writes to its counter, it needs ownership of the corresponding cache line. Under a MESI-style coherence model, another core's cached copy may become Invalid. When another core attempts to write to its own counter in the same line, ownership must transfer again.

This repeated ownership transfer causes cache-coherence traffic and can delay execution, even though the threads are updating different array elements.

## Q2. Benchmark Results

The following results were measured on the Apple M4 using four concurrent threads.

| Layout | Spacing | Median time (ns) | ns/update |
|---|---:|---:|---:|
| Stride 1 | 8 B | 79,487,666 | 0.331199 |
| Stride 4 | 32 B | 74,521,292 | 0.310505 |
| Stride 8 | 64 B | 30,347,084 | 0.126446 |
| Stride 16 | 128 B | 28,348,959 | 0.118121 |
| Private baseline | Register-local | 66,375 | 0.000277 |

The slowdown ratio between stride 1 and stride 8 was 2.6193x.

The slowdown ratio between stride 1 and stride 16 was approximately 2.80x.

Stride 16 produced the lowest measured elapsed time among the shared-buffer layouts. The private baseline was substantially faster because it performs the accumulation locally and writes the final value only once.

These are median results from three benchmark repetitions per layout. Timing differences may also be affected by scheduling, CPU frequency, and system activity.

## Q3. Stride 8 vs. Stride 16

The Apple M4 reports a cache line size of 128 bytes.

Stride 8 places counters 64 bytes apart. Therefore, some counters can still share a physical cache line.

Stride 16 places counters 128 bytes apart, separating their cache lines when the buffer is correctly aligned.

Stride 8 took 30,347,084 ns, while stride 16 took 28,348,959 ns. Stride 8 was approximately 1.07 times slower in this run.

Adjacent-line prefetching and other microarchitectural effects may influence performance, but these measurements alone do not establish whether secondary prefetch-related invalidations occurred.

## Q4. Compiler Optimization and LLVM Verification

The initial implementation was optimized by the compiler. It completed hundreds of millions of logical updates in only tens of microseconds, indicating that repeated memory operations were not being measured as intended.

The corrected implementation uses an LLVM compiler memory barrier around each native read-modify-write operation.

LLVM inspection showed the following operations in the loop:

- A side-effecting compiler memory barrier
- A native `load i64`
- A native `store i64`
- Another compiler memory barrier
- A conditional branch returning to the loop

The LLVM audit reported one static `store i64` instruction in the loop body. This instruction executes repeatedly; the static instruction count is not the total number of dynamic stores.

The number of volatile stores was zero. The implementation uses compiler memory barriers rather than LLVM volatile stores.

All counter parity checks passed, and the final automated verification returned True.

## Q5. GIL and Real Parallel Execution

In standard CPython, CPU-bound Python threads are normally restricted by the Global Interpreter Lock (GIL), which prevents simultaneous execution of Python bytecode by multiple threads.

The experiment uses Numba `@njit(nogil=True)` to execute compiled native code without holding the GIL during the worker kernel.

This allows the four threads to execute concurrently and makes cache-line contention observable.

The experiment used Python threading with native Numba kernels. CPU affinity was unavailable through the standard Linux affinity interface on macOS, so individual worker threads were not explicitly pinned to specific cores.

## Conclusion

The experiment demonstrated that memory layout can significantly affect concurrent counter-update performance.

On the Apple M4, stride 16 provided the best measured shared-buffer performance because it matched the detected 128-byte cache line size.

The results also showed why LLVM inspection is important: an optimized arithmetic loop may produce correct final values without performing the intended memory operations on every iteration.

All tested layouts produced matching final counters. The final automated verification passed.

The results should be interpreted as measurements from this specific Apple M4 system rather than universal performance ratios.
