
import multiprocessing as mp
import time
import csv
import os
import statistics

def work(args):
    start, count = args
    total = 0.0
    for i in range(start, start + count):
        x = i * 0.000001
        total += (x * x + 1) / (x + 1)
    return total

def main():
    max_cores = os.cpu_count() or 2
    counts = sorted(set([1, 2, 4, 6, 8, max_cores]))
    counts = [p for p in counts if p <= max_cores]
    k = 1000000

    print("CPU cores:", max_cores)
    print("Work per process:", k)

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cores", "run", "time"])

        for p in counts:
            times = []
            for run in range(1, 6):
                tasks = [(i * k, k) for i in range(p)]

                start = time.perf_counter()
                with mp.Pool(processes=p) as pool:
                    pool.map(work, tasks)
                elapsed = time.perf_counter() - start

                times.append(elapsed)
                writer.writerow([p, run, elapsed])

            print(
                f"Cores={p}, Mean={statistics.mean(times):.4f}s, "
                f"Std={statistics.stdev(times):.4f}s"
            )

if __name__ == "__main__":
    mp.freeze_support()
    main()

