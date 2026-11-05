
import csv
import matplotlib.pyplot as plt

cores = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
alpha = 0.04

amdahl = []
gustafson = []

for p in cores:
    a = 1 / (alpha + (1 - alpha) / p)
    g = p - alpha * (p - 1)

    amdahl.append(a)
    gustafson.append(g)

    print(f"Cores: {p}, Amdahl: {a:.2f}, Gustafson: {g:.2f}")

with open("task21_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Cores", "Amdahl", "Gustafson"])

    for p, a, g in zip(cores, amdahl, gustafson):
        writer.writerow([p, a, g])

plt.plot(cores, amdahl, marker="o", label="Amdahl")
plt.plot(cores, gustafson, marker="s", label="Gustafson")
plt.xlabel("Number of Cores")
plt.ylabel("Speedup")
plt.title("Amdahl vs Gustafson")
plt.legend()
plt.grid(True)
plt.savefig("task21_plot.png")
plt.show()
