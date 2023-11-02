import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def display_convergence(filepath,max_iter):
    with open(filepath, 'r') as f:
        min_pe = np.array([float(x) for x in f.read().split(",")])
    multiplier = 300 / len(min_pe)
    epoch = [multiplier*i for i in range(max_iter)]
    sns.set_theme()
    plt.plot(epoch, min_pe[:max_iter])
    plt.xlabel('Time (s)')
    plt.ylabel('Total Delivery Time')
    plt.show()

if __name__ == "__main__":
    display_convergence(
        "/home/pc/main/rust/network-cro/logs/2023-10-02T09:53:04.861753730+00:00.log",
        300000
    )