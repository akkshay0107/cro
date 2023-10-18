import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def display_convergence(filepath,max_iter=None):
    with open(filepath, 'r') as f:
        min_pe = np.array([float(x) for x in f.read().split(",")])
    multiplier = 300 / len(min_pe)
    if max_iter == None:
        max_iter = len(min_pe)
    epoch = [multiplier*i for i in range(max_iter)]
    for i in range(len(min_pe)):
        if min_pe[i] == min_pe[-1]:
            print(multiplier*i)
            break
    sns.set_theme()
    plt.plot(epoch, min_pe[:max_iter])
    plt.xlabel('Time (s)')
    plt.ylabel('Total Delivery Time')
    plt.show()

if __name__ == "__main__":
    display_convergence(
        "/home/pc/main/rust/network-pso/logs/2023-10-08T19:20:36.940270371+00:00.log",
    )