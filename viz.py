import matplotlib.pyplot as plt
import os


def simple_plots1d(trajectory, keys):
    time = trajectory["time"][:, 0]

    for key in keys:
        plt.xlabel("time")
        plt.ylabel(key)

        plt.plot(time, trajectory[key][:, 0])
        plt.savefig(os.path.join("plots", key + ".png"))
        plt.close()
