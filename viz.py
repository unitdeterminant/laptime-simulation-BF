import matplotlib.pyplot as plt
import matplotlib as mpl
import os


mpl.rcParams.update({
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "white"})


def ensure_folder(path):
    foldername = os.path.dirname(path)
    if not os.path.exists(foldername):
        os.makedirs(foldername)


def simple_plots1d(trajectory, keys):
    time = trajectory["time"][:, 0]

    for key in keys:
        plt.xlabel("time")
        plt.ylabel(key)
        plt.plot(time, trajectory[key][:, 0])
        
        path = os.path.join("plots", key + ".png")
        ensure_folder(path)

        plt.tight_layout()
        plt.savefig(path, transparent=True)
        plt.close()
