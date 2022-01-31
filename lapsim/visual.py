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


def simple_plots1d(trajectory, keys, save_path="plots"):
    time = trajectory["time"]

    for key in keys:
        plt.xlabel("time")
        plt.ylabel(key)
        plt.plot(time, trajectory[key])

        image_path = os.path.join(save_path, key + ".png")
        ensure_folder(image_path)

        plt.tight_layout()
        plt.savefig(image_path, transparent=True)
        plt.close()
