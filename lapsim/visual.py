import matplotlib.pyplot as plt
import matplotlib as mpl
import os

from . import qtt

mpl.rcParams.update({
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "white",
    "axes.grid": True,
    "grid.color": "white",
    "grid.alpha": 0.1,
    "savefig.transparent": True,
})


def ensure_folder(path):
    foldername = os.path.dirname(path)
    if not os.path.exists(foldername):
        os.makedirs(foldername)


def simple_plots1d(sheet, trajectory, quantities, save_path="plots", dpi=90):
    t = trajectory[sheet[qtt.time.key]]

    for quantity in quantities:
        plt.xlabel("time in [$s$]")
        plt.ylabel(f"{quantity.key} in [${quantity.unit}$]")
        plt.plot(t, trajectory[sheet[quantity.key]])

        image_path = os.path.join(save_path, quantity.key + ".png")
        ensure_folder(image_path)

        plt.tight_layout()
        plt.savefig(image_path, dpi=dpi)
        plt.close()
