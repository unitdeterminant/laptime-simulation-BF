import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import rcParams
from dataclasses import fields


def set_darkmode():
    """
    Sets a custom 'darkmode' style for matplotlib, that will affect 
    all subsequent plots.

    """

    rcParams.update({
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "axes.edgecolor": "white",
        "axes.grid": True,
        "grid.color": "white",
        "grid.alpha": 0.15,
        "figure.autolayout": True,
        "savefig.transparent": True})


def ensure_folder(path):
    """
    Ensures, that all folders that lead to path exist
    and creates them if necessary.

    Parameters
    ----------
    path : string
        path to a file (that may not have been created yet)

    """

    foldername = os.path.dirname(path)
    if not os.path.exists(foldername):
        os.makedirs(foldername)


def simple_1dplots(
        car,
        save_dir="plots",
        x_axis="time",
        exclude=[],
        dpi=90):
    """
    Creates line plots for all 1-dimensional quantities of the car.

    Parameters
    ----------
    car : instance of car class
        holds the trajectory to be plotted

    save_dir : string
        path to directory, where the plots will be saved

    x_axis : string
        name of the quantity on the x axis

    exlude : list of strings
        name of quantities not to plot

    dpi : strictly positive int
        dpi setting for saving the plots

    """

    y_data = []

    for f in fields(car):

        if np.prod(f.type.shape) == 1:
            values = car.__dict__[f.name]
            values = values[tuple([...] + [0] * (values.ndim - 1))]

            label = f.type.axis_label
            q_name = f.type.name

            if q_name == x_axis:
                x_values = values
                x_label = label

            else:
                if not q_name in exclude:
                    y_data.append((q_name, label, values))

    for q_name, y_label, y_values in y_data:
        plt.plot(x_values, y_values)

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        image_path = os.path.join(save_dir, q_name + ".png")
        ensure_folder(image_path)

        plt.savefig(image_path, dpi=dpi)
        plt.close()
