import matplotlib.pyplot as plt
import numpy as np
import h5py
import math

from matplotlib.backends.backend_agg import FigureCanvasAgg

CUBIC_BERNSTEIN = np.array(
    [[1, -3, 3, -1],
     [0, 3, -6, 3],
     [0, 0, 3, -3],
     [0, 0, 0, 1]]).T


def cubic_bernstein(t):
    ts = np.stack([t ** 0, t ** 1, t ** 2, t ** 3], -1)
    return ts @ CUBIC_BERNSTEIN


def cubic_bezier(t, p1, c1, c2, p2):
    c = cubic_bernstein(t)
    return c @ np.array([p1, c1, c2, p2])


def bezier_interpolate(points, n_interp=8192):
    n = len(points)
    n_interp //= n

    controls = get_controls(points)
    results = []

    for i in range(n):
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]

        c1 = controls[((2 * i) + 3) % (2 * n)]
        c2 = controls[((2 * i) + 4) % (2 * n)]

        ts = np.linspace(0, 1, n_interp)

        for t in ts:
            r = cubic_bezier(t, p1, c1, c2, p2)
            results.append(r)

    return np.array(results)


def normalize(x):
    return x / np.linalg.norm(x)


def angle_difference(x, y):
    s = normalize(x) @ normalize(y)
    return np.arccos(s)


def angle_to_2d(ang):
    return np.array([np.sin(ang), np.cos(ang)])


def random_points(
        n_random=12,
        ang_min=1, ang_max=1.5,
        dia_min=1, dia_max=5):

    angles = np.random.uniform(ang_min, ang_max, n_random)
    dias = np.random.uniform(dia_min, dia_max, n_random)

    angles = np.cumsum(np.abs(angles))
    angles = angles / angles.max() * math.pi * 2

    points = np.stack([np.cos(angles), np.sin(angles)], -1)
    return points * dias[:, None]


def get_controls(points):
    n = len(points)
    results = []

    for i in range(n):
        a = points[(i - 1) % n]
        b = points[(i + 0) % n]
        c = points[(i + 1) % n]

        x = a - b
        y = c - b

        alpha = angle_difference(x, y) / 2

        ang = np.arctan2(x[0], x[1]) + alpha - math.pi / 2
        steepness = np.random.uniform(0.5, 1)

        v = angle_to_2d(ang) * steepness + b
        results.append(v)

        ang = np.arctan2(x[0], x[1]) + alpha + math.pi / 2
        steepness = np.random.uniform(0.5, 1)

        v = angle_to_2d(ang) * steepness + b
        results.append(v)

    return np.array(results)


def get_random_track(
        n_random=12,
        ang_min=1, ang_max=1.5,
        dia_min=1, dia_max=3,
        n_interp=8192, size=1024,
        track_width_multiplier=2):

    points = random_points(n_random, ang_min, ang_max, dia_min, dia_max)
    interp = bezier_interpolate(points, n_interp)

    dpi = 256
    track_width_multiplier = size * track_width_multiplier / 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    plt.plot(interp[:, 0], interp[:, 1], "k", linewidth=track_width_multiplier)

    plt.axis("equal")
    plt.axis("off")

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    raster = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')

    points = plt.gca().transData.transform(points)
    points = points.astype("uint16")

    plt.close()

    raster = raster.reshape([size, size, 3])
    raster = raster[..., 0]
    raster = np.logical_not(raster.astype(bool))

    raster = raster[::-1]
    points = points[:, ::-1]
    return raster, points


def write_track(raster, points, path="track.h5"):
    with h5py.File(path, "w") as file:
        file["raster"] = raster
        file["points"] = points


def read_track(path="track.h5"):
    with h5py.File(path, "r") as file:
        raster = np.array(file["raster"])
        points = np.array(file["points"])
        return raster, points


def show_track(raster, points):
    plt.matshow(-raster.astype("float32"), cmap="gray")

    plt.plot(
        points[:, 1], points[:, 0], "o",
        color="tab:orange",
        label="checkpoints")

    plt.axis("equal")
    plt.axis("off")

    plt.legend()
    plt.show()


def check_randomization(n=8):
    rows = []
    for _ in range(n):

        row = []
        for _ in range(n):
            raster, _ = get_random_track(size=256)
            raster = -raster.astype("float32")

            row.append(raster)

        row = np.concatenate(row, 0)
        rows.append(row)

    rows = np.concatenate(rows, 1)

    plt.matshow(rows, cmap="gray")

    plt.axis("equal")
    plt.axis("off")

    plt.savefig("tracks.png", dpi=512, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    check_randomization()
