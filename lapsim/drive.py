import numpy as np
from . import qtt


def FullGas(sheet):
    g_slice = sheet[qtt.gas.key]

    def driver(car):
        car[g_slice] = 1.

    return driver


def VelDriver(sheet, v_params, track_length):
    x_slice = sheet[qtt.pos1d.key]
    v_slice = sheet[qtt.vel1d.key]
    g_slice = sheet[qtt.gas.key]

    n = v_params.shape[-1]

    def driver(car):
        pos = car[x_slice] / track_length
        i = np.minimum((pos * n).astype(int), n - 1)

        v_target = np.take_along_axis(v_params, i, -1)
        v_target = np.maximum(v_target, 0.)
        do_accel = v_target > car[v_slice]

        car[g_slice] = np.where(do_accel, 1, -1)

    return driver