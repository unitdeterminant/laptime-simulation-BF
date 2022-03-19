import numpy as np
from . import qtt


def AbortCondList(abort_conds):
    def abort_cond(car):
        aborts = [fn(car) for fn in abort_conds]
        aborts = np.concatenate(aborts, -1)
        return np.any(aborts, -1)

    return abort_cond


def KappaMaxVel(
        sheet, kappaofx, *,
        scale_max=0.5, eps=1e-6):

    x_slice = sheet[qtt.pos1d.key]
    v_slice = sheet[qtt.vel1d.key]

    def abort_cond(car):
        x, v = car[x_slice], car[v_slice]
        kappa = np.abs(kappaofx(x))

        vel_max = scale_max / (kappa + eps)
        abort = v > vel_max
        return abort

    return abort_cond


def Finish(sheet, track_length):
    x_slice = sheet[qtt.pos1d.key]

    def abort_cond(car):
        abort = car[x_slice] >= track_length
        return abort

    return abort_cond


def TimeLimit(sheet, t_max):
    t_slice = sheet[qtt.time.key]

    def abort_cond(car):
        abort = car[t_slice] >= t_max
        return abort

    return abort_cond
