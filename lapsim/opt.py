import numpy as np
from . import drive, solve, qtt


def evaluate_params(
        sheet,
        v_params,
        track_length,
        ode,
        abort_cond,
        t_max):

    drivers = drive.VelDriver(sheet, v_params, track_length)

    batch_dims = list(v_params.shape[:-1])
    trajectory = solve.euler_solve(
        sheet, drivers, ode, abort_cond,
        batch_dims, False)

    t = trajectory[sheet[qtt.time.key]][0, ..., 0]
    x = trajectory[sheet[qtt.pos1d.key]][0, ..., 0]

    v_loss = track_length - x + t_max
    f_loss = t

    finished = x >= track_length
    loss = np.where(finished, f_loss, v_loss)
    return loss


def mutate(v_params, rate=0.5):
    z = np.random.normal(size=v_params.shape)
    v_params = v_params + z * rate
    v_params = np.maximum(v_params, 0.)
    return v_params


def select(v_params, loss):
    loss_order = np.argsort(loss)
    v_params = v_params[loss_order]
    v_params = v_params[:v_params.shape[0] // 2]
    return v_params


def crossover(v_params):
    v_params = np.concatenate([v_params, v_params], 0)
    return v_params
