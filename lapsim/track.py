import jax.numpy as jnp
import numpy as np
import math


SLALOM_TRACK = np.array([
    [60, 0], [math.pi * 10, -0.1],
    [50, 0], [math.pi * 10, 0.1],
    [50, 0], [math.pi * 10, -0.1],
    [50, 0], [math.pi * 10, 0.1],
    [60, 0]], dtype='f4')


def scale_params(track_params, length):
    old_length = np.sum(track_params[:, 0])
    track_params[:, 0] *= length / old_length
    track_params[:, 1] /= length / old_length
    return track_params


def KappaOfX(track_params):
    x_cumsum = jnp.cumsum(track_params[:, 0])
    n = x_cumsum.shape[0]

    def kappaofx(x):
        idxs = jnp.searchsorted(x_cumsum, x)
        idxs = jnp.minimum(idxs, n - 1)
        kappa = jnp.take(track_params[:, 1], idxs)
        return kappa

    return kappaofx
