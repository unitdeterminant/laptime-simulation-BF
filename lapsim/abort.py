import jax.numpy as jnp

# TODO: type hints, docstrings


def AbortCondList(*abort_conds):
    def abort_cond_fn(car):
        aborted = [ac(car) for ac in abort_conds]
        aborted = jnp.stack(aborted, 0)
        return jnp.any(aborted, 0)

    return abort_cond_fn


def Finished(track_length):
    def abort_cond_fn(car):
        return car.x >= track_length

    return abort_cond_fn


def MaxTime(t_max):
    def abort_cond_fn(car):
        return car.t >= t_max

    return abort_cond_fn


def KappaMaxVel(kappaofx, *, scale=0.5, eps=1e-6):
    def abort_cond_fn(car):
        kappa = jnp.abs(kappaofx(car.x))
        vel_max = scale / (kappa + eps)
        return car.v > vel_max

    return abort_cond_fn
