import jax.numpy as jnp
import jax

# TODO: type hints, docstrings, refactor vparam_driver (api changes?)


def full_gas(car):
    car.g = jnp.ones_like(car.g)
    return car


def VParamDriver(track_length):
    def driver_fn(car, params):
        n = params.shape[-1]
        pos = car.x / track_length
        i = jnp.minimum((pos * n).astype(int), n - 1)

        old_shape = i.shape
        new_shape = i.shape[i.ndim - params.ndim:-1] + (-1,)
        i = i.reshape(new_shape)

        v_target = jnp.take_along_axis(params, i, -1)
        v_target = jnp.maximum(v_target, 0.)
        v_target = v_target.reshape(old_shape)

        do_accel = v_target > car.v
        car.g = jnp.where(do_accel, 1, -1)
        return car

    return driver_fn
