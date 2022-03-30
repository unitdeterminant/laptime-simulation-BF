import jax.numpy as jnp
import jax

from .solve import solve

# TODO: type hints, docstrings
# crossover: two random permutations then random interpolation


def mutate(key, params, rate=0.5):
    z = jax.random.normal(key, params.shape)
    params = params + z * rate
    params = jnp.maximum(params, 0.)
    return params


def select(params, loss):
    loss_order = jnp.argsort(loss, -1)
    params = params[loss_order]
    params = params[:params.shape[0] // 2]
    return params


def crossover(params):
    params = jnp.concatenate([params, params], 0)
    return params


def EvaluateParams(Car, step, track_length, t_max):

    def eval_params_fn(params):
        batch_dims = list(params.shape[:-1])
        car = solve(
            Car, step, batch_dims,
            params=params)

        t = car.t[..., 0]
        x = car.x[..., 0]

        v_loss = track_length - x + t_max
        f_loss = t

        finished = x >= track_length
        loss = jnp.where(finished, f_loss, v_loss)
        return loss

    return eval_params_fn
