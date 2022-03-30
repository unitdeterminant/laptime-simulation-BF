import jax.tree_util as jtu
import jax.numpy as jnp
import jax


def concatenate_cars(cars):
    """
    Concats cars along a new axis at the front.

    Parameters
    ----------
    cars : itereable of instances of a car class
        the cars to be concatenated

    Returns
    -------
    concat : a instance of the car class
        the concatenated cars from the given list

    Notes
    -----
    This function is not optimized for performance and should not be
    used in performance critical applications (about half a second
    for reasonable inputs).

    It is probably not a goot idea performace wise to XLA compile 
    this function if the inputs are to large e.g. a whole trajectory,
    because that can take up to half a minute even for reasonable
    inputs.

    """

    cars = [expand_dims_car(c) for c in cars]
    return jtu.tree_map(
        lambda *x: jnp.concatenate(x, 0),
        cars[0], *cars[1:])


@jax.jit
def expand_dims_car(car, axis=0):
    """
    Adds another axis to all parameters of a given car

    Parameters
    ----------
    car : instance of a car class
        the car, whose parameters are to be expanded

    axis : int, optional
        see numpy.expand_dims

    """
    return jtu.tree_map(
        lambda x: jnp.expand_dims(x, axis), car)


def fuse_step(step, nfuse):

    # not recomended to do more than 12 steps

    @jax.jit
    def fused_step(car, dkwargs):
        for _ in range(nfuse):
            car, abort = step(car, dkwargs)
        return car, abort

    return fused_step


def EulerStep(Car, ode, abort_cond, driver, dt=1e-2):

    @jax.jit
    def step_fn(car, dkwargs):
        car = driver(car, **dkwargs)

        car_dot = Car.zeros(car.batch_dims)
        car_dot = ode(car, car_dot)

        abort = abort_cond(car)
        do_update = jnp.logical_not(abort)

        car += car_dot * dt * do_update
        return car, abort

    return step_fn


def solve(
        Car,
        step,
        batch_dims=[],
        **dkwargs):

    abort = False
    car = Car.zeros(batch_dims)

    while not jnp.all(abort):
        car, abort = step(car, dkwargs)

    return car


def solve_trajectory(
        Car,
        step,
        batch_dims=[],
        **dkwargs):

    abort = False
    car = Car.zeros(batch_dims)
    trajectory = [car]

    while not jnp.all(abort):
        car, abort = step(car, dkwargs)
        trajectory.append(car)

    return concatenate_cars(trajectory)
