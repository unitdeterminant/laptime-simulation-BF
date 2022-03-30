import jax.numpy as jnp


# TODO: type hints


def physics(car, car_dot):
    """
    Implements basic physics.
    d/dt t = 1
    d/dt v = x
    d/dt a = v

    Parameters
    ----------
    car : instance of a car
        car holding the current values

    car_dot : instance of a car
        car holding the current time derivative

    Returns
    -------
    car_dot : instance of a car
        an updated instance the time derivative car

    """

    car_dot.t = jnp.ones_like(car_dot.t)
    car_dot.x = car.v
    return car_dot


def no_backward1d(car, car_dot):
    """
    Clips d/dt x to be greater or equal to zero.

    Parameters
    ----------
    car : instance of a car
        car holding the current values

    car_dot : instance of a car
        car holding the current time derivative

    Returns
    -------
    car_dot : instance of a car
        an updated instance the time derivative car

    """

    car_dot.x = jnp.maximum(car_dot.x, 0.)
    return car_dot


def drag1d(car, car_dot, *, scale=1e-3):
    """
    Minimal implementation of drag, where the force is equal to the
    negative velocity squared times some constant `scale`.

    Parameters
    ----------
    car : instance of a car
        car holding the current values

    car_dot : instance of a car
        car holding the current time derivative

    scale : float, optional
        the strenght of the drag, can be calculated by using the drag
        coefficent and the car's frontal area.

    Returns
    -------
    car_dot : instance of a car
        an updated instance the time derivative car

    """
    car_dot.v -= scale * car.v ** 2
    return car_dot


def gas_to_acceleration1d(
        car, car_dot, *,
        acc_max=10., brake_max=10.):
    """
    Minimal implementation of drag, where the force is equal to the
    negative velocity squared times some constant `scale`.

    Parameters
    ----------
    car : instance of a car
        car holding the current values

    car_dot : instance of a car
        car holding the current time derivative

    acc_max : float, optional
        the maximum ammount of acceleration

    brake_max : float, optional
        the maximum ammount of negative acceleration, the given
        parameter should still be positive

    Returns
    -------
    car_dot : instance of a car
        an updated instance the time derivative car

    """

    g = jnp.clip(car.g, -1, 1)
    g = jnp.maximum(g, g * acc_max)
    g = jnp.minimum(g, g * brake_max)

    car_dot.v += g
    return car_dot


def ODEList(*odes):
    """
    Bundles a list of odes together into one ode, which calls them
    one after the other.

    Parameters
    ----------
    ode_list : a list of odes
        odes are called in exacly that order

    Returns
    -------
    ode_fn : an ode
        one function, that calls all given odes
    
    """

    def ode_fn(car, car_dot):
        for ode in odes:
            car_dot = ode(car, car_dot)

        return car_dot

    return ode_fn
