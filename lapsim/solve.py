import numpy as np


def euler_solve(
        sheet,
        driver,
        ode,
        abort_cond,
        batch_dims=[],
        record=False,
        dt=0.1):

    car_shape = batch_dims + [sheet["ndim"]]
    car = np.zeros(car_shape)

    trajectory = []
    aborted = abort_cond(car)

    while not np.all(aborted):
        if record:
            trajectory.append(car)

        car_dot = np.zeros(car_shape)

        ode(car, car_dot)
        driver(car)

        mask = np.logical_not(aborted)[..., None]
        car = car + car_dot * dt * mask.astype("f4")

        aborted = abort_cond(car)

    trajectory.append(car)
    trajectory = np.array(trajectory)
    return trajectory
