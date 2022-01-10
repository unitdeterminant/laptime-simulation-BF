import numpy as np

class BaseSolver:
    pass


class EulerMaruyama(BaseSolver):
    def __init__(self, car, ode, track, opt, dt=1e-1):
        super(EulerMaruyama, self).__init__()
        self.car = car
        self.ode = ode
        self.track = track
        self.opt = opt
        self.dt = dt

        self.trajectory = [self.car.param_matrix.copy()]

    def solve(self):
        finished = False

        while not finished:
            self.ode.apply(self.car.param_matrix)
            self.opt.apply(self.car.param_matrix)

            self.step(self.car.param_matrix)

            finished = self.track.check_finished(self.car.param_matrix)

            self.trajectory.append(self.car.param_matrix.copy())
        
        return np.array(self.trajectory)

    def step(self, param_matrix):
        for i, params in enumerate(param_matrix):
            if i != 0:
                for key in params.dtype.names:
                    param_matrix[key][i - 1] += params[key] * self.dt

        return
