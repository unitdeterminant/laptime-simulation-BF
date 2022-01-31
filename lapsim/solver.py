import numpy as np
from .__init__ import CarStateContainer


class EulerMaruyama:
    def __init__(self, car, car_dot, ode, track, dt=1e-1):
        super(EulerMaruyama, self).__init__()
        self.car = car
        self.car_dot = car_dot
        self.ode = ode
        self.track = track
        self.dt = dt

    def solve(self):
        finished = False
        trajectory = [self.car.params.copy()]

        while not finished:
            self.car_dot.reset()
            self.ode(self.car, self.car_dot)

            self.step(self.car, self.car_dot)
            finished = self.track.check_finished(self.car.position)

            trajectory.append(self.car.params.copy())

        trajectory = np.array(trajectory)
        trajectory = CarStateContainer(self.car.param_lengths, trajectory)
        return trajectory

    def step(self, car, car_dot):
        car.params += car_dot.params * self.dt
