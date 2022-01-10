from solver import *
from track import *
from opt import *
from ode import *
from car import *
from viz import *


car = CarMinimal()
track = LineTrack(100)

ode = ListODE([Velocity(), Acceleration(), Time()])

opt = ConstantAcceleration(9.82)

solver = EulerMaruyama(car, ode, track, opt)

trajectory = solver.solve()

simple_plots1d(trajectory, ["position", "velocity", "acceleration"])

lap_time = trajectory["time"][-1, :, 0]
print(lap_time)
