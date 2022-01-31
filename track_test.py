import lapsim as ls
from lapsim.ode import CurveLoss

param_lengths = {
    "time": 1,
    "position": 1,
    "velocity": 1,
    "acceleration": 1,
    "loss": 1
}

car = ls.CarStateContainer(param_lengths)
car.acceleration[:] = 10

car_dot = ls.CarStateContainer(param_lengths)

track = ls.track.read_segment_track("track.csv")

ode = ls.ode.OdeList([
    ls.ode.def_ode,
    ls.ode.Drag(),
    CurveLoss(track)])

solver = ls.solver.EulerMaruyama(
    car, car_dot, ode, track)

trajectory = solver.solve()

ls.visual.simple_plots1d(
    trajectory,
    ["position", "velocity", "acceleration", "loss"])

laptime = float(trajectory.time[-1])
print(f'laptime: {laptime:.2f}s')
