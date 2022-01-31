import lapsim as ls

param_lengths = {
    "time": 1,
    "position": 1,
    "velocity": 1,
    "acceleration": 1}

car = ls.CarStateContainer(param_lengths)
car.acceleration[:] = 10

car_dot = ls.CarStateContainer(param_lengths)

track = ls.track.LineTrack(402.336)

ode = ls.ode.OdeList([ls.ode.def_ode, ls.ode.Drag()])

solver = ls.solver.EulerMaruyama(
    car, car_dot, ode, track)

trajectory = solver.solve()

ls.visual.simple_plots1d(
    trajectory,
    ["position", "velocity", "acceleration"])

laptime = float(trajectory.time[-1])
print(f'laptime: {laptime:.2f}s')
