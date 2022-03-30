import lapsim as ls

track_length = 1e3
abort_cond = ls.abort.AbortCondList(
    ls.abort.Finished(track_length))

ode = ls.ode.ODEList(
    ls.ode.physics,
    ls.ode.drag1d,
    ls.ode.gas_to_acceleration1d,
    ls.ode.no_backward1d)

driver = ls.drive.full_gas

step = ls.solve.fuse_step(ls.solve.EulerStep(
    ls.car.Car1D, ode, abort_cond, driver, 1e-3), 8)

trajectory = ls.solve.solve_trajectory(
    ls.car.Car1D, step)

ls.visual.set_darkmode()
ls.visual.simple_1dplots(trajectory)
