import matplotlib.pyplot as plt
import lapsim as ls
import numpy as np


track_params = ls.track.scale_params(ls.track.SLALOM_TRACK, 1e3)
kappaofx, track_length = ls.track.KappaOfX(track_params)

sheet = ls.qtt.create_sheet([
    ls.qtt.time,
    ls.qtt.gas,
    ls.qtt.pos1d,
    ls.qtt.vel1d,
    ls.qtt.acc1d,
])

ode = ls.ode.OdeList([
    ls.ode.BasicPhysics1D(sheet),
    ls.ode.Drag1D(sheet),
    ls.ode.GastoAcceleration1D(sheet),
    ls.ode.NoBackward1D(sheet),
])

t_max = 100
abort_cond = ls.abort.AbortCondList([
    ls.abort.KappaMaxVel(sheet, kappaofx),
    ls.abort.Finish(sheet, track_length),
    ls.abort.TimeLimit(sheet, t_max),
])

BATCH_SIZE = 1024

v_params = np.random.normal(size=[BATCH_SIZE, 32])
v_params = np.abs(v_params) * 5

loss_list = []

for i in range(1024):

    loss = ls.opt.evaluate_params(
        sheet, v_params, track_length, ode, abort_cond, t_max)

    loss_list.append(loss.min())

    print(f"loss: {loss_list[-1]:.2f} at iteration {i + 1}")

    v_params = ls.opt.select(v_params, loss)

    if i % 128 == 0:
        driver = ls.drive.VelDriver(sheet, v_params[0], track_length)

        trajectory = ls.solve.euler_solve(
            sheet, driver, ode, abort_cond,
            batch_dims=[], record=True, dt=5e-2)

        ls.visual.simple_plots1d(
            sheet, trajectory,
            [ls.qtt.pos1d, ls.qtt.vel1d, ls.qtt.acc1d])

        plt.hist(loss, 32)
        plt.xlabel('loss')
        plt.ylabel('#')
        plt.tight_layout()
        plt.savefig("plots/loss_hist")
        plt.close()

        plt.plot(loss_list)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.savefig("plots/opt.png")
        plt.close()


    v_params = ls.opt.crossover(v_params)
    v_params = ls.opt.mutate(v_params)


driver = ls.drive.VelDriver(sheet, v_params[0], track_length)

trajectory = ls.solve.euler_solve(
    sheet, driver, ode, abort_cond,
    batch_dims=[], record=True, dt=1e-3)

ls.visual.simple_plots1d(
    sheet, trajectory,
    [ls.qtt.pos1d, ls.qtt.vel1d, ls.qtt.acc1d],
    dpi=256)

plt.plot(loss_list)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.yscale('log')
plt.savefig("plots/opt.png", dpi=256)
plt.close()
