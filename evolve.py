import matplotlib.pyplot as plt
import lapsim as ls
import numpy as np

import jax.numpy as jnp
import jax

track_length = 1e3
t_max = 100

batch_size = 32768
n_params = 32

n_iters = 1024

key = jax.random.PRNGKey(0)
ls.visual.set_darkmode()

track = ls.track.SLALOM_TRACK
track = ls.track.scale_params(track, track_length)
kappaofx = ls.track.KappaOfX(track)

abort_cond = ls.abort.AbortCondList(
    ls.abort.Finished(track_length),
    ls.abort.MaxTime(t_max),
    ls.abort.KappaMaxVel(kappaofx))

ode = ls.ode.ODEList(
    ls.ode.physics,
    ls.ode.drag1d,
    ls.ode.gas_to_acceleration1d,
    ls.ode.no_backward1d)

driver = ls.drive.VParamDriver(track_length)

step_big = ls.solve.fuse_step(ls.solve.EulerStep(
    ls.car.Car1D, ode, abort_cond, driver, 1e-1), 8)

step_small = ls.solve.fuse_step(ls.solve.EulerStep(
    ls.car.Car1D, ode, abort_cond, driver, 1e-2), 8)

evaluate = ls.opt.EvaluateParams(
    ls.car.Car1D, step_big, track_length, t_max)

key, subkey = jax.random.split(key)

params = jax.random.normal(subkey, [batch_size, n_params])
params = jnp.abs(params) * 5

loss_list = []

for i in range(n_iters + 1):
    loss = evaluate(params)
    loss_list.append(loss.min())

    print(f"loss: {loss_list[-1]:.2f} at iteration {i + 1}")

    params = ls.opt.select(params, loss)

    if i % 128 == 0:
        plt.hist(np.array(loss), 32)
        plt.savefig("plots/loss_hist.png")
        plt.close()

        trajectory = ls.solve.solve_trajectory(
            ls.car.Car1D, step_small, params=params[0])

        ls.visual.simple_1dplots(trajectory)

    key, subkey = jax.random.split(key)
    params = ls.opt.crossover(params)
    params = ls.opt.mutate(key, params)


plt.plot(loss_list)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.yscale('log')
plt.savefig("plots/opt.png")
plt.close()
