import numpy as np
from . import qtt


def OdeList(ode_list):
    def ode(car, car_dot):
        for fn in ode_list:
            fn(car, car_dot)

    return ode


def BasicPhysics1D(sheet):
    t_slice = sheet[qtt.time.key]
    x_slice = sheet[qtt.pos1d.key]
    v_slice = sheet[qtt.vel1d.key]
    a_slice = sheet[qtt.acc1d.key]

    def ode(car, car_dot):
        car_dot[t_slice] = 1
        car_dot[x_slice] = car[v_slice]
        car_dot[v_slice] = car[a_slice]

    return ode


def Drag1D(sheet, *, scale=1e-3):
    v_slice = sheet[qtt.vel1d.key]

    def ode(car, car_dot):
        car_dot[v_slice] -= scale * car[v_slice] ** 2

    return ode


def NoBackward1D(sheet):
    x_slice = sheet[qtt.pos1d.key]
    v_slice = sheet[qtt.vel1d.key]

    def ode(car, car_dot):
        car_dot[x_slice] = np.maximum(0., car_dot[x_slice])
        car[v_slice] = np.maximum(0., car[v_slice])
        return car_dot

    return ode


def GastoAcceleration1D(sheet, *, acc_max=10.):
    a_slice = sheet[qtt.acc1d.key]
    g_slice = sheet[qtt.gas.key]

    def ode(car, car_dot):
        gas = np.clip(car[g_slice], -1, 1)
        car[a_slice] = gas * acc_max
        return car_dot

    return ode
