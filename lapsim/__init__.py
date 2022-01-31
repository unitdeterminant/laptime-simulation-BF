import numpy as np
from . import solver, ode, track, visual


class CarStateContainer:
    def __init__(self, param_lengths, init=None):
        self.param_lengths = param_lengths
        self.param_pos = {}

        i = 0
        for key, v in self.param_lengths.items():
            i = i + v
            self.param_pos[key] = (i - v, i)

        if init is None:
            self.params = np.zeros(i, "f4")
        else:
            self.params = init

        for key in self.param_lengths:
            s, e = self.param_pos[key]
            setattr(self, key, self.params[..., s:e])

    def reset(self):
        self.params[:] = 0.
        return self

    def __getitem__(self, key):
        s, e = self.param_pos[key]
        return self.params[..., s:e]

    def set_param(self, key, value):
        s, e = self.param_pos[key]
        self.params[s:e] = value
