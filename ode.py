class BaseODE:
    pass
    # def __init__(self, car):
    #     super(Position, self).__init__()
    #     self.car = car


class Velocity(BaseODE):
    def apply(self, param_matrix):
        position_dot = param_matrix["velocity"][0]
        param_matrix["position"][1] = position_dot
        return


class Acceleration(BaseODE):
    def apply(self, param_matrix):
        velocity_dot = param_matrix["acceleration"][0]
        param_matrix["velocity"][1] = velocity_dot
        return

class Time(BaseODE):
    def apply(self, param_matrix):
        param_matrix["time"][1] = 1
        return


class ListODE(BaseODE):
    def __init__(self, ode_list):
        super(ListODE, self).__init__()
        self.ode_list = ode_list

    def apply(self, param_matrix):
        for ode in self.ode_list:
            ode.apply(param_matrix)
        return
