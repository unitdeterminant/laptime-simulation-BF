class BaseOpt:
    pass


class ConstantAcceleration(BaseOpt):
    def __init__(self, acceleration):
        super(ConstantAcceleration, self).__init__()
        self.acceleration = acceleration

    def apply(self, param_matrix):
        param_matrix["acceleration"][0] = self.acceleration
        return