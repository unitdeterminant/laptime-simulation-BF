import numpy as np


class BaseCar:
    pass


class CarMinimal(BaseCar):
    def __init__(self):
        super(CarMinimal, self).__init__()

        self.dtype = np.dtype(
            [
                ("time", "f4", (1,)),
                ("position", "f4", (1,)),
                ("velocity", "f4", (1,)),
                ("acceleration", "f4", (1,)),
            ]
        )

        self.zeros = np.zeros(3, dtype=self.dtype)
        self.param_matrix = self.zeros.copy()

