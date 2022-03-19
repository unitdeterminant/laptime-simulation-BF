class Quantity:
    def __init__(
            self,
            key,
            ndim,
            unit=None,
            driver=False):

        self.key = key
        self.ndim = ndim
        self.unit = unit
        self.driver = driver

    def __repr__(self):
        return f"Quantity({self.key} in [{self.unit}])"


def create_sheet(quantity_list):
    sheet, i = {}, 0

    for q in quantity_list:
        sheet[q.key] = (..., slice(i, i + q.ndim))
        i += i + q.ndim

    sheet["ndim"] = i
    return sheet


time = Quantity("time", 1, "s")
gas = Quantity("gas", 1, driver=True)

pos1d = Quantity("pos1d", 1, "m")
vel1d = Quantity("vel1d", 1, "m/s")
acc1d = Quantity("acc1d", 1, "m/s^2")


if __name__ == "__main__":
    from pprint import pprint

    sheet = create_sheet([
        time, gas, pos1d, vel1d, acc1d
    ])

    pprint(sheet)
    print(time)