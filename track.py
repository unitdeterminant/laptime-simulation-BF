class BaseTrack:
    pass


class LineTrack(BaseTrack):
    def __init__(self, lenght=100):
        super(LineTrack, self).__init__()
        self.lenght = lenght

    def check_finished(self, param_matrix):
        return param_matrix['position'][0] >= self.lenght
