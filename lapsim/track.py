import matplotlib.pyplot as plt
import numpy as np
import math

from .visual import ensure_folder


class BaseTrack:
    def curvature(self, position):
        raise NotImplementedError

    def check_finished(self, position):
        raise NotImplementedError


class LineTrack:
    def __init__(self, length):
        self.length = length

    def curvature(self, position):
        return 0.

    def check_finished(self, position):
        return position >= self.length


def angle_vector(angle, length=1.):
    return np.array([np.cos(angle), np.sin(angle)]) * length


class BaseSegment:
    def __call__(self, s):
        if s > self.length or s < 0:
            raise ValueError(f"position out of range for segment: {s}")

        return self.call(s)

    def plot(self, path="plots/track.png", eps=1e-8):
        for s in np.linspace(0, self.length - eps, int(self.length)):
            p = self(s)
            plt.plot(p[0], p[1], "bo", markersize=2)

        plt.axis("equal")
        plt.tight_layout()

        ensure_folder(path)
        plt.savefig(path, transparent=True)

    def check_finished(self, position):
        return position >= self.length


class Curve(BaseSegment):
    def __init__(self, length, curvature, start, angle):
        self.length = length
        self.curvature = curvature
        self.start = start
        self.angle = angle
        self.radius = 1 / curvature

        self.final_angle = angle + length * curvature

        self.offset_angle = angle + math.pi / 2
        self.center_offset = angle_vector(self.offset_angle, self.radius)
        self.circle_center = start + self.center_offset

    def call(self, s):
        angle = s / self.radius + self.offset_angle
        offset = -angle_vector(angle, self.radius)
        return self.circle_center + offset


class Line(BaseSegment):
    def __init__(self, length, start, angle):
        self.length = length
        self.start = start
        self.angle = angle
        self.final_angle = angle
        self.curvature = 0

        self.angle_vector = angle_vector(angle, 1.)

    def call(self, s):
        return self.start + self.angle_vector * s


class SegmentTrack(BaseSegment, BaseTrack):
    def __init__(self, segments):
        self.segments = segments
        self.clengths = np.cumsum([s.length for s in segments])
        self.length = self.clengths[-1]

    def get_index(self, s):
        for i, l in enumerate(self.clengths):
            if l > s:
                return i
        else:
            return len(self.segments) - 1

    def curvature(self, s):
        i = self.get_index(s)
        return self.segments[i].curvature

    def call(self, s):
        i = self.get_index(s)
        if i > 0:
            s = s - self.clengths[i - 1]
        return self.segments[i](s)

    def check_finished(self, position):
        return super().check_finished(position)


def read_segment_track(path="track.csv", scale=1.):
    track_params = np.genfromtxt(path, "f4", delimiter=",")
    track_params = scale_segment_track(track_params, scale)

    segments = []
    start = np.array([0, 0], "f4")
    angle = 0

    for length, curvature in track_params:
        if curvature == 0:
            segment = Line(length, start, angle)
        else:
            segment = Curve(length, curvature, start, angle)

        segments.append(segment)
        start = segment(length)
        angle = segment.final_angle

    return SegmentTrack(segments)


def scale_segment_track(track_params, scale):
    track_params[:, 0] *= scale
    track_params[:, 1] /= scale
    return track_params