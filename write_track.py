import lapsim as ls
import numpy as np
import math


path = "track.csv"

track_params = np.array([
    [5, 0],
    [math.pi * 0.6, -0.5],
    [2, 0],
    [math.pi * 2, 0.6],
    [5, 0],
    [math.pi, -0.2],
    [math.pi * 0.6, -1.5],
    [3, 0],
    [math.pi * 1.2, 0.8],
    [4, 0],

], dtype='f4')

track_params = ls.track.scale_segment_track(
    track_params, 20)

np.savetxt(path, track_params, delimiter=",")

track = ls.track.read_segment_track(path)
print(track.length)

track.plot()
