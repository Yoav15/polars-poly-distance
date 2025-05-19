import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import pg
from frechetdist import frdist
from shapely import frechet_distance
from shapely import LineString
import time

N = 1
samples = 100
points_in_track = 5
arr_0 = [[list(np.random.rand(2)) for _ in range(points_in_track)] for _ in range(samples)]
arr_1 = [[list(np.random.rand(2)) for _ in range(points_in_track)] for _ in range(samples)]

df = pl.DataFrame({
    "a": arr_0,
    "b": arr_1,
})
# print(df)

t0 = time.perf_counter()
for _ in range(N):
    for i in range(samples):
        frechet_distance(LineString(arr_0[i]), LineString(arr_1[i]))
t1 = time.perf_counter()
print("shapely", t1 - t0)


t0 = time.perf_counter()
for _ in range(N):
    for i in range(samples):
        frdist(arr_0[i], arr_1[i])
t1 = time.perf_counter()
print("frechetdist", t1 - t0)

t0 = time.perf_counter()
for _ in range(N):
    df.select(pg.frechet_distance_expr(pl.col("a"), pl.col("b")))

t1 = time.perf_counter()
print("polars", t1 - t0)
