import polars as pl
import numpy as np
import time
from dataset import generate_dataset, collapse_dataset, find_overlapping_tracks
from ppd import poly_dist
import os
# os.environ["POLARS_VERBOSE"] = "1"

def run_benchmark(num_tracks, points_per_track, max_time):
    print(f"\nBenchmarking with {num_tracks} tracks, {points_per_track} points per track:")
    
    # Generate base dataset with maximum size once
    print("Generating base dataset...")
    df = generate_dataset(num_tracks=num_tracks, avg_points_per_track=points_per_track, max_time=max_time)
    collapsed = collapse_dataset(df)
    # Filter to desired size
    
    # Time overlap detection
    t0 = time.perf_counter()
    overlaps = find_overlapping_tracks(collapsed)
    t1 = time.perf_counter()
    overlap_time = t1 - t0
    print(f"Overlap detection: {overlap_time:.3f}s")
    
    # Time distance calculation
    t0 = time.perf_counter()
    result = overlaps.with_columns([
        poly_dist(
            pl.col("track_id_1"),
            pl.col("track_id_2"),
            pl.col("overlap_start"),
            pl.col("overlap_end"),
            collapsed["track_id"],
            collapsed["x_list"],
            collapsed["y_list"],
            collapsed["timestamp_list"]
        ).alias("avg_distance")
    ])
    t1 = time.perf_counter()
    distance_time = t1 - t0
    print(f"Distance calculation: {distance_time:.3f}s")
    # Print some stats
    print(f"Number of overlapping pairs: {len(overlaps)}")
    print(f"Total time: {overlap_time + distance_time:.3f}s")
    
    return {
        "num_tracks": num_tracks,
        "points_per_track": points_per_track,
        'max_time': max_time,
        "overlap_time": overlap_time,
        "distance_time": distance_time,
        "total_time": overlap_time + distance_time,
        "num_overlaps": len(overlaps)
    }

# Run benchmarks with different sizes
sizes = [
    (100, 50, 20),
    (100, 100, 20),
    (100, 200, 20),
    (100, 400, 20),
    (100, 800, 20),
    (100, 1600, 20),
    (100, 50, 20),
    (200, 50, 20),
    (400, 50, 20),
    (800, 50, 20),
    (100, 50, 40),
    (100, 50, 80),
    (100, 50, 160),
    (100, 50, 320),
    (100, 50, 640),
    # (3600, 100, 500),
]

results = []
for num_tracks, points_per_track, max_time in sizes:
    result = run_benchmark(num_tracks, points_per_track, max_time)
    results.append(result)

# Print summary table
print("\nPerformance Summary:")
print("Tracks | Points/Track | MaxStartTime | Overlap (s) | Distance (s) | Total (s) | Overlaps")
print("-" * 80)
for r in results:
    print(f"{r['num_tracks']:6d} | {r['points_per_track']:6d} | {r['max_time']:12d} | {r['overlap_time']:10.3f} | {r['distance_time']:10.3f} | {r['total_time']:8.3f} | {r['num_overlaps']:8d}") 
    
"""
regular build:

Performance Summary:
Tracks | Points/Track | MaxStartTime | Overlap (s) | Distance (s) | Total (s) | Overlaps
--------------------------------------------------------------------------------
   100 |     50 |           20 |      0.003 |      0.022 |    0.025 |     4950
   100 |    100 |           20 |      0.002 |      0.035 |    0.036 |     4950
   100 |    200 |           20 |      0.002 |      0.064 |    0.066 |     4950
   100 |    400 |           20 |      0.002 |      0.121 |    0.123 |     4950
   100 |    800 |           20 |      0.002 |      0.291 |    0.293 |     4950
   100 |   1600 |           20 |      0.002 |      0.513 |    0.515 |     4950
   100 |     50 |           20 |      0.002 |      0.017 |    0.019 |     4950
   200 |     50 |           20 |      0.002 |      0.054 |    0.055 |    19900
   400 |     50 |           20 |      0.006 |      0.215 |    0.221 |    79800
   800 |     50 |           20 |      0.023 |      0.933 |    0.955 |   319600
   100 |     50 |           40 |      0.002 |      0.020 |    0.022 |     4950
   100 |     50 |           80 |      0.002 |      0.015 |    0.017 |     4341
   100 |     50 |          160 |      0.002 |      0.009 |    0.012 |     2596
   100 |     50 |          320 |      0.002 |      0.008 |    0.010 |     1360
   100 |     50 |          640 |      0.002 |      0.004 |    0.006 |      719
   
   number of tracks has the most effect
   
   
release build - honestly the boost is insane... it still scales polynomialy with track count though

Performance Summary:
Tracks | Points/Track | MaxStartTime | Overlap (s) | Distance (s) | Total (s) | Overlaps
--------------------------------------------------------------------------------
   100 |     50 |           20 |      0.019 |      0.008 |    0.027 |     4950
   100 |    100 |           20 |      0.003 |      0.003 |    0.006 |     4950
   100 |    200 |           20 |      0.002 |      0.004 |    0.006 |     4950
   100 |    400 |           20 |      0.002 |      0.010 |    0.012 |     4950
   100 |    800 |           20 |      0.002 |      0.025 |    0.027 |     4950
   100 |   1600 |           20 |      0.003 |      0.051 |    0.054 |     4950
   100 |     50 |           20 |      0.002 |      0.002 |    0.004 |     4950
   200 |     50 |           20 |      0.002 |      0.004 |    0.007 |    19900
   400 |     50 |           20 |      0.007 |      0.015 |    0.022 |    79800
   800 |     50 |           20 |      0.020 |      0.059 |    0.079 |   319600
   100 |     50 |           40 |      0.002 |      0.002 |    0.004 |     4950
   100 |     50 |           80 |      0.003 |      0.002 |    0.004 |     3736
   100 |     50 |          160 |      0.002 |      0.002 |    0.004 |     2523
   100 |     50 |          320 |      0.002 |      0.001 |    0.003 |     1360
   100 |     50 |          640 |      0.002 |      0.001 |    0.003 |      708
"""