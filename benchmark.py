import polars as pl
import numpy as np
import time
from dataset import generate_dataset, collapse_dataset, find_overlapping_tracks, join_overlapping_tracks
from pg import match_nearest_point
import os
# os.environ["POLARS_VERBOSE"] = "1"

def run_benchmark(num_tracks, points_per_track, max_time):
    print(f"\nBenchmarking with {num_tracks} tracks, {points_per_track} points per track:")
    
    # Generate base dataset with maximum size once
    print("Generating base dataset...")
    df = generate_dataset(num_tracks=num_tracks, avg_points_per_track=points_per_track, max_time=max_time)
    collapsed = collapse_dataset(df)
    # Filter to desired size
    filtered_collapsed = collapsed.filter(pl.col("track_id") < num_tracks)
    
    # Time overlap detection
    t0 = time.perf_counter()
    overlaps = find_overlapping_tracks(filtered_collapsed)
    t1 = time.perf_counter()
    overlap_time = t1 - t0
    print(f"Overlap detection: {overlap_time:.3f}s")
    
    # Time track joining
    t0 = time.perf_counter()
    overlapping_tracks = join_overlapping_tracks(filtered_collapsed, overlaps)
    t1 = time.perf_counter()
    join_time = t1 - t0
    print(f"Track joining: {join_time:.3f}s")
    
    # Time distance calculation
    t0 = time.perf_counter()
    result = overlapping_tracks.with_columns([
        match_nearest_point(
            pl.col("x_list"),
            pl.col("y_list"),
            pl.col("timestamp_list"),
            pl.col("x_list_2"),
            pl.col("y_list_2"),
            pl.col("timestamp_list_2"),
            pl.col("overlap_start"),
            pl.col("overlap_end")
        ).alias("avg_distance")
    ])
    t1 = time.perf_counter()
    distance_time = t1 - t0
    print(f"Distance calculation: {distance_time:.3f}s")
    # Print some stats
    print(f"Number of overlapping pairs: {len(overlaps)}")
    print(f"Total time: {overlap_time + join_time + distance_time:.3f}s")
    
    return {
        "num_tracks": num_tracks,
        "points_per_track": points_per_track,
        'max_time': max_time,
        "overlap_time": overlap_time,
        "join_time": join_time,
        "distance_time": distance_time,
        "total_time": overlap_time + join_time + distance_time,
        "num_overlaps": len(overlaps)
    }

# Run benchmarks with different sizes
sizes = [
    # (100, 50, 20),
    # (100, 100, 20),
    # (100, 200, 20),
    # (100, 400, 20),
    # (100, 800, 20),
    # (100, 1600, 20),
    # (100, 50, 20),
    # (200, 50, 20),
    # (400, 50, 20),
    # (800, 50, 20),
    # # (1600, 50, 20),
    (1600, 100, 100),
    (1600, 100, 200),
    # (100, 50, 40),
    # (100, 50, 80),
    # (100, 50, 160),
    # (100, 50, 320),
    # (100, 50, 640),
]

results = []
for num_tracks, points_per_track, max_time in sizes:
    result = run_benchmark(num_tracks, points_per_track, max_time)
    results.append(result)

# Print summary table
print("\nPerformance Summary:")
print("Tracks | Points/Track | MaxStartTime | Overlap (s) | Join (s) | Distance (s) | Total (s) | Overlaps")
print("-" * 80)
for r in results:
    print(f"{r['num_tracks']:6d} | {r['points_per_track']:6d} | {r['max_time']:12d} | {r['overlap_time']:10.3f} | {r['join_time']:8.3f} | {r['distance_time']:10.3f} | {r['total_time']:8.3f} | {r['num_overlaps']:8d}") 
    
"""
Performance Summary:
Tracks | Points/Track | Overlap (s) | Join (s) | Distance (s) | Total (s) | Overlaps
--------------------------------------------------------------------------------
   100 |           50 |      0.004 |    0.007 |      0.029 |    0.040 |     4950
   100 |          100 |      0.001 |    0.007 |      0.039 |    0.047 |     4950
   100 |          200 |      0.001 |    0.017 |      0.067 |    0.085 |     4950
   100 |          400 |      0.002 |    0.024 |      0.125 |    0.151 |     4950
   100 |          800 |      0.002 |    0.044 |      0.247 |    0.293 |     4950
   100 |         1600 |      0.002 |    0.092 |      0.494 |    0.588 |     4950
   100 |           50 |      0.002 |    0.006 |      0.023 |    0.030 |     4950
   200 |           50 |      0.001 |    0.024 |      0.092 |    0.118 |    19900
   400 |           50 |      0.005 |    0.103 |      0.391 |    0.499 |    79800
   800 |           50 |      0.024 |    0.480 |      1.525 |    2.029 |   319600
  1600 |           50 |      0.066 |    5.166 |      6.692 |   11.925 |  1279200
"""