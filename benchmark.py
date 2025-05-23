import polars as pl
import numpy as np
import time
from dataset import generate_dataset, collapse_dataset, find_overlapping_tracks, join_overlapping_tracks
from pg import match_nearest_point

# Generate base dataset with maximum size once
print("Generating base dataset...")
df = generate_dataset(num_tracks=200, avg_points_per_track=50)
collapsed = collapse_dataset(df)

def run_benchmark(num_tracks, points_per_track):
    print(f"\nBenchmarking with {num_tracks} tracks, {points_per_track} points per track:")
    
    # Filter to desired size
    filtered_df = df.filter(pl.col("track_id") < num_tracks)
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
        "overlap_time": overlap_time,
        "join_time": join_time,
        "distance_time": distance_time,
        "total_time": overlap_time + join_time + distance_time,
        "num_overlaps": len(overlaps)
    }

# Run benchmarks with different sizes
sizes = [
    (100, 50),
    (200, 50),
]

results = []
for num_tracks, points_per_track in sizes:
    result = run_benchmark(num_tracks, points_per_track)
    results.append(result)

# Print summary table
print("\nPerformance Summary:")
print("Tracks | Points/Track | Overlap (s) | Join (s) | Distance (s) | Total (s) | Overlaps")
print("-" * 80)
for r in results:
    print(f"{r['num_tracks']:6d} | {r['points_per_track']:12d} | {r['overlap_time']:10.3f} | {r['join_time']:8.3f} | {r['distance_time']:10.3f} | {r['total_time']:8.3f} | {r['num_overlaps']:8d}") 