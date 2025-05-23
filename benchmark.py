import polars as pl
import numpy as np
import time
from dataset import generate_dataset, collapse_dataset, find_overlapping_tracks, join_overlapping_tracks
from pg import match_nearest_point
import os
# os.environ["POLARS_VERBOSE"] = "1"

# Generate base dataset with maximum size once
print("Generating base dataset...")
df = generate_dataset(num_tracks=1000, avg_points_per_track=500)
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
    # (100, 200),
    # (1000, 500),
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
    
    
    
"""
for now we get roughly;
Benchmarking with 100 tracks, 50 points per track:
Overlap detection: 0.003s
Track joining: 0.008s
Distance calculation: 0.246s
Number of overlapping pairs: 4950
Total time: 0.256s

Benchmarking with 200 tracks, 50 points per track:
Overlap detection: 0.002s
Track joining: 0.030s
Distance calculation: 0.967s
Number of overlapping pairs: 19900
Total time: 0.999s
"""

"""
after optimization we get worse performance:
Benchmarking with 100 tracks, 50 points per track:
Overlap detection: 0.005s
Track joining: 0.022s
Distance calculation: 1.058s
Number of overlapping pairs: 4950
Total time: 1.085s

Benchmarking with 200 tracks, 50 points per track:
Overlap detection: 0.002s
Track joining: 0.118s
Distance calculation: 4.144s
Number of overlapping pairs: 19900
Total time: 4.264s
"""

"""
Point 200 timing: get_point=600ns, search=3µs, dist=600ns
Point 100 timing: get_point=700ns, search=2.9µs, dist=500ns
Point 200 timing: get_point=500ns, search=2.3µs, dist=400ns
Point 300 timing: get_point=500ns, search=2.4µs, dist=400ns
Point 100 timing: get_point=500ns, search=2.7µs, dist=400ns
Point 400 timing: get_point=700ns, search=2.9µs, dist=500ns
Row 3802 timing: extract=26.4µs, filter=267.3µs, calc=5.6684ms, total=5.9642ms
Point 300 timing: get_point=400ns, search=2.4µs, dist=300ns
Row 3414 timing: extract=28.9µs, filter=238µs, calc=5.5695ms, total=5.8395ms
Row 2566 timing: extract=30.1µs, filter=285.8µs, calc=7.1375ms, total=7.4567ms
Point 200 timing: get_point=700ns, search=2.9µs, dist=400ns
Point 400 timing: get_point=400ns, search=1.7µs, dist=300ns
Point 300 timing: get_point=800ns, search=2.5µs, dist=400ns
Point 100 timing: get_point=600ns, search=2.8µs, dist=500ns
Row 1634 timing: extract=24.1µs, filter=194.6µs, calc=5.3249ms, total=5.5452ms
"""

"""
Point 300 timing: get_point=0ns, search=300ns, dist=0ns
Point 300 timing: get_point=0ns, search=300ns, dist=100ns
Point 200 timing: get_point=0ns, search=400ns, dist=100ns
Point 100 timing: get_point=0ns, search=300ns, dist=100ns
Point 400 timing: get_point=0ns, search=300ns, dist=0ns
Point 400 timing: get_point=0ns, search=300ns, dist=100ns
Row 4467 timing: filter=48.6µs, calc=4.55ms, total=4.6357ms
Point 400 timing: get_point=100ns, search=300ns, dist=0ns
Point 100 timing: get_point=0ns, search=400ns, dist=0ns
Point 200 timing: get_point=0ns, search=300ns, dist=100ns
Point 100 timing: get_point=0ns, search=300ns, dist=0ns
Point 300 timing: get_point=0ns, search=300ns, dist=100ns
Point 400 timing: get_point=0ns, search=400ns, dist=0ns
Point 400 timing: get_point=0ns, search=400ns, dist=0ns
Point 300 timing: get_point=0ns, search=400ns, dist=100ns
Point 200 timing: get_point=100ns, search=300ns, dist=0ns
Row 137 timing: filter=44.6µs, calc=4.5185ms, total=4.5868ms
Row 755 timing: filter=65µs, calc=4.4754ms, total=4.5727ms
Point 100 timing: get_point=100ns, search=500ns, dist=100ns
Row 3848 timing: filter=44.5µs, calc=4.1563ms, total=4.223ms
Point 200 timing: get_point=100ns, search=300ns, dist=0ns
Point 300 timing: get_point=0ns, search=300ns, dist=100ns
Point 200 timing: get_point=0ns, search=300ns, dist=100ns
Point 400 timing: get_point=0ns, search=300ns, dist=0ns
Row 2612 timing: filter=65.5µs, calc=4.3482ms, total=4.4424ms
Row 908 timing: filter=44.5µs, calc=4.2555ms, total=4.3219ms
Point 400 timing: get_point=100ns, search=200ns, dist=0ns
Point 300 timing: get_point=100ns, search=300ns, dist=0ns
Point 100 timing: get_point=100ns, search=500ns, dist=100ns
Point 100 timing: get_point=100ns, search=400ns, dist=0ns
Point 200 timing: get_point=0ns, search=400ns, dist=100ns
Point 100 timing: get_point=100ns, search=300ns, dist=100ns
Point 300 timing: get_point=0ns, search=300ns, dist=100ns
Point 400 timing: get_point=100ns, search=300ns, dist=0ns
Point 300 timing: get_point=0ns, search=300ns, dist=0ns
Row 1374 timing: filter=44µs, calc=4.3514ms, total=4.4168ms
Point 100 timing: get_point=0ns, search=500ns, dist=100ns
Point 100 timing: get_point=0ns, search=300ns, dist=100ns
Row 1993 timing: filter=44.5µs, calc=4.3858ms, total=4.452ms
Point 400 timing: get_point=0ns, search=300ns, dist=100ns
Point 200 timing: get_point=0ns, search=300ns, dist=100ns
Point 200 timing: get_point=100ns, search=500ns, dist=100ns
Point 300 timing: get_point=100ns, search=300ns, dist=0ns
"""

"""
Benchmarking with 100 tracks, 50 points per track:
Overlap detection: 0.003s
Track joining: 0.025s
Distance calculation: 0.158s
Number of overlapping pairs: 4950
Total time: 0.185s

Benchmarking with 200 tracks, 50 points per track:
Overlap detection: 0.002s
Track joining: 0.124s
Distance calculation: 0.659s
Number of overlapping pairs: 19900
Total time: 0.786s
"""