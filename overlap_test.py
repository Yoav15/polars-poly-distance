import polars as pl
import numpy as np
import time
from dataset import plot_tracks, generate_dataset, collapse_dataset, find_overlapping_tracks
from ppd import match_nearest_point

def test_entire_pipeline(num_tracks=100, avg_points_per_track=20, max_time=50):
    start_time = time.perf_counter()
    # Generate test dataset
    df = generate_dataset(num_tracks=num_tracks, avg_points_per_track=avg_points_per_track, max_time=max_time)
    collapsed = collapse_dataset(df)
    overlaps = find_overlapping_tracks(collapsed)
    track_data = collapsed.select(
        pl.col("track_id"),
        pl.col("x_list"),
        pl.col("y_list"),
        pl.col("timestamp_list")
    )
    
    # Calculate nearest points between overlapping tracks
    result = overlaps.with_columns([
        match_nearest_point(
            pl.col("track_id_1"),
            pl.col("track_id_2"),
            pl.col("overlap_start"),
            pl.col("overlap_end"),
            track_data["track_id"],
            track_data["x_list"],
            track_data["y_list"],
            track_data["timestamp_list"]
        ).alias("avg_distance")
    ])
    end_time = time.perf_counter()
    return end_time - start_time

def test_batched_pipeline(num_tracks=100, avg_points_per_track=20, max_time=50, batch_size=10):
    start_time = time.perf_counter()
    # Generate test dataset
    df = generate_dataset(num_tracks=num_tracks, avg_points_per_track=avg_points_per_track, max_time=max_time)
    collapsed = collapse_dataset(df)
    track_data = collapsed.select(
        pl.col("track_id"),
        pl.col("x_list"),
        pl.col("y_list"),
        pl.col("timestamp_list")
    )
    total_time = 0
    
    for batch in collapsed.sort('start_timestamp').iter_slices(n_rows=batch_size):
        batch_start = time.perf_counter()
        overlaps = find_overlapping_tracks(batch)
        # Calculate nearest points between overlapping tracks
        result = overlaps.with_columns([
            match_nearest_point(
                pl.col("track_id_1"),
                pl.col("track_id_2"),
                pl.col("overlap_start"),
                pl.col("overlap_end"),
                track_data["track_id"],
                track_data["x_list"],
                track_data["y_list"],
                track_data["timestamp_list"]
            ).alias("avg_distance")
        ])
        batch_end = time.perf_counter()
        total_time += (batch_end - batch_start)
    
    end_time = time.perf_counter()
    return end_time - start_time, total_time

def run_performance_comparison():
    test_cases = [
        (50, 10, 20),    # Small dataset
        (100, 20, 50),   # Medium dataset
        (200, 30, 100),  # Large dataset
        (500, 40, 200),  # Very large dataset
        (200, 100, 200),  # Very large dataset
    ]
    
    batch_sizes = [20, 50]
    
    print("\nPerformance Comparison: Batched vs Non-batched Pipeline")
    print("=" * 120)
    # Create header with batch sizes
    header = f"{'Test Case':<20} {'Non-batched (s)':<15}"
    for batch_size in batch_sizes:
        header += f"Batch {batch_size} (s):  "
    header += "Best Speedup"
    print(header)
    print("-" * 120)
    
    for num_tracks, avg_points, max_time in test_cases:
        non_batched_time = test_entire_pipeline(num_tracks, avg_points, max_time)
        
        # Test different batch sizes
        batch_times = []
        best_speedup = 0
        
        for batch_size in batch_sizes:
            if batch_size > num_tracks:
                batch_times.append("N/A")
                continue
            total_time, _ = test_batched_pipeline(num_tracks, avg_points, max_time, batch_size)
            batch_times.append(f"{total_time:.2f}")
            speedup = non_batched_time / total_time
            best_speedup = max(best_speedup, speedup)
        
        # Print results for this test case
        result = f"{f'{num_tracks} tracks':<20} {non_batched_time:<15.2f}"
        for time_str in batch_times:
            result += f"{time_str:<15}"
        result += f"{best_speedup:.2f}x"
        print(result)

if __name__ == "__main__":
    # run_performance_comparison()
    test_entire_pipeline()