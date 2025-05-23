import polars as pl
import numpy as np
import time

def generate_dataset(num_tracks: int, avg_points_per_track: int, max_time: float = 20.0):
    """
    Generate a dataset of linear tracks with random noise.
    Each track has sequential timestamps and follows a linear pattern (y = mx + b)
    with random slope and intercept for each track.
    Tracks can overlap in time but have unique track IDs.
    
    Args:
        num_tracks: Number of unique tracks to generate
        avg_points_per_track: Average number of points per track
        max_time: Maximum possible start time for tracks
        
    Returns:
        DataFrame with columns: track_id, x, y, timestamp
    """
    # Generate random start times and lengths for each track
    start_times = np.random.uniform(0, max_time, num_tracks)
    track_lengths = np.random.normal(avg_points_per_track, 2, num_tracks).astype(int)
    track_lengths = np.clip(track_lengths, 3, avg_points_per_track * 2)  # Ensure reasonable lengths
    
    # Generate track IDs and timestamps
    track_ids = []
    timestamps = []
    for i in range(num_tracks):
        track_ids.extend([i] * track_lengths[i])
        # Generate timestamps with fixed delta of 1
        track_times = np.arange(track_lengths[i]) + start_times[i]
        timestamps.extend(track_times)
    
    track_ids = np.array(track_ids)
    timestamps = np.array(timestamps)
    
    # Generate random slopes and intercepts for each track
    slopes = np.random.uniform(-2, 2, num_tracks)
    intercepts = np.random.uniform(-5, 5, num_tracks)
    
    # Calculate x coordinates (using timestamps as x) with some noise
    x = timestamps + np.random.normal(0, 0.2, len(timestamps))
    
    # Calculate y coordinates using linear equation y = mx + b
    y = np.zeros_like(x, dtype=np.float32)
    for i in range(num_tracks):
        track_mask = track_ids == i
        y[track_mask] = slopes[i] * x[track_mask] + intercepts[i]
    
    # Add some random noise to y coordinates
    noise = np.random.normal(0, 0.5, len(y))
    y += noise
    
    df = pl.DataFrame({
        "track_id": track_ids,
        "x": x,
        "y": y,
        "timestamp": timestamps,
    })
    return df


def collapse_dataset(df: pl.DataFrame):
    """
    Collapse each track into a single row with a list of points,
    with 2 extra columns for each track_id, the start and end timestamp.
    """
    df = df.group_by("track_id").agg(
        pl.col("x").alias("x_list"),
        pl.col("y").alias("y_list"),
        pl.col("timestamp").alias("timestamp_list"),
        pl.col("timestamp").min().alias("start_timestamp"),
        pl.col("timestamp").max().alias("end_timestamp"),
    )
    return df


def find_overlapping_tracks(df: pl.DataFrame):
    """
    Find tracks that overlap in time.
    df: DataFrame with columns: track_id, start_timestamp, end_timestamp
    
    Returns:
        DataFrame with columns: track_id_1, track_id_2, overlap_start, overlap_end
        where track_id_1 and track_id_2 are the overlapping tracks,
        and overlap_start/end are the time period of overlap
    """
    # Create a self-join to compare each track with every other track
    df1 = df.select(
        pl.col("track_id").alias("track_id_1"),
        pl.col("start_timestamp").alias("start_1"),
        pl.col("end_timestamp").alias("end_1")
    )
    df2 = df.select(
        pl.col("track_id").alias("track_id_2"),
        pl.col("start_timestamp").alias("start_2"),
        pl.col("end_timestamp").alias("end_2")
    )
    
    # Cross join to compare all pairs
    joined = df1.join(df2, how="cross")
    
    # Filter out self-comparisons and find overlapping periods
    overlaps = joined.filter(
        (pl.col("track_id_1") < pl.col("track_id_2")) &  # Avoid duplicate pairs and self-comparisons
        (pl.col("end_1") >= pl.col("start_2")) &  # Track 1 ends after Track 2 starts
        (pl.col("start_1") <= pl.col("end_2"))    # Track 1 starts before Track 2 ends
    ).select(
        pl.col("track_id_1"),
        pl.col("track_id_2"),
        pl.max_horizontal("start_1", "start_2").alias("overlap_start"),
        pl.min_horizontal("end_1", "end_2").alias("overlap_end")
    )
    
    return overlaps


def join_overlapping_tracks(df: pl.DataFrame, overlaps: pl.DataFrame):
    """
    Join the original track data with the overlap information to get the actual points
    that overlap between tracks.
    
    Args:
        df: Collapsed DataFrame with track points (from collapse_dataset)
        overlaps: DataFrame with overlap information from find_overlapping_tracks
        
    Returns:
        DataFrame with columns: track_id_1, track_id_2, x_list_1, y_list_1, x_list_2, y_list_2
        where x_list_1,y_list_1 are points from track_id_1 and x_list_2,y_list_2 are points from track_id_2
        that occur during the overlap period
    """
    # Join overlaps with track 1 points
    df1 = overlaps.join(
        df.select(
            pl.col("track_id"),
            pl.col("x_list"),
            pl.col("y_list"),
            pl.col("timestamp_list")
        ),
        left_on="track_id_1",
        right_on="track_id"
    )
    
    # Join with track 2 points
    result = df1.join(
        df.select(
            pl.col("track_id"),
            pl.col("x_list").alias("x_list_2"),
            pl.col("y_list").alias("y_list_2"),
            pl.col("timestamp_list").alias("timestamp_list_2")
        ),
        left_on="track_id_2",
        right_on="track_id"
    )
    
    # # Filter points to only those in the overlap period
    # result = result.with_columns([
    #     # For track 1
    #     pl.col("x_list").list.eval(
    #         pl.element().filter(
    #             pl.col("timestamp_list").is_between(pl.col("overlap_start"), pl.col("overlap_end"))
    #         )
    #     ).alias("x_list_1"),
    #     pl.col("y_list").list.eval(
    #         pl.element().filter(
    #             pl.col("timestamp_list").is_between(pl.col("overlap_start"), pl.col("overlap_end"))
    #         )
    #     ).alias("y_list_1"),
    #     # For track 2
    #     pl.col("x_list_2").list.eval(
    #         pl.element().filter(
    #             pl.col("timestamp_list_2").is_between(pl.col("overlap_start"), pl.col("overlap_end"))
    #         )
    #     ).alias("x_list_2"),
    #     pl.col("y_list_2").list.eval(
    #         pl.element().filter(
    #             pl.col("timestamp_list_2").is_between(pl.col("overlap_start"), pl.col("overlap_end"))
    #         )
    #     ).alias("y_list_2")
    # ])
    
    # # Select only the columns we want
    # result = result.select([
    #     "track_id_1", "track_id_2",
    #     "x_list_1", "y_list_1",
    #     "x_list_2", "y_list_2",
    #     "overlap_start", "overlap_end"
    # ])
    
    return result

# Generate dataset with overlapping time periods
df = generate_dataset(100, 15, max_time=100.0)
df = collapse_dataset(df)
overlaps = find_overlapping_tracks(df)
overlapping_dataset = join_overlapping_tracks(df, overlaps)
print("\nOverlapping tracks with their points:")
print(overlapping_dataset)