import polars as pl
import numpy as np
import time
import plotly.graph_objects as go
from pg import match_nearest_point


def generate_dataset(num_tracks: int, avg_points_per_track: int, max_time: float = 20.0):
    """
    Generate a dataset of linear tracks with random noise.
    Each track has sequential timestamps and follows a linear pattern (y = mx + b)
    with random slope and intercept for each track.
    
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
    
    # Generate x coordinates independently for each track
    x = np.zeros_like(timestamps, dtype=np.float32)
    for i in range(num_tracks):
        track_mask = track_ids == i
        # Generate random x coordinates in range [-10, 10] and sort them
        x[track_mask] = np.sort(np.random.uniform(-10, 10, track_lengths[i]))
    
    # Calculate y coordinates using linear equation y = mx + b
    y = np.zeros_like(x, dtype=np.float32)
    for i in range(num_tracks):
        track_mask = track_ids == i
        y[track_mask] = slopes[i] * x[track_mask] + intercepts[i]
    
    # Add some random noise to y coordinates
    noise = np.random.normal(0, 0.5, len(y))
    y += noise
    
    df = pl.DataFrame(
        {
            "track_id": track_ids,
            "x": x,
            "y": y,
            "timestamp": timestamps,
        }, 
        schema={
            "track_id": pl.Int32,
            "x": pl.Float32,
            "y": pl.Float32,
            "timestamp": pl.Float32,
        }
        )
    return df


def collapse_dataset(df: pl.DataFrame):
    """
    Collapse each track into a single row with a list of points,
    with 2 extra columns for each track_id, the start and end timestamp.
    """
    df = df.group_by("track_id", maintain_order=True).agg(
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
    
    return result


def plot_tracks(df: pl.DataFrame):
    """
    Create an interactive 3D plot of the tracks in x,y,time space using Plotly.
    
    Args:
        df: DataFrame with columns: track_id, x, y, timestamp
    """
    # Create figure
    fig = go.Figure()
    
    # Get unique track IDs
    track_ids = df["track_id"].unique()
    
    # Plot each track with a different color
    for track_id in track_ids:
        track_data = df.filter(pl.col("track_id") == track_id)
        fig.add_trace(go.Scatter3d(
            x=track_data["timestamp"],
            y=track_data["x"],
            z=track_data["y"],
            mode='lines',
            name=f"Track {track_id}",
            line=dict(width=2),
            opacity=0.7
        ))
    
    # Update layout
    fig.update_layout(
        title="Tracks in 3D Space",
        scene=dict(
            xaxis_title="Time",
            yaxis_title="X",
            zaxis_title="Y"
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    # Save as interactive HTML file
    fig.write_html("tracks_3d.html")
    
    # Show the plot
    fig.show()


if __name__ == "__main__":
    # Generate dataset with overlapping time periods
    df = generate_dataset(15, 10, max_time=20.0)
    # plot_tracks(df)

    collapsed_df = collapse_dataset(df)
    overlaps = find_overlapping_tracks(collapsed_df)
    overlapping_dataset = join_overlapping_tracks(collapsed_df, overlaps)
    print("\nOverlapping tracks with their points:")
    print(overlapping_dataset)
    result = overlapping_dataset.with_columns([
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
    result.select(['track_id_1', 'track_id_2', 'avg_distance']).sort('avg_distance')
    plot_tracks(df)
    # use the match_nearest_point function on our data

