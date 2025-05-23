import polars as pl
import numpy as np
from dataset import plot_tracks, generate_dataset, collapse_dataset, find_overlapping_tracks, join_overlapping_tracks
from pg import match_nearest_point

# Generate test dataset
df = generate_dataset(num_tracks=10, avg_points_per_track=20)
collapsed = collapse_dataset(df)
overlaps = find_overlapping_tracks(collapsed)
overlapping_tracks = join_overlapping_tracks(collapsed, overlaps)

# Calculate nearest points between overlapping tracks
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

# Print results
print("\nOverlapping tracks with average distances:")
print(result.select(["track_id_1", "track_id_2", "avg_distance"]).sort('avg_distance'))
plot_tracks(df)#.filter(pl.col('track_id').is_in([3, 6, 0, 7])))