from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from ppd._internal import __version__ as __version__

if TYPE_CHECKING:
    from ppd.typing import IntoExprColumn

LIB = Path(__file__).parent

    
def poly_dist(
    track_id_1: IntoExprColumn,
    track_id_2: IntoExprColumn,
    overlap_start: IntoExprColumn,
    overlap_end: IntoExprColumn,
    track_ids: IntoExprColumn,
    x_lists: IntoExprColumn,
    y_lists: IntoExprColumn,
    timestamp_lists: IntoExprColumn,
) -> pl.Expr:
    """
    Calculate the average distance between nearest points in overlapping tracks.
    
    Args:
        track_id_1: Column containing the first track ID
        track_id_2: Column containing the second track ID
        overlap_start: Column containing the start time of overlap
        overlap_end: Column containing the end time of overlap
        track_ids: Column containing track IDs
        x_lists: Column containing x coordinates lists
        y_lists: Column containing y coordinates lists
        timestamp_lists: Column containing timestamp lists
        
    Returns:
        Expression that calculates the average distance between nearest points
    """
    return register_plugin_function(
        args=[track_id_1, track_id_2, overlap_start, overlap_end, track_ids, x_lists, y_lists, timestamp_lists],
        plugin_path=LIB,
        function_name="poly_dist",
        is_elementwise=True,
    )

