from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from pg._internal import __version__ as __version__

if TYPE_CHECKING:
    from pg.typing import IntoExprColumn

LIB = Path(__file__).parent


def frechet_distance_expr(expr: IntoExprColumn, other: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr, other],
        plugin_path=LIB,
        function_name="frechet_distance_expr",
        is_elementwise=True,
    )
    
def match_nearest_point(
    x1: IntoExprColumn, 
    y1: IntoExprColumn, 
    t1: IntoExprColumn,
    x2: IntoExprColumn, 
    y2: IntoExprColumn,
    t2: IntoExprColumn,
    overlap_start: IntoExprColumn,
    overlap_end: IntoExprColumn
) -> pl.Expr:
    return register_plugin_function(
        args=[x1, y1, t1, x2, y2, t2, overlap_start, overlap_end],
        plugin_path=LIB,
        function_name="match_nearest_point",
        is_elementwise=True,
    )

