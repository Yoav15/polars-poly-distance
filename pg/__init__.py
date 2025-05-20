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
    
def match_nearest_point(expr: IntoExprColumn, weights: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr, weights],
        plugin_path=LIB,
        function_name="match_nearest_point",
        is_elementwise=True,
    )

