#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use curve_similarities::{frechet, DistMetric};
use polars::prelude::PlSmallStr;
use polars::prelude::Series;
use ndarray::Array2;

#[polars_expr(output_type = Float64)]
fn frechet_distance_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let lhs = &inputs[0];
    let rhs = &inputs[1];

    let mut distances = Vec::with_capacity(lhs.len());

    for i in 0..lhs.len() {
        let a_val = lhs.get(i)?;
        let b_val = rhs.get(i)?;

        // Expect: List of points
        let a_series = match a_val {
            AnyValue::List(s) => s,
            _ => {
                return Err(PolarsError::ComputeError(
                    "Expected left input to be a list of points".into(),
                ))
            }
        };

        let b_series = match b_val {
            AnyValue::List(s) => s,
            _ => {
                return Err(PolarsError::ComputeError(
                    "Expected right input to be a list of points".into(),
                ))
            }
        };

        let a_array = convert_curve_series(&a_series)?;
        let b_array = convert_curve_series(&b_series)?;

        let dist = frechet::<f64>(&a_array, &b_array, DistMetric::Euclidean);
        distances.push(dist);
    }

    Ok(Series::new(PlSmallStr::from_str("frechet"), distances))
}

fn convert_curve_series(series: &Series) -> PolarsResult<Array2<f64>> {
    let mut points = Vec::with_capacity(series.len());

    for i in 0..series.len() {
        let point_val = series.get(i)?;

        let point_series = match point_val {
            AnyValue::List(s) => s,
            _ => {
                return Err(PolarsError::ComputeError(
                    "Expected each point to be a list of 2 floats".into(),
                ))
            }
        };

        let casted = point_series.cast(&DataType::Float64)?;
        let coords = casted.f64()?;

        if coords.len() != 2 {
            return Err(PolarsError::ComputeError(
                "Each point must contain exactly 2 coordinates".into(),
            ));
        }

        let x = coords.get(0).ok_or_else(|| {
            PolarsError::ComputeError("Missing x coordinate in point".into())
        })?;
        let y = coords.get(1).ok_or_else(|| {
            PolarsError::ComputeError("Missing y coordinate in point".into())
        })?;

        points.push([x, y]);
    }

    let flat: Vec<f64> = points.iter().flat_map(|&[x, y]| [x, y]).collect();

    Array2::from_shape_vec((points.len(), 2), flat)
        .map_err(|e| PolarsError::ComputeError(format!("Failed to build ndarray: {e}").into()))
}
