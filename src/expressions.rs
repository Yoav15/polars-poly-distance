#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use curve_similarities::{frechet, DistMetric};
use polars::prelude::PlSmallStr;
use polars::prelude::Series;
use ndarray::{Array2, ArrayView1};

#[polars_expr(output_type = Float64)]
fn match_nearest_point(inputs: &[Series]) -> PolarsResult<Series> {
    let lhs = &inputs[0]; // a: List[List[f64]]
    let rhs = &inputs[1]; // b: List[List[f64]]

    let mut result_rows = Vec::with_capacity(lhs.len());

    for row in 0..lhs.len() {
        let mut total: f64 = 0.0;


        let a_val = lhs.get(row)?;
        let b_val = rhs.get(row)?;

        let a_series = match a_val {
            AnyValue::List(s) => s,
            _ => return Err(PolarsError::ComputeError("Expected list of points in column A".into())),
        };

        let b_series = match b_val {
            AnyValue::List(s) => s,
            _ => return Err(PolarsError::ComputeError("Expected list of points in column B".into())),
        };

        let points_a = convert_point_list(&a_series)?;
        let points_b = convert_point_list(&b_series)?;

        let mut indices = Vec::with_capacity(points_a.nrows());

        for i in 0..points_a.nrows() {
            let a_point = points_a.row(i);
            let mut min_dist = f64::MAX;
            // let mut min_idx = -1;

            for j in 0..points_b.nrows() {
                let b_point = points_b.row(j);
                let dist = euclidean(&a_point, &b_point);
                if dist < min_dist {
                    min_dist = dist;
                    // min_idx = j as i64;
                }
            }
            total += min_dist;

            indices.push(min_dist);
        }

        result_rows.push(total);
    }

    Ok(Series::new(PlSmallStr::from_str("result"), &result_rows))
}

fn convert_point_list(series: &Series) -> PolarsResult<Array2<f64>> {
    let mut points = Vec::with_capacity(series.len());

    for i in 0..series.len() {
        let val = series.get(i)?;

        let coord_series = match val {
            AnyValue::List(s) => s,
            _ => {
                return Err(PolarsError::ComputeError(
                    "Expected a point to be a list of floats".into(),
                ))
            }
        };

        let coord_series_f64 = coord_series.cast(&DataType::Float64)?;
        let coords = coord_series_f64.f64()?;

        if coords.len() != 2 {
            return Err(PolarsError::ComputeError(
                "Each point must contain exactly 2 coordinates".into(),
            ));
        }

        let x = coords.get(0).unwrap();
        let y = coords.get(1).unwrap();
        points.push([x, y]);
    }

    Array2::from_shape_vec((points.len(), 2), points.iter().flat_map(|p| *p).collect())
        .map_err(|e| PolarsError::ComputeError(format!("Failed to build ndarray: {e}").into()))
}

fn euclidean(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

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
