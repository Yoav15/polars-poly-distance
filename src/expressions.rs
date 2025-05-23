#![allow(clippy::unused_unit)]
use polars::prelude::NamedFrom;
use pyo3_polars::derive::polars_expr;
use curve_similarities::{frechet, DistMetric};
use polars::prelude::PlSmallStr;
use polars::prelude::Series;
use polars::prelude::AnyValue;
use polars::error::PolarsError;
use polars::datatypes::DataType;
use polars::prelude::PolarsResult;
use polars::prelude::CompatLevel;
use polars::chunked_array::ops::ChunkAgg;
use ndarray::{Array2};
use flame;

#[polars_expr(output_type = Float64)]
fn match_nearest_point(inputs: &[Series]) -> PolarsResult<Series> {
    let _guard = flame::start_guard("match_nearest_point");
    let lhs = &inputs[0]; // a: List[List[f64]]
    let rhs = &inputs[1]; // b: List[List[f64]]

    let mut result_rows = Vec::with_capacity(lhs.len());

    for row in 0..lhs.len() {
        // let _guard = flame::start_guard("process_row");
        let a_val: AnyValue<'_> = lhs.get(row)?;
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

        // Convert to Polars Series for vectorized operations
        let x_a = Series::new(PlSmallStr::from_str("y_a"), points_a.column(0).to_vec());
        let y_a = Series::new(PlSmallStr::from_str("y_a"), points_a.column(1).to_vec());
        let x_b = Series::new(PlSmallStr::from_str("x_b"), points_b.column(0).to_vec());
        let y_b = Series::new(PlSmallStr::from_str("y_b"), points_b.column(1).to_vec());

        // Calculate distances using vectorized operations
        let mut min_distances = Vec::with_capacity(points_a.nrows());
        
        for i in 0..points_a.nrows() {
            // let _guard = flame::start_guard("process_point");
            let x_a_val = x_a.get(i).unwrap().extract::<f64>().unwrap();
            let y_a_val = y_a.get(i).unwrap().extract::<f64>().unwrap();
            
            let x_diff = &x_b - x_a_val;
            let y_diff = &y_b - y_a_val;
            
            // Vectorized distance calculation
            let x_squared = x_diff.f64()? * x_diff.f64()?;
            let y_squared = y_diff.f64()? * y_diff.f64()?;
            // TODO: use actual L2 distance
            let distances = x_squared + y_squared;
            
            // Get minimum distance
            let min_dist = distances.min().unwrap();
            min_distances.push(min_dist);
        }

        // Sum all minimum distances
        let total: f64 = min_distances.iter().sum();
        result_rows.push(total);
    }

    // Write the flamegraph to a file
    flame::dump_html(std::fs::File::create("flamegraph.html").unwrap()).unwrap();

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

// fn euclidean(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
//     ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
// }

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
