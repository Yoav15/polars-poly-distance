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
use ndarray::{Array2};
use rayon::prelude::*;
use std::sync::Mutex;
// use flame;

#[polars_expr(output_type = Float64)]
fn match_nearest_point(inputs: &[Series]) -> PolarsResult<Series> {
    let x1 = &inputs[0];
    let y1 = &inputs[1];
    let t1 = &inputs[2];
    let x2 = &inputs[3];
    let y2 = &inputs[4];
    let t2 = &inputs[5];
    let overlap_start = &inputs[6];
    let overlap_end = &inputs[7];

    let result_rows = Mutex::new(Vec::with_capacity(x1.len()));

    // Process rows in parallel
    (0..x1.len()).into_par_iter().for_each(|row| {
        let x1_val = x1.get(row).unwrap();
        let y1_val = y1.get(row).unwrap();
        let t1_val = t1.get(row).unwrap();
        let x2_val = x2.get(row).unwrap();
        let y2_val = y2.get(row).unwrap();
        let t2_val = t2.get(row).unwrap();
        let start = match overlap_start.get(row).unwrap() {
            AnyValue::Float64(v) => v,
            _ => return,
        };
        let end = match overlap_end.get(row).unwrap() {
            AnyValue::Float64(v) => v,
            _ => return,
        };

        // Extract and convert lists
        let x1_series = match x1_val {
            AnyValue::List(s) => s,
            _ => return,
        };
        let y1_series = match y1_val {
            AnyValue::List(s) => s,
            _ => return,
        };
        let t1_series = match t1_val {
            AnyValue::List(s) => s,
            _ => return,
        };
        let x2_series = match x2_val {
            AnyValue::List(s) => s,
            _ => return,
        };
        let y2_series = match y2_val {
            AnyValue::List(s) => s,
            _ => return,
        };
        let t2_series = match t2_val {
            AnyValue::List(s) => s,
            _ => return,
        };

        // Convert to f64 series
        let x1_f64 = x1_series.cast(&DataType::Float64).unwrap();
        let y1_f64 = y1_series.cast(&DataType::Float64).unwrap();
        let t1_f64 = t1_series.cast(&DataType::Float64).unwrap();
        let x2_f64 = x2_series.cast(&DataType::Float64).unwrap();
        let y2_f64 = y2_series.cast(&DataType::Float64).unwrap();
        let t2_f64 = t2_series.cast(&DataType::Float64).unwrap();

        // Get points as vectors
        let x1_points = x1_f64.f64().unwrap();
        let y1_points = y1_f64.f64().unwrap();
        let t1_points = t1_f64.f64().unwrap();
        let x2_points = x2_f64.f64().unwrap();
        let y2_points = y2_f64.f64().unwrap();
        let t2_points = t2_f64.f64().unwrap();

        // Pre-filter points in overlap region for both tracks
        let mut valid_indices_1 = Vec::with_capacity(t1_points.len());
        let mut valid_indices_2 = Vec::with_capacity(t2_points.len());
        
        for i in 0..t1_points.len() {
            let t = t1_points.get(i).unwrap();
            if t >= start && t <= end {
                valid_indices_1.push(i);
            }
        }
        
        for i in 0..t2_points.len() {
            let t = t2_points.get(i).unwrap();
            if t >= start && t <= end {
                valid_indices_2.push(i);
            }
        }

        // Calculate distances using vectorized operations
        let mut total_distance = 0.0;
        let mut count = 0;

        for &i in &valid_indices_1 {
            let t1_val = t1_points.get(i).unwrap();
            let x1_val = x1_points.get(i).unwrap();
            let y1_val = y1_points.get(i).unwrap();

            // Find closest time point in track 2
            let mut min_time_diff = f64::MAX;
            let mut closest_j = 0;
            
            for &j in &valid_indices_2 {
                let t2_val = t2_points.get(j).unwrap();
                let time_diff = (t1_val - t2_val).abs();
                if time_diff < min_time_diff {
                    min_time_diff = time_diff;
                    closest_j = j;
                }
            }

            if min_time_diff != f64::MAX {
                let x2_val = x2_points.get(closest_j).unwrap();
                let y2_val = y2_points.get(closest_j).unwrap();
                
                // Use SIMD-friendly distance calculation
                let dx = x1_val - x2_val;
                let dy = y1_val - y2_val;
                total_distance += (dx * dx + dy * dy).sqrt();
                count += 1;
            }
        }

        let avg_distance = if count > 0 { total_distance / count as f64 } else { 0.0 };
        result_rows.lock().unwrap().push(avg_distance);
    });

    Ok(Series::new(PlSmallStr::from_str("result"), result_rows.into_inner().unwrap()))
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
