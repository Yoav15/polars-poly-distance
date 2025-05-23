#![allow(clippy::unused_unit)]
use polars::prelude::NamedFrom;
use pyo3_polars::derive::polars_expr;
use polars::prelude::PlSmallStr;
use polars::prelude::Series;
use polars::prelude::AnyValue;
use polars::prelude::PolarsResult;
use polars::prelude::CompatLevel;
use rayon::prelude::*;
use polars::datatypes::Float32Chunked;
// use std::time::Instant;
// use flame;

#[polars_expr(output_type = Float32)]
fn match_nearest_point(inputs: &[Series]) -> PolarsResult<Series> {
    // let start_total = Instant::now();
    let x1 = &inputs[0];
    let y1 = &inputs[1];
    let t1 = &inputs[2];
    let x2 = &inputs[3];
    let y2 = &inputs[4];
    let t2 = &inputs[5];
    let overlap_start = &inputs[6];
    let overlap_end = &inputs[7];

    let result_vec: Vec<f32> = (0..x1.len()).into_par_iter().map(|row| {
        // let start_row = Instant::now();

        // Unwrap scalar start/end
        let (start, end) = match (overlap_start.get(row), overlap_end.get(row)) {
            (Ok(AnyValue::Float32(s)), Ok(AnyValue::Float32(e))) => (s, e),
            _ => return 0.0,
        };

        let get_chunk = |series: &Series| -> Option<Float32Chunked> {
            match series.get(row).ok()? {
                AnyValue::List(inner) => inner.f32().ok().map(|ca| ca.clone()),
                _ => None,
            }
        };

        let (x1_chunk, y1_chunk, t1_chunk, x2_chunk, y2_chunk, t2_chunk) = match (
            get_chunk(x1),
            get_chunk(y1),
            get_chunk(t1),
            get_chunk(x2),
            get_chunk(y2),
            get_chunk(t2),
        ) {
            (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) => (a, b, c, d, e, f),
            _ => return 0.0,
        };

        let (x1_vals, y1_vals, t1_vals, x2_vals, y2_vals, t2_vals) = match (
            x1_chunk.cont_slice().ok(),
            y1_chunk.cont_slice().ok(),
            t1_chunk.cont_slice().ok(),
            x2_chunk.cont_slice().ok(),
            y2_chunk.cont_slice().ok(),
            t2_chunk.cont_slice().ok(),
        ) {
            (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) => (a, b, c, d, e, f),
            _ => return 0.0,
        };

        // let start_filter = Instant::now();
        let valid_1: Vec<usize> = (0..t1_vals.len())
            .filter(|&i| t1_vals[i] >= start && t1_vals[i] <= end)
            .collect();

        let valid_2: Vec<usize> = (0..t2_vals.len())
            .filter(|&i| t2_vals[i] >= start && t2_vals[i] <= end)
            .collect();
        // let filter_time = start_filter.elapsed();

        if valid_1.is_empty() || valid_2.is_empty() {
            return 0.0;
        }

        // let start_calc = Instant::now();
        let mut total_distance = 0.0f32;
        let mut count = 0;

        for &i in valid_1.iter() {
            // let start_point = Instant::now();
            let t1v = t1_vals[i];
            let x1v = x1_vals[i];
            let y1v = y1_vals[i];
            // let point_time = start_point.elapsed();

            // Inline binary search for closest t2
            // let start_search = Instant::now();
            let mut left = 0;
            let mut right = valid_2.len();
            while left < right {
                let mid = (left + right) / 2;
                let t2v = t2_vals[valid_2[mid]];
                if t2v < t1v {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }

            let j_closest = if left == 0 {
                valid_2[0]
            } else if left >= valid_2.len() {
                valid_2[valid_2.len() - 1]
            } else {
                let before = valid_2[left - 1];
                let after = valid_2[left];
                let d1 = (t1v - t2_vals[before]).abs();
                let d2 = (t1v - t2_vals[after]).abs();
                if d1 < d2 { before } else { after }
            };
            // let search_time = start_search.elapsed();

            // let start_dist = Instant::now();
            let dx = x1v - x2_vals[j_closest];
            let dy = y1v - y2_vals[j_closest];
            total_distance += (dx * dx + dy * dy).sqrt();
            // let dist_time = start_dist.elapsed();
            count += 1;

            // if count % 100 == 0 {
            //     println!(
            //         "Point {} timing: get_point={:?}, search={:?}, dist={:?}", 
            //         count, point_time, search_time, dist_time
            //     );
            // }
        }

        // let calc_time = start_calc.elapsed();
        // let row_time = start_row.elapsed();

        // println!(
        //     "Row {} timing: filter={:?}, calc={:?}, total={:?}",
        //     row, filter_time, calc_time, row_time
        // );

        if count > 0 {
            total_distance / count as f32
        } else {
            0.0
        }
    }).collect();

    // let total_time = start_total.elapsed();
    // println!("Total execution time: {:?}", total_time);

    Ok(Series::new(PlSmallStr::from_str("result"), result_vec))
}
