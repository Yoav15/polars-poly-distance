#![allow(clippy::unused_unit)]
use polars::error::PolarsError;
use polars::prelude::NamedFrom;
use pyo3_polars::derive::polars_expr;
use polars::prelude::PlSmallStr;
use polars::prelude::Series;
// use polars::prelude::AnyValue;
use polars::prelude::PolarsResult;
use polars::prelude::CompatLevel;
// use rayon::prelude::*;
// use polars::datatypes::Float32Chunked;
// use polars::datatypes::DatetimeChunked;
// use std::collections::HashMap;
use polars_arrow::array::Float32Array;
// use std::time::Instant;
// use flame;

// // The actual implementation
// #[polars_expr(output_type = Float32)]
// pub fn poly_dist(inputs: &[Series]) -> PolarsResult<Series> {
//     let track_id_1 = &inputs[0];
//     let track_id_2 = &inputs[1];
//     let overlap_start = &inputs[2];
//     let overlap_end = &inputs[3];
//     let track_ids = &inputs[4];
//     let x_lists = &inputs[5];
//     let y_lists = &inputs[6];
//     let t_lists = &inputs[7];

//     // Pre-compute track data mapping for faster lookups
//     let track_data_map: HashMap<i32, (Float32Chunked, Float32Chunked, DatetimeChunked)> =
//     (0..track_ids.len())
//         .filter_map(|i| {
//             let id = match track_ids.get(i).ok()? {
//                 AnyValue::Int32(v) => v,
//                 _ => return None,
//             };

//             let x = match x_lists.get(i).ok()? {
//                 AnyValue::List(s) => s.f32().ok()?.clone(),
//                 _ => return None,
//             };

//             let y = match y_lists.get(i).ok()? {
//                 AnyValue::List(s) => s.f32().ok()?.clone(),
//                 _ => return None,
//             };

//             let t = match t_lists.get(i).ok()? {
//                 AnyValue::List(s) => s.datetime().ok()?.clone(),
//                 _ => return None,
//             };

//             Some((id, (x, y, t)))
//         })
//         .collect();

//     let result_vec: Vec<f32> = (0..track_id_1.len()).into_par_iter().map(|row| {
//         // Get track IDs and overlap period
//         let (id1, id2, start, end) = match (
//             track_id_1.get(row),
//             track_id_2.get(row),
//             overlap_start.get(row),
//             overlap_end.get(row)
//         ) {
//             (
//                 Ok(AnyValue::Int32(i1)),
//                 Ok(AnyValue::Int32(i2)),
//                 Ok(AnyValue::Datetime(s, _, _)),
//                 Ok(AnyValue::Datetime(e, _, _))
//             ) => (i1, i2, s, e),
//             _ => return 0.0,
//         };

//         // Find track data using pre-computed map
//         let (x1_vals, y1_vals, t1_vals) = match track_data_map.get(&id1) {
//             Some((x, y, t)) => match (x.cont_slice(), y.cont_slice(), t.cont_slice()) {
//                 (Ok(x), Ok(y), Ok(t)) => (x, y, t),
//                 _ => return 0.0,
//             },
//             None => return 0.0,
//         };

//         let (x2_vals, y2_vals, t2_vals) = match track_data_map.get(&id2) {
//             Some((x, y, t)) => match (x.cont_slice(), y.cont_slice(), t.cont_slice()) {
//                 (Ok(x), Ok(y), Ok(t)) => (x, y, t),
//                 _ => return 0.0,
//             },
//             None => return 0.0,
//         };
//         // Filter points within overlap period
//         let valid_1: Vec<usize> = (0..t1_vals.len())
//             .filter(|&i| t1_vals[i] >= start && t1_vals[i] <= end)
//             .collect();

//         let valid_2: Vec<usize> = (0..t2_vals.len())
//             .filter(|&i| t2_vals[i] >= start && t2_vals[i] <= end)
//             .collect();

//         if valid_1.is_empty() || valid_2.is_empty() {
//             return 0.0;
//         }

//         let mut total_distance = 0.0f32;
//         let mut count = 0;

//         for &i in &valid_1 {
//             let t1v = t1_vals[i];
//             let x1v = x1_vals[i];
//             let y1v = y1_vals[i];

//             // Binary search for closest t2
//             let mut left = 0;
//             let mut right = valid_2.len();
//             while left < right {
//                 let mid = (left + right) / 2;
//                 let t2v = t2_vals[valid_2[mid]];
//                 if t2v < t1v {
//                     left = mid + 1;
//                 } else {
//                     right = mid;
//                 }
//             }

//             let j_closest = if left == 0 {
//                 valid_2[0]
//             } else if left >= valid_2.len() {
//                 valid_2[valid_2.len() - 1]
//             } else {
//                 let before = valid_2[left - 1];
//                 let after = valid_2[left];
//                 let d1 = (t1v - t2_vals[before]).abs();
//                 let d2 = (t1v - t2_vals[after]).abs();
//                 if d1 < d2 { before } else { after }
//             };

//             let dx = x1v - x2_vals[j_closest];
//             let dy = y1v - y2_vals[j_closest];
//             total_distance += (dx * dx + dy * dy).sqrt();
//             count += 1;
//         }

//         if count > 0 {
//             total_distance / count as f32
//         } else {
//             0.0
//         }
//     }).collect();

//     Ok(Series::new(PlSmallStr::from_str("result"), result_vec))
// }


#[polars_expr(output_type = Float32)]
pub fn mini_rocket_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let series_list = inputs[0].list()?;      // List[Float32]
    let kernels_list = inputs[1].list()?;     // List[Float32]
    let dilations = inputs[2].f32()?;         // Float32 Series
    let biases = inputs[3].f32()?;            // Float32 Series

    let mut features: Vec<Option<Series>> = Vec::with_capacity(series_list.len());

    for ts in series_list.into_iter() {

        let time_series_series = ts.expect("Time series missing");

        let time_series: Vec<f32> = time_series_series
            .f32()
            .map_err(|_| PolarsError::ComputeError("Expected Float32 series".into()))?
            .into_no_null_iter()
            .collect();

        let mut ppv_vec = Vec::with_capacity(kernels_list.len());

        for ((kernel_val, dilation_val), bias_val) in kernels_list.iter().zip(dilations.into_iter()).zip(biases.into_iter()) {

            let kernel_series = kernel_val.expect("Kernel missing");
            let float_array = kernel_series
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| PolarsError::ComputeError("Expected Float32Array".into()))?;

            let kernel_vec: Vec<f32> = (0..float_array.len())
                .map(|i| float_array.value(i))
                .collect();


            let dilation = dilation_val.expect("Missing dilation") as usize;
            let bias = bias_val.expect("Missing bias");

            let klen = kernel_vec.len();
            let max_i = time_series.len().saturating_sub(klen * dilation);

            let mut pos_count = 0;
            let mut total_count = 0;

            for i in 0..=max_i {
                let mut conv = 0.0f32;
                for (j, &kval) in kernel_vec.iter().enumerate() {
                    let ts_idx = i + j * dilation;
                    if ts_idx < time_series.len() {
                        conv += kval * time_series[ts_idx];
                    } else {
                    }
                }
                if conv > bias {
                    pos_count += 1;
                }
                total_count += 1;
            }

            let ppv = if total_count > 0 {
                pos_count as f32 / total_count as f32
            } else {
                0.0
            };
            ppv_vec.push(ppv);
        }

        let ppv_series = Series::new(PlSmallStr::from_str(""), ppv_vec);
        features.push(Some(ppv_series));
    }

    Ok(Series::new(PlSmallStr::from_str("features"), features))
}
