use criterion::{black_box, criterion_group, criterion_main, Criterion};
use polars::prelude::*;
use pg::match_nearest_point;

fn benchmark_match_nearest_point(c: &mut Criterion) {
    // Create test data
    let num_tracks = 100;
    let points_per_track = 50;
    
    // Create track IDs
    let track_ids: Vec<i32> = (0..num_tracks).collect();
    let track_id_series = Series::new(PlSmallStr::from_str("track_id"), track_ids);
    
    // Create x, y, and timestamp lists for each track
    let mut x_lists = Vec::with_capacity(num_tracks as usize);
    let mut y_lists = Vec::with_capacity(num_tracks as usize);
    let mut t_lists = Vec::with_capacity(num_tracks as usize);
    
    for i in 0..num_tracks {
        let x: Vec<f32> = (0..points_per_track).map(|j| (i + j) as f32).collect();
        let y: Vec<f32> = (0..points_per_track).map(|j| (i + j) as f32).collect();
        let t: Vec<i64> = (0..points_per_track).map(|j| (i + j) as i64).collect();
        
        x_lists.push(Series::new(PlSmallStr::from_str("x"), x));
        y_lists.push(Series::new(PlSmallStr::from_str("y"), y));
        t_lists.push(Series::new(PlSmallStr::from_str("t"), t));
    }
    
    let x_list_series = Series::new(PlSmallStr::from_str("x_list"), x_lists);
    let y_list_series = Series::new(PlSmallStr::from_str("y_list"), y_lists);
    let t_list_series = Series::new(PlSmallStr::from_str("t_list"), t_lists);
    
    // Create overlap pairs
    let num_pairs = 1000;
    let track_id_1: Vec<i32> = (0..num_pairs).map(|i| (i % num_tracks) as i32).collect();
    let track_id_2: Vec<i32> = (0..num_pairs).map(|i| ((i + 1) % num_tracks) as i32).collect();
    let overlap_start: Vec<i64> = (0..num_pairs).map(|i| i as i64).collect();
    let overlap_end: Vec<i64> = (0..num_pairs).map(|i| (i + 10) as i64).collect();
    
    let track_id_1_series = Series::new(PlSmallStr::from_str("track_id_1"), track_id_1);
    let track_id_2_series = Series::new(PlSmallStr::from_str("track_id_2"), track_id_2);
    let overlap_start_series = Series::new(PlSmallStr::from_str("overlap_start"), overlap_start);
    let overlap_end_series = Series::new(PlSmallStr::from_str("overlap_end"), overlap_end);
    
    c.bench_function("match_nearest_point", |b| {
        b.iter(|| {
            let inputs = &[
                black_box(track_id_1_series.clone()),
                black_box(track_id_2_series.clone()),
                black_box(overlap_start_series.clone()),
                black_box(overlap_end_series.clone()),
                black_box(track_id_series.clone()),
                black_box(x_list_series.clone()),
                black_box(y_list_series.clone()),
                black_box(t_list_series.clone()),
            ];
            match_nearest_point(inputs).unwrap()
        })
    });
}

criterion_group!(benches, benchmark_match_nearest_point);
criterion_main!(benches); 