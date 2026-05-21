pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn percentile(mut values: Vec<f64>, p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (p / 100.0) * (values.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(values.len() - 1);
    let frac = idx - lo as f64;
    values[lo] * (1.0 - frac) + values[hi] * frac
}

pub fn trt_element_size(dtype: i32) -> usize {
    match dtype {
        0 => 4,  // kFLOAT
        1 => 2,  // kHALF
        2 => 4,  // kINT8 - unused
        3 => 4,  // kINT32
        4 => 1,  // kBOOL
        5 => 8,  // kINT64
        _ => 4,
    }
}

pub fn trt_volume(dims: &[i64]) -> i64 {
    dims.iter().product()
}
