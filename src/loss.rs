pub struct Loss {
    pub func: fn(result: &Vec<f64>, expected: &Vec<f64>) -> f64,
    pub deriv: fn(result: &Vec<f64>, expected: &Vec<f64>) -> Vec<f64>,
}

pub fn cross_entropy() -> Loss {
    Loss {
        func: |y_p, y| {
            y_p.iter()
                .zip(y)
                .map(|(y_p, y)| y_p.log10() * y)
                .sum::<f64>()
                * -1.0
        },
        deriv: |y_p, y| {
            y_p.iter()
                .zip(y)
                .map(|(y_p, y)| y / (y_p * 10_f64.ln()))
                .collect()
        },
    }
}
