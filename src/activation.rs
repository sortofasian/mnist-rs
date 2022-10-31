#[derive(Clone)]
pub struct Activation {
    pub func: fn(xn: &f64, x: &Vec<f64>) -> f64,
    pub deriv: fn(xn: &f64, x: &Vec<f64>) -> f64,
}

pub fn relu() -> Activation {
    Activation {
        func: |x, _| x.max(0.0),                // x for all values > 0
        deriv: |x, _| x.clamp(0.0, 1.0).ceil(), // 1 for all values > 0
    }
}

pub fn softmax() -> Activation {
    Activation {
        func: |xn, x| {
            // Normalize exponents to prevent NaN
            let max = x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

            let exp = (xn - max).exp();
            let exp_sum: f64 = x.iter().map(|xn| (xn - max).exp()).sum();
            exp / exp_sum // e^(xn - max) / sum(e^(xn - max))
        },
        deriv: |xn, x| {
            let sum: f64 = x.iter().map(|x| x.exp()).sum();
            let exp: f64 = xn.exp();
            (sum * exp - exp.powi(2)) / sum.powi(2) // (sum(e^xn)*e^xn - (e^xn)^2) / sum(e^xn)^2
        },
    }
}
