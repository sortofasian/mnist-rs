pub fn relu(x: &f32, _: &Vec<f32>) -> f32 {
    x.max(0.0) // x for all values > 0
}
pub fn d_relu(x: &f32, _: &Vec<f32>) -> f32 {
    x.clamp(0.0, 1.0).ceil() // 1 for all values > 0
}

pub fn softmax(xn: &f32, x: &Vec<f32>) -> f32 {
    xn.exp() / x.iter().map(|x| x.exp()).sum::<f32>() // e^x(n) / sum(e^x) where x is an array
}
pub fn d_softmax(x: &Vec<f32>) -> Vec<f32> {
    let exp: Vec<f32> = x.iter().map(|x| x.exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter()
        .map(|exp| (exp * sum - exp.powi(2)) / sum.powi(2)) // sum(x) * x(n) - x(n)^2 / sum^2
        .collect() // Could be simplified
}

pub fn cross_entropy(output: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    let loss: f32 = output
        .iter()
        .zip(expected.iter())
        .map(|(p, y)| p.log10() * y) // log(predicted) * actual
        .sum();
    loss * -1.0
}
pub fn d_cross_entropy(output: &Vec<f32>, expected: &Vec<f32>) -> Vec<f32> {
    output
        .iter()
        .zip(expected)
        .map(|(p, y)| y / (p * 10_f32.ln())) // y/ (p_n * ln(10))
        .collect()
}
