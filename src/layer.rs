use rand::thread_rng;
use rand::Rng;

#[derive(Clone)]
pub struct Layer {
    pub inputs: u16,
    pub outputs: u16,
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
    pub activation: fn(f32) -> f32,
}
impl Layer {
    pub fn new(inputs: u16, outputs: u16, activation: fn(f32) -> f32) -> Self {
        let mut rng = thread_rng();

        // Initialize arrays for weights and biases with random values from -1.0 to 1.0;
        let mut weights: Vec<Vec<f32>> = Vec::new();
        for _ in 0..outputs {
            let mut weights_n: Vec<f32> = Vec::new();
            for _ in 0..inputs {
                weights_n.push(rng.gen_range(-1.0..1.0));
            }
            weights.push(weights_n);
        }

        let mut biases: Vec<f32> = Vec::new();
        for _ in 0..outputs {
            biases.push(rng.gen_range(-1.0..1.0));
        }

        Layer {
            inputs,
            outputs,
            weights,
            biases,
            activation,
        }
    }
}
