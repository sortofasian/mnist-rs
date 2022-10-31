use rand::thread_rng;
use rand::Rng;

use crate::activation::Activation;

#[derive(Clone)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: Activation,
    pub hypothesis: Vec<f64>,
    pub output: Vec<f64>,
}

impl Layer {
    pub fn new(inputs: u16, neurons: u16, activation: Activation, ) -> Self {
        let mut rng = thread_rng();

        // Initialize arrays for weights and biases with random values from -1.0 to 1.0;
        let mut weights: Vec<Vec<f64>> = Vec::new();
        for _ in 0..neurons {
            let mut weights_n: Vec<f64> = Vec::new();
            for _ in 0..inputs {
                weights_n.push(rng.gen_range(-1.0..1.0));
            }
            weights.push(weights_n);
        }

        let mut biases: Vec<f64> = Vec::new();
        for _ in 0..neurons {
            biases.push(rng.gen_range(-1.0..1.0));
        }

        Layer {
            weights,
            biases,
            activation,
            hypothesis: Vec::new(),
            output: Vec::new(),
        }
    }

    pub fn propogate(&mut self, input: &Vec<f64>) -> &Vec<f64> {
        if input.len() != self.weights[0].len() {
            panic!("Input vector is not the right shape")
        }

        self.hypothesis = self
            .biases
            .iter()
            .zip(&self.weights)
            .map(|(b, weights)| {
                let input_weighted: f64 = input.iter().zip(weights).map(|(x, w)| x * w).sum();
                input_weighted + b
            })
            .collect();

        self.output = self.hypothesis
            .iter()
            .map(|input| (self.activation.func)(input, &self.hypothesis))
            .collect();

        &self.output
    }
}
