use rand::thread_rng;
use rand::Rng;

#[derive(Clone)]
pub struct Layer {
    pub inputs: u16,
    pub neurons: u16,
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
    pub activation: fn(&f32, &Vec<f32>) -> f32,
    pub output: Vec<f32>,
}

impl Layer {
    pub fn new(inputs: u16, neurons: u16, activation: fn(&f32, &Vec<f32>) -> f32) -> Self {
        let mut rng = thread_rng();

        // Initialize arrays for weights and biases with random values from -1.0 to 1.0;
        let mut weights: Vec<Vec<f32>> = Vec::new();
        for _ in 0..neurons {
            let mut weights_n: Vec<f32> = Vec::new();
            for _ in 0..inputs {
                weights_n.push(rng.gen_range(-1.0..1.0));
            }
            weights.push(weights_n);
        }

        let mut biases: Vec<f32> = Vec::new();
        for _ in 0..neurons {
            biases.push(rng.gen_range(-1.0..1.0));
        }

        Layer {
            inputs,
            neurons,
            weights,
            biases,
            activation,
            output: Vec::new(),
        }
    }

    pub fn propogate(&mut self, input: Vec<f32>) -> Vec<f32> {
        if input.len() != self.inputs.into() {
            panic!("Input vector is not the right shape")
        }

        let inputs_biased: Vec<f32> = self
            .biases
            .iter()
            .zip(&self.weights)
            .map(|(b, weights)| {
                let input_weighted: f32 = input.iter().zip(weights).map(|(x, w)| x * w).sum();
                input_weighted + b
            })
            .collect();

        self.output = inputs_biased
            .iter()
            .map(|input| (self.activation)(input, &inputs_biased.clone()))
            .collect();

        self.output.clone()
    }
}
