use crate::functions::Function;
use rand::{distributions::Uniform, thread_rng, Rng};
use rayon::prelude::*;

pub struct Layer {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    function: Box<dyn Function>,
    pub inputs: usize,
    pub outputs: usize,
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize, function: Box<dyn Function>) -> Layer {
        let dist = Uniform::new(-1.0, 1.0);

        let (weights, bias) = (0..outputs)
            .par_bridge()
            .map(|_| {
                let weights = (0..inputs)
                    .par_bridge()
                    .map(|_| thread_rng().sample(dist))
                    .collect();
                let bias = thread_rng().sample(dist);
                (weights, bias)
            })
            .collect();

        Layer {
            weights,
            bias,
            function,
            inputs,
            outputs,
        }
    }

    pub fn weights(&self) -> &Vec<Vec<f32>> {
        &self.weights
    }
    pub fn update_weights(&mut self, weights: Vec<Vec<f32>>) -> Result<(), &str> {
        if (weights.len() != self.outputs) | (weights[0].len() != self.inputs) {
            return Err("Invalid weights");
        }

        self.weights = weights;
        Ok(())
    }
    pub fn bias(&self) -> &Vec<f32> {
        &self.bias
    }
    pub fn update_bias(&mut self, bias: Vec<f32>) -> Result<(), &str> {
        if bias.len() != self.outputs {
            return Err("Invalid bias");
        }

        self.bias = bias;
        Ok(())
    }

    pub fn calc(&self, input: &Vec<f32>) -> Vec<f32> {
        if input.len() != self.inputs {
            panic!("Invalid input")
        }

        let z = &self
            .weights
            .par_iter()
            .zip(self.bias.par_iter())
            .map(|(w, b)| {
                w.par_iter()
                    .zip(input.par_iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
                    + b
            })
            .collect();
        let output = self.function.f(z);

        output
    }

    pub fn back(&self, input: &Vec<f32>, d_parent: &Vec<f32>) -> (Vec<f32>, Delta) {
        let hypothesis = &self
            .weights
            .par_iter()
            .zip(self.bias.par_iter())
            .map(|(w, b)| {
                w.par_iter()
                    .zip(input.par_iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
                    + b
            })
            .collect();

        let df_dz = self.function.df(&hypothesis);

        let dp_dz: Vec<f32> = df_dz
            .par_iter()
            .zip(d_parent.par_iter())
            .map(|(dz, dp)| dz * dp)
            .collect();

        let dp_dw: Vec<f32> = input
            .par_iter()
            .zip(dp_dz.par_iter())
            .map(|(x, dz)| x * dz)
            .collect();

        // dp_db = db_dz

        let mut t_weights: Vec<Vec<f32>> = (0..self.inputs)
            .map(|_| Vec::with_capacity(self.outputs))
            .collect();
        for weights in self.weights.iter() {
            for (i, weight) in weights.iter().enumerate() {
                t_weights[i].push(*weight)
            }
        }

        let dp_dx = t_weights
            .par_iter()
            .zip(dp_dz.par_iter())
            .map(|(w, dz)| w.iter().map(|w| w * dz).sum())
            .collect();

        let delta = Delta {
            weights: dp_dw,
            bias: dp_dz,
        };
        (dp_dx, delta)
    }
}
pub struct Delta {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}
