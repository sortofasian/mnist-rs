#![allow(unreachable_code)]
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::{fs::File, io::Read, path::Path};

mod functions;
mod image;
mod layer;

use crate::functions::{Cost, Crossent, Relu, Softmax};
use crate::image::Image;
use crate::layer::{Delta, Layer};

const A: f32 = 0.0001;
const BATCH_SIZE: usize = 10;

fn main() {
    // Read file and split data into rows then shuffle
    eprintln!("Loading dataset...");
    let mut file = String::new();
    File::open(Path::new("train.csv"))
        .expect("File open failed")
        .read_to_string(&mut file)
        .expect("Read to string failed");
    let mut data: Vec<&str> = file.lines().skip(1).collect(); // Skip column labels
    let mut rng = thread_rng();
    data.shuffle(&mut rng);

    let (mut l1, mut l2) = (
        Layer::new(784, 64, Box::new(Relu)),
        Layer::new(64, 10, Box::new(Softmax)),
    );
    let cost = Crossent;

    fn calc_b(l: &Layer, db: &Vec<f32>) -> Vec<f32> {
        l.bias()
            .par_iter()
            .zip(db.par_iter())
            .map(|(b, db)| b - A * db)
            .collect()
    }
    fn calc_w(l: &Layer, dw: &Vec<f32>) -> Vec<Vec<f32>> {
        l.weights()
            .par_iter()
            .zip(dw.par_iter())
            .map(|(w, dw)| w.par_iter().map(|w| w - A * dw).collect())
            .collect()
    }

    let imgs: Vec<Image> = data.par_iter().map(|data| Image::new(data)).collect();
    loop {
        for imgs in imgs.chunks(BATCH_SIZE) {
            let mut c: Vec<f32> = Vec::new();
            let mut grad: Vec<(Delta, Delta)> = Vec::new();
            for img in imgs.iter() {
                let (x, y) = (&img.values, &img.y_values);
                let y1 = l1.calc(&x);
                let y2 = l2.calc(&y1);
                let dc = cost.df(&y2, &y);
                let (dx2, dl2) = l2.back(&y2, &dc);
                let (_, dl1) = l1.back(&y1, &dx2);
                c.push(cost.f(&y2, &y));
                grad.push((dl1, dl2));
            }
            print!("{}\r", c.par_iter().sum::<f32>() / c.len() as f32);
            let len = grad.len();
            let sum = grad
                .into_par_iter()
                .reduce_with(|(adl1, adl2), (dl1, dl2)| {
                    (
                        Delta {
                            bias: adl1
                                .bias
                                .iter()
                                .zip(dl1.bias.iter())
                                .map(|(adl1, dl1)| adl1 + dl1)
                                .collect(),
                            weights: adl1
                                .weights
                                .iter()
                                .zip(dl1.weights.iter())
                                .map(|(adl1, dl1)| adl1 + dl1)
                                .collect(),
                        },
                        Delta {
                            bias: adl2
                                .bias
                                .iter()
                                .zip(dl2.bias.iter())
                                .map(|(adl1, dl1)| adl1 + dl1)
                                .collect(),
                            weights: adl2
                                .weights
                                .iter()
                                .zip(dl2.weights.iter())
                                .map(|(adl1, dl1)| adl1 + dl1)
                                .collect(),
                        },
                    )
                })
                .unwrap();
            let dl1 = Delta {
                weights: sum.0.weights.par_iter().map(|w| w / len as f32).collect(),
                bias: sum.0.weights.par_iter().map(|b| b / len as f32).collect(),
            };
            let dl2 = Delta {
                weights: sum.1.weights.par_iter().map(|w| w / len as f32).collect(),
                bias: sum.1.weights.par_iter().map(|b| b / len as f32).collect(),
            };

            l2.update_weights(calc_w(&l2, &dl2.weights)).unwrap();
            l2.update_bias(calc_b(&l2, &dl2.bias)).unwrap();
            l1.update_weights(calc_w(&l1, &dl1.weights)).unwrap();
            l1.update_bias(calc_b(&l1, &dl1.bias)).unwrap();
        }
    }
}
