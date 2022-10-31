use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs::File;
use std::io::Read;
use std::path::Path;

mod activation;
mod image;
mod layer;
mod loss;

use crate::layer::Layer;
use crate::image::Image;
use crate::loss::*;
use crate::activation::*;

fn main() {
    // Read file and split data into rows then shuffle
    println!("Loading dataset...");
    let mut file = String::new();
    File::open(Path::new("train.csv"))
        .expect("File open failed")
        .read_to_string(&mut file)
        .expect("Read to string failed");
    let mut data: Vec<&str> = file.lines().skip(1).collect(); // Skip column labels
    let mut rng = thread_rng();
    data.shuffle(&mut rng);

    fn predict(input: &Vec<f64>, layers: &mut Vec<Layer>) -> Vec<f64> {
        let mut prev_output: Vec<f64> = input.clone();
        for layer in layers {
            prev_output = layer.propogate(&prev_output).to_vec();
        }
        prev_output
    }

    fn report(output: &Vec<f64>, img: &Image) {
        let predicted = output.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        let predicted_index = output.iter().position(|value| value == predicted).unwrap();

        println!(
            "Prediction: {0}, Confidence: {1}%",
            predicted_index,
            predicted * 100.0
        );
        println!("Actual: {0}", img.label);
    }

    println!("Initializing neural network...\n");

    let mut layers: Vec<Layer> = Vec::new();
    layers.push(Layer::new(784, 128, relu())); // Hidden 1
    layers.push(Layer::new(128, 10, softmax())); // Output

    let loss_fn = cross_entropy();
    let img = Image::new(data[0]);
    let result = predict(&img.values, &mut layers);
    let loss: f64 = (loss_fn.func)(&result, &img.y_values);

    report(&result, &img);
    println!("Loss: {:?}\n", loss);

    println!("loss deriv = {:?}", (loss_fn.deriv)(&result, &img.y_values));

    fn backprop(loss: &f64, loss_fn: &Loss, layers: &mut Vec<Layer>) {}
    backprop(&loss, &loss_fn, &mut layers);
}
