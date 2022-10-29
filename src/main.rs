use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs::File;
use std::io::Read;
use std::path::Path;

mod layer;
use crate::layer::Layer;

mod image;
use crate::image::Image;

fn main() {
    println!("Loading dataset...");
    let mut file = String::new();
    File::open(Path::new("train.csv"))
        .expect("File open failed")
        .read_to_string(&mut file)
        .expect("Read to string failed");
    let mut data: Vec<&str> = file.lines().skip(1).collect(); // Skip column labels
    let mut rng = thread_rng();
    data.shuffle(&mut rng);

    println!("Initializing neural network...");

    fn predict(input: Vec<f32>, layers: Vec<Layer>) -> Vec<f32> {
        layers.iter().reduce(|acc, layer| {
            if acc.neurons != layer.inputs {
                panic!("Layer shapes do not match") // Sanity check
            }
            layer
        });

        let mut prev_output: Vec<f32> = input;
        for mut layer in layers {
            prev_output = layer.propogate(prev_output.clone()).to_vec();
        }

        prev_output
    }

    fn report(output: Vec<f32>, img: Image) {
        let predicted = output.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        let predicted_index = output.iter().position(|value| value == predicted).unwrap();

        println!(
            "Prediction: {0}, Confidence: {1}%",
            predicted_index,
            predicted * 100.0
        );
        println!("Actual: {0}", img.label);
    }

    fn loss(output: Vec<f32>, expected: Vec<f32>) -> f32 {
        let loss = output
            .iter()
            .zip(expected.iter())
            .map(|(p, y)| p.log10() * y) // log(predicted) * actual
            .sum::<f32>();
        loss * -1.0
    }

    let mut layers: Vec<Layer> = Vec::new();
    layers.push(Layer::new(784, 10, |xn, _| xn.max(0.0))); // Hidden
    layers.push(Layer::new(10, 10, |xn, x| softmax(xn, x))); // Output
    fn softmax(xn: f32, x: Vec<f32>) -> f32 {
        xn.exp() / x.iter().map(|x| x.exp()).sum::<f32>()
    }

    let img = Image::new(data[0]);
    let result = predict(img.clone().values, layers.clone());
    let loss: f32 = loss(result.clone(), img.clone().y_values);

    report(result.clone(), img.clone());
    println!("{:?}", loss);
}
