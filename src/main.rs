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
    let mut data = file.lines().skip(1); // Skip column labels

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

    fn normalize(layers: Vec<Layer>, img: Image) -> Vec<f32> {
        fn softmax(x: Vec<f32>, x_n: f32) -> f32 {
            x_n.exp() / x.iter().map(|x| x.exp()).sum::<f32>()
        }

        let output = predict(img.values, layers);
        output
            .iter()
            .map(|value| softmax(output.clone(), *value))
            .collect::<Vec<f32>>()
    }

    fn report(output: Vec<f32>, img: Image) {
        let predicted = output.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        let predicted_index = output.iter().position(|value| value == predicted).unwrap();

        println!(
            "Prediction: {0}, Confidence: {1}%",
            predicted_index + 1,
            predicted * 100.0
        );
        println!("Actual: {0}", img.label);
    }

    let mut layers: Vec<Layer> = Vec::new();
    layers.push(Layer::new(784, 10, |x| x.max(0.0))); // Hidden
    layers.push(Layer::new(10, 10, |x| x)); // Output

    let img = Image::new(data.next().unwrap());
    let result = normalize(layers.clone(), img.clone());
    report(result.clone(), img.clone());

    let mut correct_vector = [0_f32; 10].to_vec();
    correct_vector[(img.label - 1) as usize] = 1.0;
    let loss: f32 = (result
        .iter()
        .zip(correct_vector.iter())
        .map(|(p, y)| p.log10() * y)
        .sum::<f32>())
        * -1.0
        / correct_vector.len() as f32;
    println!("{:?}", loss);
}
