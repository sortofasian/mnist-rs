use rand::thread_rng;
use rand::Rng;
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn main() {
    println!("Loading dataset...");
    let mut file = String::new();
    File::open(Path::new("train.csv"))
        .expect("File open failed")
        .read_to_string(&mut file)
        .expect("Read to string failed");
    let mut data = file.lines().skip(1); // Skip column labels

    struct Image {
        label: u8,
        values: Vec<f32>,
    }
    impl Image {
        pub fn new(csv: &str) -> Self {
            let mut data = csv.split_terminator(',');

            let label = data
                .next()
                .unwrap()
                .parse::<u8>()
                .expect("Parsing label failed");
            let mut values: Vec<f32> = Vec::new();

            for value in data {
                values.push(value.parse::<f32>().expect("Parsing values failed") / 255.0);
            }

            Image { label, values }
        }
    }

    struct Layer {
        inputs: u16,
        outputs: u16,
        weights: Vec<Vec<f32>>,
        biases: Vec<f32>,
        activation: fn(f32) -> f32,
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

    println!("Initializing neural network...");

    fn predict(input: Vec<f32>, layers: Vec<Layer>) -> Vec<f32> {
        layers.iter().reduce(|acc, layer| {
            if acc.outputs != layer.inputs {
                panic!("Layer shapes do not match") // Sanity check
            }
            layer
        });

        let mut prev_output: Vec<f32> = input;
        for layer in layers {
            let mut temp: Vec<f32> = Vec::new();
            for (weights, bias) in layer.weights.iter().zip(layer.biases) {
                let value = prev_output
                    .iter()
                    .zip(weights)
                    .map(|(value, weight)| value * weight)
                    .reduce(|sum, value| value + sum)
                    .unwrap()
                    + bias;
                temp.push((layer.activation)(value));
            }
            prev_output.clear();
            prev_output = temp;
        }

        prev_output
    }

    let mut layers: Vec<Layer> = Vec::new();

    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }
    layers.push(Layer::new(784, 10, relu));

    fn softmax(x: Vec<f32>, x_n: f32) -> f32 {
        x_n.exp() / x.iter().map(|x| x.exp()).sum::<f32>()
    }
    layers.push(Layer::new(10, 10, |x| x));

    let img = Image::new(data.next().unwrap());
    let predict_raw = predict(img.values, layers);
    let probabilities: Vec<f32> = predict_raw
        .iter()
        .map(|xn| softmax(predict_raw.clone(), *xn))
        .collect();

    let mut prediction = (0, 0.0);
    for (num, prob) in probabilities.iter().enumerate() {
        if prob >= &prediction.1 {
            prediction = (num, *prob)
        }
    }
    println!(
        "Prediction: {0}, Confidence: {1}%",
        prediction.0 + 1,
        prediction.1 * 100.0
    );
    println!("Actual: {0}", img.label);
}
