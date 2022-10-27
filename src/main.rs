use rand::thread_rng;
use rand::Rng;
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn main() {
    println!("Loading dataset...");

    let file = {
        let fp = Path::new("train.csv");
        let mut file = File::open(fp).expect("Failed to open train file");

        let mut buffer = String::new();
        file.read_to_string(&mut buffer)
            .expect("Failed to read file");

        buffer
    };

    let mut dataset: Vec<Image> = Vec::new();
    struct Image {
        label: u8,
        values: [f32; 784],
    }

    println!("Parsing data...");
    let mut current_line = 1;

    for line in file.lines() {
        let mut label: u8 = 0;
        let mut values: [f32; 784] = [0.0; 784];

        let mut skip = false;
        for (i, value) in line.split_terminator(',').enumerate() {
            // Handle column labels; check value for any non-numeric chars
            if !value.chars().all(|c| c.is_numeric()) {
                skip = true;
                break;
            }

            if i == 0 {
                label = value.parse::<u8>().expect("Failed to parse label");
                continue;
            }

            let parsed_val = value.parse::<f32>().expect("Failed to parse value");
            values[i - 1] = parsed_val / 255.0; // Convert 0-255 values into 0-1
        }
        if skip {
            continue;
        }

        dataset.push(Image { label, values });

        print!("\rLine {0}", current_line);
        current_line += 1;
    }

    println!("\rLoaded {:?} images\n", dataset.len());

    println!("Initializing neural network...");

    // Initialize arrays for weights and biases with random values from -1.0 to 1.0;
    let mut rng = thread_rng();
    let mut w1: [[f32; 784]; 10] = [[0.0; 784]; 10];
    for neuron in w1.iter_mut() {
        for weight in neuron.iter_mut() {
            *weight = rng.gen::<f32>() * 2.0 - 1.0;
        }
    }

    let mut b1: [f32; 10] = [0.0; 10];
    for bias in b1.iter_mut() {
        *bias = rng.gen::<f32>() * 2.0 - 1.0;
    }

    let mut w2: [[f32; 10]; 10] = [[0.0; 10]; 10];
    for neuron in w2.iter_mut() {
        for weight in neuron.iter_mut() {
            *weight = rng.gen::<f32>() * 2.0 - 1.0;
        }
    }

    let mut b2: [f32; 10] = [0.0; 10];
    for bias in b2.iter_mut() {
        *bias = rng.gen::<f32>() * 2.0 - 1.0;
    }

    println!("NN initialized\n");

    fn relu(x: f32) -> f32 {
        match x > 0.0 {
            true => x,
            false => 0.0,
        }
    }
    fn softmax(x: [f32; 10], x_n: f32) -> f32 {
        x_n.exp() / x.iter().sum::<f32>().exp()
    }

    fn predict(
        x: [f32; 784],
        w1: [[f32; 784]; 10],
        b1: [f32; 10],
        w2: [[f32; 10]; 10],
        b2: [f32; 10],
    ) -> [f32; 10] {
        // Layer 1: ReLU
        let mut o1: [f32; 10] = [0.0; 10];
        for (neuron, weights) in w1.iter().enumerate() {
            for (i, value) in x.iter().enumerate() {
                o1[neuron] = relu(value * weights[i] + b1[neuron]);
            }
        }

        //Layer 2: Softmax output
        let mut o2: [f32; 10] = [0.0; 10];
        for (neuron, weights) in w2.iter().enumerate() {
            for (i, value) in o1.iter().enumerate() {
                o2[neuron] = softmax(o1, value * weights[i] + b2[neuron]);
            }
        }

        return o2;
    }

    let probabilities = predict(dataset[0].values, w1, b1, w2, b2);
    let mut prediction = (0, 0.0);
    for (num, prob) in probabilities.iter().enumerate() {
        if prob >= &prediction.1 {
            prediction = (num, *prob)
        }
    }
    
    //println!("{:?}", probabilities);
    println!("Prediction: {0}, Confidence: {1}%", prediction.0 + 1, prediction.1 * 100.0);
    println!("Actual: {0}", dataset[0].label);
}
