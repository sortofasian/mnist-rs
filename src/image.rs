#[derive(Clone)]
pub struct Image {
    pub label: u8,
    pub y_values: Vec<f64>,
    pub values: Vec<f64>,
}
impl Image {
    pub fn new(csv: &str) -> Self {
        let mut data = csv.split_terminator(',');

        let label = data
            .next()
            .unwrap()
            .parse::<u8>()
            .expect("Parsing label failed");
        let mut values: Vec<f64> = Vec::new();

        for value in data {
            values.push(value.parse::<f64>().expect("Parsing values failed") / 255.0);
        }

        let mut y_values = [0_f64; 10].to_vec();
        y_values[label as usize] = 1.0;

        Image {
            label,
            y_values,
            values,
        }
    }
}
