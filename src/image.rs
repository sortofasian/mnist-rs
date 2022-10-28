#[derive(Clone)]
pub struct Image {
    pub label: u8,
    pub values: Vec<f32>,
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
