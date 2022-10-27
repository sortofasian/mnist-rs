use std::fs::File;
use std::io::Read;

fn main() {
    let data = File::open("../train.csv").expect("Failed to open train file");
    for byte in data.bytes() {
        match byte {
            Ok(byte) => {
                println!("{:?}", byte)
            }
            Err(e) => println!("Failed to read byte: {:?}", e),
        }
    }
}
