use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use serde_json;
use ndarray::{Array1,Array2};


#[derive(Serialize, Deserialize)]
struct ModelData {
    weights1: Vec<Vec<f32>>,
    weights2: Vec<Vec<f32>>,
    biases1: Vec<f32>,
    biases2: Vec<f32>,
}

pub fn save_model(weights1: &Array2<f32>, weights2: &Array2<f32>, biases1: &Array1<f32>, biases2: &Array1<f32>, filename: &str) -> std::io::Result<()> {
    let model_data = ModelData {
        weights1: weights1.outer_iter().map(|row| row.to_vec()).collect(),
        weights2: weights2.outer_iter().map(|row| row.to_vec()).collect(),
        biases1: biases1.to_vec(),
        biases2: biases2.to_vec(),
    };

    let file = File::create(filename)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &model_data)?;

    Ok(())
}
