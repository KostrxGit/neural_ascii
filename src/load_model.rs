use serde::{Deserialize};
use serde_json;
use std::fs::File;
use std::io::BufReader;
use ndarray::{Array2, Array1};
use crate::model::SimpleNN;

#[derive(Deserialize)]
struct Model {
    weights1: Vec<Vec<f32>>,
    weights2: Vec<Vec<f32>>,
    biases1: Vec<f32>,
    biases2: Vec<f32>,
}

pub fn load_model(filename: &str) -> Result<SimpleNN, Box<dyn std::error::Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let model: Model = serde_json::from_reader(reader)?;

    // Convert Vec<Vec<f32>> to Array2<f32>
    let weights1 = Array2::from_shape_vec(
        (model.weights1.len(), model.weights1[0].len()), 
        model.weights1.into_iter().flatten().collect(),
    )?;
    
    let weights2 = Array2::from_shape_vec(
        (model.weights2.len(), model.weights2[0].len()), 
        model.weights2.into_iter().flatten().collect(),
    )?;

    // Convert Vec<f32> to Array1<f32>
    let biases1 = Array1::from_vec(model.biases1);
    let biases2 = Array1::from_vec(model.biases2);

    Ok(SimpleNN {
        weights1,
        weights2,
        biases1,
        biases2,
    })
}
