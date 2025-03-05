use ndarray::{Array2, Array1, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::io::{BufReader, BufRead};
use serde_json;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct SimpleNN {
    pub weights1: Array2<f32>,
    pub weights2: Array2<f32>,
    pub biases1: Array1<f32>,
    pub biases2: Array1<f32>,
}

impl SimpleNN {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let fan_in = input_size;
        let stddev = (2.0 / fan_in as f32).sqrt();
        let weights1: Array2<f32> = Array2::random((fan_in, hidden_size), Normal::new(0.0, stddev).unwrap());
        let weights2: Array2<f32> = Array2::random((hidden_size, output_size), Normal::new(0.0, stddev).unwrap());
        let biases1: Array1<f32> = Array1::zeros(hidden_size);
        let biases2: Array1<f32> = Array1::zeros(output_size);

        SimpleNN { weights1, weights2, biases1, biases2 }
    }

    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let std = input.std(0.0).max(1e-8);
        let mean = input.mean().unwrap_or(0.0);

        let input = (input - mean) / std;
        let mut hidden = input.dot(&self.weights1) + &self.biases1;

        hidden.mapv_inplace(|x| x.clamp(-10.0, 10.0));
        hidden.mapv_inplace(|x| if x > 0.0 { x } else { 0.01 * x }); // Leaky ReLU

        let output = hidden.dot(&self.weights2) + &self.biases2;
        output
    }

    pub fn train(&mut self, inputs: &Array2<f32>, targets: &Vec<Array1<f32>>, epochs: usize, learning_rate: f32) {
        for epoch in 0..epochs {
            println!("Starting Epoch {}/{}", epoch + 1, epochs); 
            let mut total_loss = 0.0;
            
            for (i, target) in targets.iter().enumerate() {
                let input = inputs.row(i); // Pobierz cały wiersz jako `Array1<f32>`
    
                // Normalizacja wejścia
                let std = input.std(1.0).max(1e-8); // Standardowe odchylenie
                let mean = input.mean().unwrap(); // Średnia
    
                let input_norm = (&input - mean) / std;
    
                // Forward pass
                let mut hidden = input_norm.dot(&self.weights1) + &self.biases1;
                hidden.mapv_inplace(|x| x.clamp(-10.0, 10.0));
                hidden.mapv_inplace(|x| if x > 0.0 { x } else { 0.01 * x }); // Leaky ReLU
                
                let output = hidden.dot(&self.weights2) + &self.biases2;
    
                // Compute loss (Mean Squared Error)
                let loss = (&output - target).mapv(|x| x.powi(2)).sum();
                total_loss += loss;
    
                // Backward pass
                let output_error = &output - target;
                let output_error_clone = output_error.clone();
                let hidden_error = output_error.dot(&self.weights2.t());
                
    
                let hidden_grad = hidden.mapv(|x| if x > 0.0 { 1.0 } else { 0.01 }); // Derivative of Leaky ReLU
                let hidden_delta = &hidden_error * &hidden_grad;
                let hidden_delta_clone = hidden_delta.clone();
                // Update weights and biases
                self.weights2 -= &(hidden.insert_axis(Axis(1)).dot(&output_error_clone.insert_axis(Axis(0))) * learning_rate);
                self.biases2 -= &(output_error * learning_rate);
                
                self.weights1 -= &(input_norm.insert_axis(Axis(1)).dot(&hidden_delta_clone.insert_axis(Axis(0))) * learning_rate);
                self.biases1 -= &(hidden_delta * learning_rate);

                println!("Epoch: {}, Sample {}/{}, Loss {:.4}", epoch + 1, i + 1, inputs.nrows(), loss);
            }

            println!("Epoch: {}, Average Loss: {:.4}", epoch + 1, total_loss / inputs.nrows() as f32);
        }
    }
}



