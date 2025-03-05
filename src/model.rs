use ndarray::{Array2, Array1, Axis};
use ndarray_rand::RandomExt;
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::io::{BufReader, BufRead};
use serde_json;

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

    pub fn train(&mut self, inputs: &Array2<f32>, labels: &Array2<f32>, epochs: usize, learning_rate: f32) {
        let lambda = 0.01; 
        let max_grad_norm = 0.01;
        let grad_clipping_threshold = 0.1;

        for epoch in 0..epochs {
            let mut total_error = 0.0;
            for (step, (input, label)) in inputs.outer_iter().zip(labels.outer_iter()).enumerate() {
                let output = self.forward(&input.to_owned());
                let output_error = &label - &output;
                let output_delta = &output_error * &(output.clone() * (1.0 - &output) + 1e-8);
                
                total_error += output_error.iter().map(|&x| x * x).sum::<f32>();

                let hidden = input.dot(&self.weights1) + &self.biases1;
                let hidden = hidden.mapv(|x| if x > 0.0 { x } else { 0.01 * x });

                let mean_hidden = hidden.mean_axis(Axis(0)).unwrap();
                let std_hidden = hidden.std_axis(Axis(0), 0.0).mapv(|x| (x + 1e-3).max(1e-3));

                let hidden = (hidden - &mean_hidden) / (&std_hidden + 1e-8);
                let hidden_cloned = hidden.clone();

                let hidden_error = output_delta.dot(&self.weights2.t());
                let hidden_delta = hidden_error.mapv(|x| x.clamp(-10.0, 10.0)) * hidden.mapv(|x| if x > 0.0 { 1.0 } else { 0.01 });

                let hidden_expanded = hidden_cloned.insert_axis(Axis(1));
                let output_delta_expanded = output_delta.clone().insert_axis(Axis(0));

                let mut grad_w1 = input.view().insert_axis(Axis(1)).dot(&hidden_delta.view().insert_axis(Axis(0))).into_owned();
                let mut grad_w2 = hidden_expanded.dot(&output_delta_expanded).into_owned();

                grad_w1 = grad_w1.mapv(|x| x.clamp(-grad_clipping_threshold, grad_clipping_threshold));
                grad_w2 = grad_w2.mapv(|x| x.clamp(-grad_clipping_threshold, grad_clipping_threshold));

                let norm_w1 = grad_w1.map(|x| x.powi(2)).sum().sqrt();
                let norm_w2 = grad_w2.map(|x| x.powi(2)).sum().sqrt();

                if norm_w1 > max_grad_norm {
                    let scale = max_grad_norm / norm_w1;
                    grad_w1 *= scale;
                }

                if norm_w2 > max_grad_norm {
                    let scale = max_grad_norm / norm_w2;
                    grad_w2 *= scale;
                }

                println!(
                    "Epoch: {}, Step: {}, Grad W1 Norm: {:.6e}, Grad W2 Norm: {:.6e} , Mean Error per Sample: {:.6}",
                    epoch, step, norm_w1, norm_w2, total_error / inputs.shape()[0] as f32
                );

                

                

                // Check for NaNs before updating
                if !grad_w1.iter().any(|&x| x.is_nan() || x.is_infinite()) &&
                   !grad_w2.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                    
                    self.weights1 -= &grad_w1.mapv(|x| x * learning_rate);
                    self.weights1 *= 1.0 - learning_rate * lambda; // L2 Regularization

                    self.weights2 -= &grad_w2.mapv(|x| x * learning_rate);
                    self.weights2 *= 1.0 - learning_rate * lambda; // L2 Regularization
                }

                println!("Grad Bias1 Mean: {:?}", hidden_delta.sum_axis(Axis(0)).mean());
                println!("Grad Bias2 Mean: {:?}", output_delta.sum_axis(Axis(0)).mean());

                // Update biases
                self.biases1 = &self.biases1 + &(hidden_delta.sum_axis(Axis(0)) * learning_rate);
                self.biases1 *= 1.0 - learning_rate * lambda;
                
                self.biases2 = &self.biases2 + &(output_delta.sum_axis(Axis(0)) * learning_rate);
                self.biases2 *= 1.0 - learning_rate * lambda;

                if step % 100 == 0 {
                    println!(
                        "Epoch: {}, Step: {}, Mean W1: {:.6e}, Mean W2: {:.6e}, Bias1 Mean: {:.6}, Bias2 Mean: {:.6} ",
                        epoch, step,
                        self.weights1.mean().unwrap(),
                        self.weights2.mean().unwrap(), 
                        self.biases1.mean().unwrap(),
                        self.biases2.mean().unwrap()
                    );
                }        
            }
            println!("Epoch: {}, Error: {}", epoch, total_error);
        }
    }
}

pub fn load_model(filename: &str) -> std::io::Result<(Array2<f32>, Array2<f32>, Array1<f32>, Array1<f32>)> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();
    let weights1_shape: Vec<usize> = serde_json::from_str(&lines.next().unwrap()?).unwrap();
    let weights1: Vec<f32> = serde_json::from_str(&lines.next().unwrap()?).unwrap();
    let weights2_shape: Vec<usize> = serde_json::from_str(&lines.next().unwrap()?).unwrap();
    let weights2: Vec<f32> = serde_json::from_str(&lines.next().unwrap()?).unwrap();
    let biases1_shape: Vec<usize> = serde_json::from_str(&lines.next().unwrap()?).unwrap();
    let biases1: Vec<f32> = serde_json::from_str(&lines.next().unwrap()?).unwrap();
    let biases2_shape: Vec<usize> = serde_json::from_str(&lines.next().unwrap()?).unwrap();
    let biases2: Vec<f32> = serde_json::from_str(&lines.next().unwrap()?).unwrap();

    let weights1 = Array2::from_shape_vec((weights1_shape[0], weights1_shape[1]), weights1).unwrap();
    let weights2 = Array2::from_shape_vec((weights2_shape[0], weights2_shape[1]), weights2).unwrap();
    let biases1 = Array1::from_shape_vec(biases1_shape[0], biases1).unwrap();
    let biases2 = Array1::from_shape_vec(biases2_shape[0], biases2).unwrap();

    Ok((weights1, weights2, biases1, biases2))
}

