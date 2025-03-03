use ndarray::{Array2, Array1, Axis};
use ndarray_rand::RandomExt;
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::io::{Write, BufWriter, BufReader, BufRead};
use rand::Rng;
use serde_json;

pub struct SimpleNN {
    pub weights1: Array2<f32>,
    pub weights2: Array2<f32>,
    pub biases1: Array1<f32>,
    pub biases2: Array1<f32>,
}

impl SimpleNN {
    
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let std = (2.0 / hidden_size as f32).sqrt();
        let weights1: Array2<f32> = Array2::random((input_size, hidden_size), Normal::new(0.0, std).unwrap());
        let weights2: Array2<f32> = Array2::random((hidden_size, output_size), Normal::new(0.0, std).unwrap());
        let biases1: Array1<f32> = Array1::zeros(hidden_size);
        let biases2: Array1<f32> = Array1::zeros(output_size);

        SimpleNN { weights1, weights2, biases1, biases2 }
    }

    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {

        let hidden = input.dot(&self.weights1) + &self.biases1;

        // println!("Hidden before activation: {:?}", hidden);

        let hidden = hidden.map(|x| if *x > 0.0 {*x} else {0.01 * *x}); // Leaky ReLU activation

        // println!("Hidden after activation: {:?}", hidden);

        let output = hidden.dot(&self.weights2) + &self.biases2;

        output
    }

    

    pub fn train(&mut self, inputs: &Array2<f32>, labels: &Array2<f32>, epochs: usize, learning_rate: f32) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;
            for (step, (input, label)) in inputs.outer_iter().zip(labels.outer_iter()).enumerate() {
                let output = self.forward(&input.to_owned());

                let output_error = &label - &output;
                let output_delta = output_error.clone();
                //output_delta = output_error * (output * (1.0 - output))

                total_error += output_error.iter().map(|&x| x * x).sum::<f32>();
                
                let hidden = input.dot(&self.weights1) + &self.biases1;

                //println!("Hidden before activation: {:?}", hidden);

                let hidden = hidden.map(|x| x.max(0.0)); // ReLU activation

                //println!("Hidden after activation: {:?}", hidden);

                let hidden_error = output_delta.dot(&self.weights2.t());
                let hidden_delta = hidden_error * hidden.map(|x| if *x > 0.0 { 1.0 } else { 0.01 });

                //println!("Hidden delta: {:?}", hidden_delta);
                // println!("Gradient weights1: {:?}", hidden_delta.view().insert_axis(Axis(0)));

                let hidden_expanded = hidden.insert_axis(Axis(1));  // Rozszerzenie wymiaru do (128, 1)
                let output_delta_expanded = output_delta.clone().insert_axis(Axis(0));  // Rozszerzenie wymiaru do (1, 10)

                 
              

                //Adaptiv scaling
                //let norm = self.weights2.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                // let scale = (grad_clipping_threshold / norm).min(1.0);
                // self.weights2 *= scale;
                // Ograniczanie gradientów (gradient clipping)
                let grad_clipping_threshold = 5.0;
                let lambda = 0.01; // Współczynnik regularizacji
                self.weights2 = self.weights2.clone() + hidden_expanded.dot(&output_delta_expanded).into_owned().map(|x| x.min(grad_clipping_threshold).max(-grad_clipping_threshold)) * learning_rate - lambda * &self.weights2;
                self.weights1 = self.weights1.clone() + input.view().insert_axis(Axis(1)).dot(&hidden_delta.view().insert_axis(Axis(0))).into_owned().map(|x| x.min(grad_clipping_threshold).max(-grad_clipping_threshold)) * learning_rate - lambda * &self.weights1;
                self.biases2 = self.biases2.clone() + output_delta.sum_axis(Axis(0)).into_owned().map(|x| x.min(grad_clipping_threshold).max(-grad_clipping_threshold)) * learning_rate;
                self.biases1 = self.biases1.clone() + hidden_delta.sum_axis(Axis(0)).into_owned().map(|x| x.min(grad_clipping_threshold).max(-grad_clipping_threshold)) * learning_rate;

                println!("After weights1: {:?}", self.weights1);
                println!("After  weights2: {:?}", self.weights2);
                // println!("After  biases1: {:?}", self.biases1);
                // println!("After  biases2: {:?}", self.biases2);
                
          
                

                // Wyświetlanie postępu kroków
                if step % 100 == 0 {
                    println!("Epoch: {}, Step: {}, hidden_expanded shape: {:?}, output_delta_expanded shape: {:?}", epoch, step, hidden_expanded.shape(), output_delta_expanded.shape());
                }
                if self.weights1.iter().any(|x| x.is_nan()) {
                    panic!("NaN detected in weights1!");
                }
                if self.weights2.iter().any(|x| x.is_nan()) {
                    panic!("NaN detected in weights2!");
                }
                if self.biases1.iter().any(|x| x.is_nan()) {
                    panic!("NaN detected in biases1!");
                }
                if self.biases2.iter().any(|x| x.is_nan()) {
                    panic!("NaN detected in biases2!");
                }                
            }
            println!("Epoch: {}, Error: {}", epoch, total_error);
        }
    }
}


pub fn save_model(weights1: &Array2<f32>, weights2: &Array2<f32>, biases1: &Array1<f32>, biases2: &Array1<f32>, filename: &str) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "{:?}", weights1.shape())?;
    writeln!(writer, "{:?}", weights1.as_slice().unwrap())?;
    writeln!(writer, "{:?}", weights2.shape())?;
    writeln!(writer, "{:?}", weights2.as_slice().unwrap())?;
    writeln!(writer, "{:?}", biases1.shape())?;
    writeln!(writer, "{:?}", biases1.as_slice().unwrap())?;
    writeln!(writer, "{:?}", biases2.shape())?;
    writeln!(writer, "{:?}", biases2.as_slice().unwrap())?;

    Ok(())
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

