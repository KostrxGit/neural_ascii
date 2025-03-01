use ndarray::{Array2, Array1, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use std::fs::File;
use std::io::{Write, BufWriter, BufReader, BufRead};

pub struct SimpleNN {
    pub weights1: Array2<f32>,
    pub weights2: Array2<f32>,
    pub biases1: Array1<f32>,
    pub biases2: Array1<f32>,
}

impl SimpleNN {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let weights1: Array2<f32> = Array2::random((input_size, hidden_size), Normal::new(0.0, 0.1).unwrap());
        let weights2: Array2<f32> = Array2::random((hidden_size, output_size), Normal::new(0.0, 0.1).unwrap());
        let biases1: Array1<f32> = Array1::zeros(hidden_size);
        let biases2: Array1<f32> = Array1::zeros(output_size);

        // Diagnostyka inicjalizacji
        assert!(!weights1.iter().any(|&x| x.is_nan()), "NaN detected in weights1 during initialization");
        assert!(!weights2.iter().any(|&x| x.is_nan()), "NaN detected in weights2 during initialization");
        assert!(!biases1.iter().any(|&x| x.is_nan()), "NaN detected in biases1 during initialization");
        assert!(!biases2.iter().any(|&x| x.is_nan()), "NaN detected in biases2 during initialization");

        SimpleNN { weights1, weights2, biases1, biases2 }
    }

    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let hidden = input.dot(&self.weights1) + &self.biases1;

        // Diagnostyka przed aktywacją
        if hidden.iter().any(|&x| x.is_nan()) {
            println!("NaN detected in hidden layer before activation");
            println!("input: {:?}", input);
            println!("weights1: {:?}", self.weights1);
            println!("biases1: {:?}", self.biases1);
            println!("hidden: {:?}", hidden);
        }

        let hidden = hidden.map(|x| x.max(0.0)); // ReLU activation

        // Diagnostyka po aktywacji
        if hidden.iter().any(|&x| x.is_nan()) {
            println!("NaN detected in hidden layer after activation");
            println!("hidden: {:?}", hidden);
        }

        let output = hidden.dot(&self.weights2) + &self.biases2;

        // Diagnostyka wartości output
        if output.iter().any(|&x| x.is_nan()) {
            println!("NaN detected in output layer");
            println!("hidden: {:?}", hidden);
            println!("weights2: {:?}", self.weights2);
            println!("biases2: {:?}", self.biases2);
            println!("output: {:?}", output);
        }

        output
    }

    pub fn train(&mut self, inputs: &Array2<f32>, labels: &Array2<f32>, epochs: usize, learning_rate: f32) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;
            for (step, (input, label)) in inputs.outer_iter().zip(labels.outer_iter()).enumerate() {
                let output = self.forward(&input.to_owned());

                let output_error = &label - &output;
                let output_delta = output_error.clone();

                if output_error.iter().any(|&x| x.is_nan()) || output_delta.iter().any(|&x| x.is_nan()) {
                    println!("NaN detected in output_error or output_delta at step {}", step);
                    println!("input: {:?}", input);
                    println!("output: {:?}", output);
                    println!("label: {:?}", label);
                    println!("output_error: {:?}", output_error);
                    println!("output_delta: {:?}", output_delta);
                    return;
                }

                total_error += output_error.iter().map(|&x| x * x).sum::<f32>();

                let hidden = input.dot(&self.weights1) + &self.biases1;
                let hidden = hidden.map(|x| x.max(0.0)); // ReLU activation
                let hidden_error = output_delta.dot(&self.weights2.t());
                let hidden_delta = hidden_error * hidden.map(|x| if *x > 0.0 { 1.0 } else { 0.0 });

                let hidden_expanded = hidden.insert_axis(Axis(1));  // Rozszerzenie wymiaru do (128, 1)
                let output_delta_expanded = output_delta.clone().insert_axis(Axis(0));  // Rozszerzenie wymiaru do (1, 10)

                self.weights2 = self.weights2.clone() + hidden_expanded.dot(&output_delta_expanded).into_owned();
                self.weights1 = self.weights1.clone() + input.view().insert_axis(Axis(1)).dot(&hidden_delta.view().insert_axis(Axis(0))).into_owned();
                self.biases2 = self.biases2.clone() + output_delta.sum_axis(Axis(0)).into_owned();
                self.biases1 = self.biases1.clone() + hidden_delta.sum_axis(Axis(0)).into_owned();

                // Wyświetlanie postępu kroków
                if step % 100 == 0 {
                    println!("Epoch: {}, Step: {}, hidden_expanded shape: {:?}, output_delta_expanded shape: {:?}", epoch, step, hidden_expanded.shape(), output_delta_expanded.shape());
                }
            }
            // Wypisanie postępu co epokę i średniego błędu
            println!("Epoch: {}, Average Error: {}", epoch, total_error / inputs.len() as f32);
            println!("Weights1 shape: {:?}", self.weights1.shape());
            println!("Weights2 shape: {:?}", self.weights2.shape());
            println!("Biases1 shape: {:?}", self.biases1.shape());
            println!("Biases2 shape: {:?}", self.biases2.shape());
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

