mod data_loader;
mod model;
mod ascii_renderer;
mod cli;
mod save_model;
mod load_model;
use ndarray::{Array2, Axis, Array1};
use data_loader::load_mnist;
use model::{SimpleNN};
use save_model::{save_model};


fn normalize(inputs: &Array2<f32>) -> Array2<f32> {
        let mean = inputs.mean_axis(Axis(0)).unwrap();
        let std = inputs.std_axis(Axis(0), 1.0);
        (inputs - &mean) / &std
    }
    
    fn limit_values(inputs: &Array2<f32>, threshold: f32) -> Array2<f32> {
        inputs.map(|x| x.min(threshold).max(-threshold))
    }

fn main() {
    // Inicjalizacja modelu i trening
    let mode = "cli"; // change to train to train model

    if mode == "cli" 
    {
        cli::run();
        return;
    }
    else if mode == "train" 
    {
        
        let (train_images, train_labels, _test_images, _test_labels) = load_mnist();


        assert!(!train_images.iter().any(|&x| x.is_nan()), "NaN detected in train_images");
        assert!(!train_labels.iter().any(|&x| x.is_nan()), "NaN detected in train_labels");

        let limited_train_images = limit_values(&train_images, 1.0);

        let train_labels_vec: Vec<Array1<f32>> = train_labels
        .axis_iter(Axis(0))
        .map(|row| row.to_owned())
        .collect();



        let mut model = SimpleNN::new(28 * 28, 128, 10);
        model.train(&limited_train_images, &train_labels_vec, 20, 0.01); 

        // Zapisz model po treningu
        save_model(&model.weights1, &model.weights2, &model.biases1, &model.biases2, "model_weights.json").expect("Failed to save model");
    }
}
