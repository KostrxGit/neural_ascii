mod data_loader;
mod model;
mod ascii_renderer;
mod cli;

use data_loader::load_mnist;
use model::{SimpleNN, save_model};

fn main() {
    // Inicjalizacja modelu i trening
    let (train_images, train_labels, test_images, _test_labels) = load_mnist();


    assert!(!train_images.iter().any(|&x| x.is_nan()), "NaN detected in train_images");
    assert!(!train_labels.iter().any(|&x| x.is_nan()), "NaN detected in train_labels");

    let mut model = SimpleNN::new(28 * 28, 128, 10);
    model.train(&train_images, &train_labels, 10, 0.01);

    // Zapisz model po treningu
    save_model(&model.weights1, &model.weights2, &model.biases1, &model.biases2, "model_weights.txt").expect("Failed to save model");
}
