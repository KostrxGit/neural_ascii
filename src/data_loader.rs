use mnist::{Mnist, MnistBuilder};
use ndarray::Array2;

fn file_exists(path: &str) -> bool {
    std::fs::metadata(path).is_ok()
}

pub fn load_mnist() -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let paths = [
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
    ];
    
    for path in &paths {
        if !file_exists(path) {
            panic!("Unable to find path to {}", path);
        }
    }

    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .finalize();

    println!("Train images length: {}", trn_img.len());
    println!("Train labels length: {}", trn_lbl.len());
    println!("Test images length: {}", tst_img.len());
    println!("Test labels length: {}", tst_lbl.len());

    let trn_img_shape = (60_000, 28 * 28);
    let tst_img_shape = (10_000, 28 * 28);
    let trn_lbl_shape = (60_000, 10);
    let tst_lbl_shape = (10_000, 10);

    println!("Train images shape: {:?}", trn_img_shape);
    println!("Train labels shape: {:?}", trn_lbl_shape);
    println!("Test images shape: {:?}", tst_img_shape);
    println!("Test labels shape: {:?}", tst_lbl_shape);

    let trn_img = Array2::from_shape_vec(trn_img_shape, trn_img).expect("Error converting train images").mapv(|x| x as f32 / 255.0);
    let tst_img = Array2::from_shape_vec(tst_img_shape, tst_img).expect("Error converting test images").mapv(|x| x as f32 / 255.0);

    let trn_lbl = to_one_hot(trn_lbl, 10);
    let tst_lbl = to_one_hot(tst_lbl, 10);

    let trn_lbl = Array2::from_shape_vec(trn_lbl_shape, trn_lbl).expect("Error converting train labels");
    let tst_lbl = Array2::from_shape_vec(tst_lbl_shape, tst_lbl).expect("Error converting test labels");

    // Diagnostyka kształtów w `data_loader.rs`
    println!("Loaded train images shape: {:?}", trn_img.shape());
    println!("Loaded train labels shape: {:?}", trn_lbl.shape());
    println!("Loaded test images shape: {:?}", tst_img.shape());
    println!("Loaded test labels shape: {:?}", tst_lbl.shape());

    (trn_img, trn_lbl, tst_img, tst_lbl)
}

fn to_one_hot(labels: Vec<u8>, num_classes: usize) -> Vec<f32> {
    let mut one_hot = vec![0.0; labels.len() * num_classes];
    for (i, &label) in labels.iter().enumerate() {
        one_hot[i * num_classes + label as usize] = 1.0;
    }
    one_hot
}
