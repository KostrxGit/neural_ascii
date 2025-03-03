use std::fs::File;
use std::io::BufReader;

pub fn load_model(filename: &str) -> std::io::Result<(Array2<f32>, Array2<f32>, Array1<f32>, Array1<f32>)> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let model_data: ModelData = serde_json::from_reader(reader)?;

    let weights1 = Array2::from_shape_vec(
        (model_data.weights1.len(), model_data.weights1[0].len()), 
        model_data.weights1.into_iter().flatten().collect()
    ).unwrap();
    
    let weights2 = Array2::from_shape_vec(
        (model_data.weights2.len(), model_data.weights2[0].len()), 
        model_data.weights2.into_iter().flatten().collect()
    ).unwrap();
    
    let biases1 = Array1::from_vec(model_data.biases1);
    let biases2 = Array1::from_vec(model_data.biases2);

    Ok((weights1, weights2, biases1, biases2))
}
