use clap::{Arg, Command};
use ndarray::Array1;
use crate::data_loader::load_mnist;
use crate::model::SimpleNN;
use crate::model::load_model;

pub fn run() {
    let matches = Command::new("Neural ASCII")
        .version("0.1")
        .author("Twoje Imię")
        .about("Generuje ASCII art liczb na podstawie sieci neuronowej")
        .arg(
            Arg::new("NUMBER")
                .help("Liczba do wygenerowania ASCII art (0-9)")
                .required(true)
                .index(1),
        )
        .get_matches();

    let _number: u8 = matches
        .get_one::<String>("NUMBER")
        .expect("Liczba powinna być w przedziale 0-9")
        .parse()
        .expect("Nieprawidłowy format liczby");

    // Wczytaj model zamiast trenować od nowa
    let (weights1, weights2, biases1, biases2) = load_model("model_weights.txt").expect("Nie udało się wczytać modelu");
    let model = SimpleNN {
        weights1,
        weights2,
        biases1,
        biases2,
    };

    // Użyj liczby jako danych wejściowych do generowania obrazu
    let input = Array1::zeros(28 * 28); // Możesz tutaj wprowadzić obraz odpowiadający liczbie
    let prediction = model.forward(&input);

    // Diagnostyka kształtów w `cli.rs`
    println!("Input shape: {:?}", input.shape());
    println!("Prediction shape: {:?}", prediction.shape());

    // Znajdź indeks (liczbę) z największym prawdopodobieństwem
    let (predicted_number, _probability) = prediction
        .indexed_iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("Predicted number: {}", predicted_number);

    // Wygeneruj obraz ASCII art dla przewidywanej liczby
    let ascii_art = generate_number_ascii_art(predicted_number as u8);
    println!("{}", ascii_art);
}

// Funkcja generująca ASCII art dla danej liczby
fn generate_number_ascii_art(number: u8) -> String {
    match number {
        0 => "#####\n#   #\n#   #\n#   #\n#####\n".to_string(),
        1 => "  #  \n ##  \n  #  \n  #  \n#####\n".to_string(),
        2 => "#####\n    #\n#####\n#    \n#####\n".to_string(),
        3 => "#####\n    #\n#####\n    #\n#####\n".to_string(),
        4 => "#   #\n#   #\n#####\n    #\n    #\n".to_string(),
        5 => "#####\n#    \n#####\n    #\n#####\n".to_string(),
        6 => "#####\n#    \n#####\n#   #\n#####\n".to_string(),
        7 => "#####\n    #\n    #\n    #\n    #\n".to_string(),
        8 => "#####\n#   #\n#####\n#   #\n#####\n".to_string(),
        9 => "#####\n#   #\n#####\n    #\n#####\n".to_string(),
        _ => "Invalid number".to_string(),
    }
}
