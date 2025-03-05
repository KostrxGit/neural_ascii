use clap::{Arg, Command};
use ndarray::Array1;
// use crate::data_loader::load_mnist;
use crate::model::SimpleNN;
use crate::load_model::load_model;

fn ascii_to_array(ascii: &str) -> Array1<f32> {
    let mut input = vec![0.0; 28 * 28]; // Inicjalizacja pustej tablicy
    let lines: Vec<&str> = ascii.lines().collect();

    for (y, line) in lines.iter().enumerate() {
        for (x, c) in line.chars().enumerate() {
            if c == '#' {
                let index = y * 28 + x;
                if index < 28 * 28 {
                    input[index] = 1.0; // Zaznacz "pomalowane" piksele
                }
            }
        }
    }

    Array1::from_vec(input)
}


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

    let number: u8 = matches
        .get_one::<String>("NUMBER")
        .expect("Liczba powinna być w przedziale 0-9")
        .parse()
        .expect("Nieprawidłowy format liczby");

    // Wczytaj model zamiast trenować od nowa
    let model = load_model("model_weights.json").expect("Nie udało się wczytać modelu");
    
    

    // Użyj liczby jako danych wejściowych do generowania obrazu
    let input = ascii_to_array(&generate_number_ascii_art(number));
    let prediction = model.forward(&input);
    println!("Raw prediction output: {:?}", prediction);
    // Diagnostyka kształtów w `cli.rs`
    println!("Input shape: {:?}", input.shape());
    println!("Prediction shape: {:?}", prediction.shape());

    // Znajdź indeks (liczbę) z największym prawdopodobieństwem
    let (predicted_number, _probability) = prediction
        .indexed_iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    // println!("Predicted number: {}", predicted_number);

    // println!("Input ASCII Art:\n{}", generate_number_ascii_art(number));
    // println!("Converted Input Array: {:?}", input);
    

    // Wygeneruj obraz ASCII art dla przewidywanej liczby
    let ascii_art = generate_number_ascii_art(predicted_number as u8);
    println!("You entered number:\n{}", generate_number_ascii_art(number));
    println!("Model predicted number {}:\n{}", predicted_number, generate_number_ascii_art(predicted_number as u8));

    //println!("{}", ascii_art);
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
