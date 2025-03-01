pub fn generate_ascii_art(matrix: &ndarray::Array2<f32>) {
    for row in matrix.rows() {
        for &pixel in row.iter() {
            if pixel > 0.5 {
                print!("#");
            } else {
                print!(" ");
            }
        }
        println!();
    }
}
