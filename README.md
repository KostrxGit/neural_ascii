# Neural ASCII

A neural network built from scratch in Rust that recognizes handwritten digits from the MNIST dataset and renders predictions as ASCII art.

## Overview

Neural ASCII implements a simple feedforward neural network with one hidden layer to classify digits (0–9). It trains on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and uses a CLI interface where you provide a digit — the model then predicts it and displays both the input and predicted number as ASCII art.

### Architecture

- **Input layer:** 784 neurons (28×28 pixel images)
- **Hidden layer:** 128 neurons with Leaky ReLU activation
- **Output layer:** 10 neurons (one per digit class)
- **Loss function:** Mean Squared Error
- **Weight initialization:** He initialization

## Project Structure

```
neural_ascii/
├── Cargo.toml              # Dependencies and project metadata
├── model_weights.json      # Saved model weights (generated after training)
├── data/                   # MNIST dataset files
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── src/
│   ├── main.rs             # Entry point — switches between training and CLI mode
│   ├── model.rs            # Neural network definition (forward pass, training loop)
│   ├── cli.rs              # CLI interface for digit prediction
│   ├── data_loader.rs      # MNIST data loading and preprocessing
│   ├── ascii_renderer.rs   # ASCII art rendering utilities
│   ├── save_model.rs       # Serialize model weights to JSON
│   └── load_model.rs       # Deserialize model weights from JSON
└── tests/
    └── test_model.rs       # Test file for the model
```

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (edition 2024)
- MNIST dataset files placed in the `data/` directory

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd neural_ascii
```

### 2. Obtain the MNIST dataset

Place the following files in the `data/` directory:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

These can be downloaded from the [MNIST homepage](http://yann.lecun.com/exdb/mnist/).

### 3. Train the model

In `src/main.rs`, set the mode to `"train"`:

```rust
let mode = "train";
```

Then build and run:

```bash
cargo run
```

Training runs for 20 epochs over the full 60,000-image training set. Once complete, the weights are saved to `model_weights.json`.

### 4. Run predictions via CLI

Set the mode back to `"cli"` (the default) and run with a digit argument:

```bash
cargo run -- <digit>
```

**Example:**

```bash
cargo run -- 7
```

**Output:**

```
You entered number:
#####
    #
    #
    #
    #

Model predicted number 7:
#####
    #
    #
    #
    #
```

## Dependencies

| Crate                  | Purpose                                           |
| ---------------------- | ------------------------------------------------- |
| `ndarray`              | N-dimensional array operations                    |
| `ndarray-rand`         | Random array generation for weight initialization |
| `rand` / `rand_distr`  | Random number generation and distributions        |
| `mnist`                | MNIST dataset loader                              |
| `clap`                 | Command-line argument parsing                     |
| `serde` / `serde_json` | Model weight serialization and deserialization    |

## How It Works

1. **Data loading** — MNIST images are loaded, flattened to 784-element vectors, and normalized to `[0, 1]`. Labels are one-hot encoded.
2. **Training** — The network performs a forward pass, computes MSE loss, and updates weights via backpropagation with gradient descent.
3. **Inference** — In CLI mode, the user provides a digit. An ASCII art template is converted to a 784-element binary array, fed through the network, and the predicted class is displayed as ASCII art.
4. **Persistence** — Trained weights are saved to and loaded from `model_weights.json` using serde.

## License

This project is provided as-is for educational purposes.
