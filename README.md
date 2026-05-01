# MNIST Digit Classifier — Neural Network from Scratch

A feedforward neural network built from scratch using NumPy to recognize handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). 
No deep learning frameworks, just math.

---

## Architecture

```
Input Layer (784)  →  Hidden Layer 1 (16)  →  Hidden Layer 2 (16)  →  Output Layer (10)
                             ReLU                    ReLU                   Softmax
```

- **Input**: 28×28 grayscale images flattened to 784-dimensional vectors
- **Hidden layers**: ReLU activation
- **Output**: Softmax over 10 classes (digits 0–9)
- **Optimizer**: Stochastic Gradient Descent (SGD), learning rate `0.01`

---

## Project Structure

```
.
├── main.py            # CLI entrypoint — training, testing, and visualization
├── model_struct.py    # Neural network math: forward pass and backpropagation
├── network.npz        # Saved model weights and biases
├── requirements.txt   # Python dependencies
├── train_images/      # MNIST training images (idx3-ubyte)
├── train_labels/      # MNIST training labels (idx1-ubyte)
├── test_images/       # MNIST test images (idx3-ubyte)
└── test_labels/       # MNIST test labels (idx1-ubyte)
```

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download MNIST Data

Download the four binary files from the [MNIST website](http://yann.lecun.com/exdb/mnist/) and place them in the appropriate directories:

| File | Directory |
|------|-----------|
| `train-images.idx3-ubyte` | `train_images/` |
| `train-labels.idx1-ubyte` | `train_labels/` |
| `t10k-images.idx3-ubyte` | `test_images/` |
| `t10k-labels.idx1-ubyte` | `test_labels/` |

### Initialize weights

Run this once before training to generate a fresh `network.npz`:

```python
from main import initialize_parameters
initialize_parameters()
```

---

## Usage

The project is driven from the command line via `main.py`.

### Train

```bash
python main.py --config train
```

Trains for 50 epochs over the full 60,000-image training set. Weights are saved to `network.npz` on completion.

### Test

```bash
python main.py --config test
```

Runs inference over the 10,000-image test set, prints each guess, and displays the image.  
Add '-p' after the test config in the command line to use matplotlib rendering instead of ASCII art.

---

## How It Works

**Forward pass** (`model_struct.py`): Each layer computes `z = W·a + b`, then applies its activation — ReLU for hidden layers, Softmax for the output.

**Backward pass** (`model_struct.py`): Gradients flow back analytically. The output error `dz3 = A3 - y` propagates through the network via the chain rule, with ReLU's derivative being a simple step function (`z > 0`).

**Weight update** (`main.py`): `W -= lr * dW`, applied after every sample (online SGD).

---

## Configuration

Hyperparameters are set at the top of `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layer_sizes` | `[784, 16, 16, 10]` | Network architecture |
| `lr` | `0.01` | Learning rate |
| `PARAMETERS_PATH` | `network.npz` | Path to saved weights |
