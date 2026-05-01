import numpy as np
import argparse
import model_struct as model
import matplotlib.pyplot as plt

layer_sizes = [784, 16, 16, 10]
lr = 0.01 # learning rate

# filepaths for the images and labels
TRAIN_IMAGES = "train_images/train-images.idx3-ubyte"
TRAIN_LABELS = "train_labels/train-labels.idx1-ubyte"
TEST_IMAGES = "test_images/t10k-images.idx3-ubyte"
TEST_LABELS = "test_labels/t10k-labels.idx1-ubyte"

PARAMETERS_PATH = 'network.npz'

# there first layer contains weights that transform a 784d vector
# (one for every activation in input layer)
# into a 16d vector (one dimension for every activation in the first hidden layer)


# w0 is weights between layers 0 and 1 (shape is (16, 784)
# w1 is weights between layers 1 and 2 (shape is (16, 16))
# w2 is weights between layers 2 and 3 (shape is (10, 16))

# b1 is biases for layer 1
# b2 is biases for layer 2
# b3 is biases for layer 3

# <---------- Utility Funcs ---------->
def ideal_output(num: int) -> list:
    output = [0 for _ in range(10)]
    output[num] = 1
    return output

def to_onehot(labels: np.ndarray) -> list:
    output = []
    for label in labels:
        output.append(ideal_output(label))
    return output

def get_guess_and_label(x: list) -> tuple:
    maximum = 0
    index = 0
    for i in range(len(x)):
        if x[i] > maximum:
            maximum = x[i]
            index = i
    return index, x[index]

# <---------- Parse input files ---------->
def parse_images(path: str) -> np.ndarray:
    with open(path, "rb") as file:
        data = file.read()

        magic = int.from_bytes(data[0:4], byteorder="big")
        num_images = int.from_bytes(data[4:8], byteorder="big")
        rows = int.from_bytes(data[8:12], byteorder="big")
        cols = int.from_bytes(data[12:16], byteorder="big")

        if magic != 2051:
            print("Invalid magic number; this is not a idx3 image file.")

        pixels = np.frombuffer(data, dtype=np.uint8, offset=16)
        pixels = pixels.reshape(num_images, rows, cols)

        pixels = pixels.astype(np.float32) / 255.0

        # noinspection PyTypeChecker
        return pixels

def parse_labels(path: str) -> np.ndarray:
    with open(path, "rb") as file:
        data = file.read()

        magic = int.from_bytes(data[0:4], byteorder="big")

        # if necessary to find the number of labels:
        # num_labels = int.from_bytes(data[4:8], byteorder="big")

        if magic != 2049:
            print("Invalid magic number; this is not an idx1 label file.")

        labels = np.frombuffer(data, dtype=np.uint8, offset=8)
        return labels



# <---------- Parameter handling --------->
def load_parameters() -> tuple:
    data = np.load(PARAMETERS_PATH)
    w0 = data["w0"]
    w1 = data["w1"]
    w2 = data["w2"]
    b1 = data["b1"]
    b2 = data["b2"]
    b3 = data["b3"]

    return w0, w1, w2, b1, b2, b3

def initialize_parameters() -> None:
    w0 = np.random.randn(16, 784) * 0.01
    w1 = np.random.randn(16, 16) * 0.01
    w2 = np.random.randn(10, 16) * 0.01
    b1 = np.random.randn(16, 1) * 0.01
    b2 = np.random.randn(16, 1) * 0.01
    b3 = np.random.randn(10, 1) * 0.01

    np.savez(PARAMETERS_PATH, w0=w0, w1=w1, w2=w2, b1=b1, b2=b2, b3=b3)



# <---------- train/test/eval ---------->
def train(pixels: np.ndarray, labels: list, epochs: int = 50) -> None:
    w0, w1, w2, b1, b2, b3 = load_parameters()
    for epoch in range(epochs):

        print(f"epochs: {epoch} / {epochs}")
        for i in range(len(pixels)):
            # get flattened representations of labels and images
            x = pixels[i].reshape(-1, 1)
            y = np.array(labels[i]).reshape(-1, 1)

            # find backprop gradients
            z1, a1, z2, a2, z3, a3 = model.feed_forward(x, [w0, w1, w2], [b1, b2, b3])
            dw0, db1, dw1, db2, dw2, db3 = model.feed_backward(x, y, z1, z2, a1, a2, a3, w1, w2)

            # increment parameters
            w0 -= lr * dw0
            w1 -= lr * dw1
            w2 -= lr * dw2
            b1 -= lr * db1
            b2 -= lr * db2
            b3 -= lr * db3

    np.savez(PARAMETERS_PATH, w0=w0, w1=w1, w2=w2, b1=b1, b2=b2, b3=b3)

def test_images(pixels: np.ndarray, labels: np.ndarray, plot: bool = False) -> None:
    w0, w1, w2, b1, b2, b3 = load_parameters()
    weights = [w0, w1, w2]
    biases = [b1, b2, b3]

    guesses = []
    for i in range(len(pixels)):
        image = pixels[i].reshape(-1,1)
        _, _, _, _, _, output = model.feed_forward(image, weights, biases)
        guess, label = get_guess_and_label(output), labels[i]
        guesses.append((guess, label))
        print(f'Guess: {guess}')
        display_image(image.reshape(28, 28), plot=plot)
        input()
    print(get_accuracy(guesses))

def get_accuracy(guesses: list) -> str:
    correct = 0
    count = 0
    for guess, label in guesses:
        if guess == label:
            correct += 1
        count += 1
    return f"{correct / count * 100}%"



# <---------- Plot funcs ---------->
def pixel_to_char(value: int) -> str:
    chars = " .:-=+*#%@"
    index = int((value / 255) * (len(chars) - 1))
    return chars[index]

def display_image(image: list, plot: bool) -> None:
    for i in range(len(image)):
        image[i] *= 255
    if plot:
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.show()
    else:
        for row in image:
            print("".join(pixel_to_char(pixel) for pixel in row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NIDS - Network Intrusion Detection System')
    parser.add_argument('--config', '-c', type=str, required=True, help='Configuration of model ("train" or "test")')
    parser.add_argument('--plot', '-p', action='store_true', help='For testing, entered when wanting to plot the number instead of rendering it in ASCII')
    args = parser.parse_args()
    config = args.config
    plot = args.plot

    if config == "train":
        img = parse_images(TRAIN_IMAGES)
        lab = parse_labels(TRAIN_LABELS)
        train(img, to_onehot(lab))
    elif config == "test":
        img = parse_images(TEST_IMAGES)
        lab = parse_labels(TEST_LABELS)
        test_images(img, lab, plot=plot)