import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e, axis=0, keepdims=True)

def feed_forward(input_layer, weights, biases):
    z1 = np.dot(weights[0], input_layer) + biases[0]
    a1 = relu(z1)

    z2 = np.dot(weights[1], a1) + biases[1]
    a2 = relu(z2)

    z3 = np.dot(weights[2], a2) + biases[2]
    a3 = softmax(z3)

    return z1, a1, z2, a2, z3, a3

def feed_backward(x, y, z1, z2, A1, A2, A3, w1, w2):
    # error of the output layer (third layer)
    dz3 = A3 - y
    # backprop into the second hidden layer
    dA2 = np.dot(np.transpose(w2), dz3)
    dz2 = dA2 * (z2 > 0)
    # backprop into the third hidden layer
    dA1 = np.dot(np.transpose(w1), dz2)
    dz1 = dA1 * (z1 > 0)
    # calculate the gradients for the weights (first gradient for weights connecting layer 2 and 3, and biases for layer 3)
    dw2 = np.dot(dz3, np.transpose(A2))
    db3 = dz3
    # gradient for weights connecting layer 1 and 2, and biases for layer 2
    dw1 = np.dot(dz2, np.transpose(A1))
    db2 = dz2
    # gradient for weights connecting layer 0 and 1, and biases for layer 1
    dw0 = np.dot(dz1, np.transpose(x))
    db1 = dz1

    return dw0, db1, dw1, db2, dw2, db3
