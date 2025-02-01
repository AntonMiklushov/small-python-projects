from Network import *
from Drawing import *
import numpy as np
import gzip


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic_number, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows, cols)
        return images


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic_number, num_labels = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

LEARNING = False
SAVE = r'C:\Users\Anton\Desktop\ai_python\state.txt'

def main():
    images = load_mnist_images('t10k-images-idx3-ubyte.gz')
    labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    inputs = images.reshape(images.shape[0], -1).T / 255
    one_hot = np.eye(10)[labels].T
    nn = Network([784, 256, 10])
    nn.load_weights_biases(SAVE)
    if LEARNING:
        for epoch in range(1000):
            predictions = nn.forward_propagation(inputs)
            loss = Network.loss_function(predictions, one_hot)
            gradients = nn.backward_propagation(predictions, one_hot)
            nn.update_parameters(gradients, learning_rate=0.3)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        nn.save_weights_biases(SAVE)
    else:
        win = Drawing(nn)
        while win.get_state():
            win.window_loop()


if __name__ == '__main__':
    main()
