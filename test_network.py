import mnist_loader
from Network import Network
from timeit import timeit


def setup():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(
        "./mnist.pkl.gz"
    )
    return (training_data, validation_data, test_data)


def train(training_data, test_data):
    testNet = Network([784, 30, 10])
    testNet.train(training_data, 30, 10, 3.0, test_data=test_data)


# # Debug: take a look at the first input image to ensure it has loaded correctly
# import numpy as np
# from matplotlib import pyplot as plt
# im = np.reshape(training_data[0][0], (28, 28))
# plt.imshow(im)
# plt.show()

if __name__ == "__main__":
    print(
        timeit(
            "train(training_data, test_data)",
            "from __main__ import setup, train\ntraining_data, validation_data, test_data = setup()",
            number=1,
        )
    )
