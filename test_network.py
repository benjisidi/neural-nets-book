import mnist_loader
from Network import Network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper(
    "./mnist.pkl.gz"
)

# # Debug: take a look at the first input image to ensure it has loaded correctly
# import numpy as np
# from matplotlib import pyplot as plt
# im = np.reshape(training_data[0][0], (28, 28))
# plt.imshow(im)
# plt.show()

testNet = Network([784, 30, 10])
testNet.train(training_data, 30, 10, 3.0, test_data=test_data)
