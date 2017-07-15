# %%
%matplotlib inline
%reload_ext autoreload
%autoreload 2

%aimport tempfile
%aimport os
%aimport numpy
%aimport scipy.special
%aimport seaborn

from neural_net.three_layer_network import ThreeLayerNetwork
from neural_net import import_data
from neural_net import train_network

large_data_path = os.path.join(os.getcwd(), 'large_data')
data_path = os.path.join(os.getcwd(), 'data')

# %%
# Basic operations fro 3 layer network

weights_hidden = numpy.random.normal(0, pow(2, -0.5), (2, 2))
weights_out = numpy.random.normal(0, 1.0, (2, 2))
inputs = numpy.array([2.0, 1.0])
inputs_transpose = inputs.reshape((2, 1))

inputs * inputs_transpose
inputs_transpose * inputs
numpy.matmul(inputs[:, numpy.newaxis], inputs)
hidden_in = numpy.dot(weights_hidden, inputs)
hidden_out = scipy.special.expit(hidden_in)

result_in = numpy.dot(weights_out, hidden_out)
result_out = scipy.special.expit(result_in)


# %%
# Checkout MNIST hand writting
(numbers, number_images) = import_data.read_mnist_file(os.path.join(data_path, 'mnist_10.csv'))
seaborn.heatmap(number_images[8].reshape((28, 28)), cmap='Greys')

# %%
# Test network
test_input = numpy.array([0.01, 0.99, 0.01])
test_target = numpy.array([0.5, 0.5])
test_network = ThreeLayerNetwork(3, 2, 2, 0.3)
hidden_weights = test_network.hidden_weights
result_weights = test_network.result_weights
hidden_out = test_network.hidden_output(test_input)
result_out = test_network.result_output(hidden_out)
result_errors = test_network.errors_output(test_target, result_out)
hidden_errors = test_network.errors_hidden(result_errors)
delta_result_weights = test_network.delta_weights(result_errors, hidden_out, result_out)
delta_hidden_weights = test_network.delta_weights(hidden_errors, test_input, hidden_out)

# %%
# Train 3 layer neywork on MINST handwritting data
(train_numbers, train_number_images) = import_data.read_mnist_file(os.path.join(large_data_path, 'mnist_train.csv'))
network = ThreeLayerNetwork(784, 10, 100, 0.3)
network.train(train_number_images[0], train_numbers[0])
train_network(network, train_number_images, train_numbers)
