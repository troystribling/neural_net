# %%
%matplotlib inline
%reload_ext autoreload
%autoreload 2

%aimport os
%aimport numpy
%aimport scipy.special
%aimport seaborn

from neural_net.three_layer_network import ThreeLayerNetwork
from neural_net import import_data
from neural_net import train_network

large_data_path = os.path.join(os.getcwd(), 'large_data')
data_path = os.path.join(os.getcwd(), 'data')

def count_success(results, answeres):
    hits = 0
    for i in range(len(results)):
        if numpy.argmax(results[i]) == numpy.argmax(answeres[i]):
            hits += 1
    return hits

# %%
# Basic operations fro 3 layer network
weights_hidden = numpy.random.normal(0, pow(2, -0.5), (2, 2))
weights_out = numpy.random.normal(0, 1.0, (2, 2))
inputs_list = numpy.array([2.0, 3.0])
inputs = numpy.array(inputs_list, ndmin=2).T
inputs.shape
numpy.transpose(inputs).shape

hidden_in = numpy.dot(weights_hidden, inputs)
hidden_out = scipy.special.expit(hidden_in)

result_in = numpy.dot(weights_out, hidden_out)
result_out = scipy.special.expit(result_in)

# %%
# Checkout MNIST hand writting
(numbers, number_images) = import_data.read_mnist_file(os.path.join(data_path, 'mnist_10.csv'))
len(number_images)
seaborn.heatmap(number_images[8].reshape((28, 28)), cmap='Greys')

# %%
# Test network
test_inputs = numpy.array([0.01, 0.99, 0.01], ndmin=2).T
test_targets = numpy.array([0.5, 0.5], ndmin=2).T
test_network = ThreeLayerNetwork(3, 2, 2, 0.3)

hidden_weights = test_network.hidden_weights
result_weights = test_network.result_weights

hidden_out = test_network.hidden_output(test_inputs)
result_out = test_network.result_output(hidden_out)
result_errors = test_network.errors_output(test_targets, result_out)
hidden_errors = test_network.errors_hidden(result_errors)
delta_result_weights = test_network.delta_weights(result_errors, hidden_out, result_out)
delta_hidden_weights = test_network.delta_weights(hidden_errors, test_input, hidden_out)

# %%
# Test with one trial neywork on MINST handwritting data
(train_numbers, train_number_images) = import_data.read_mnist_file(os.path.join(large_data_path, 'mnist_train.csv'))
network = ThreeLayerNetwork(784, 10, 100, 0.3)
seaborn.heatmap(train_number_images[8].reshape((28, 28)), cmap='Greys')
network.query(train_number_images[8])

# %%
# Train over all data
train_network(network, train_number_images, train_numbers)

# %%
# Compare with test data
(test_numbers, test_number_images) = import_data.read_mnist_file(os.path.join(large_data_path, 'mnist_test.csv'))
test_results = [network.query(test_number_image) for test_number_image in test_number_images]
success = count_success(test_results, test_numbers)
100.*success/len(test_results)

# %%
# Compare with untrained random network
untrained_network = ThreeLayerNetwork(784, 10, 100, 0.3)
test_results = [untrained_network.query(test_number_image) for test_number_image in test_number_images]
success = count_success(test_results, test_numbers)
100.*success/len(test_results)

# %%
test_run = 6600
numpy.argmax(test_numbers[test_run])
numpy.argmax(test_results[test_run])
seaborn.heatmap(test_number_images[test_run].reshape((28, 28)), cmap='Greys')
