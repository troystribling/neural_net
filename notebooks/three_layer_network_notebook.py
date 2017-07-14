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

large_data_path = os.path.join(os.getcwd(), 'large_data')
data_path = os.path.join(os.getcwd(), 'data')

# %%
# Basic operations fro 3 layer network

weights_hidden = numpy.random.normal(0, pow(2, -0.5), (2, 2))
weights_out = numpy.random.normal(0, 1.0, (2, 2))
inputs = numpy.array([2.0, 1.0])

weights_out
hidden_in = numpy.dot(weights_hidden, inputs)
hidden_out = scipy.special.expit(hidden_in)

result_in = numpy.dot(weights_out, hidden_out)
result_out = scipy.special.expit(result_in)

# %%
# Checkout MNIST hand writting
(train_numbers, train_data) = import_data.read_mnist_file(os.path.join(data_path, 'mnist_10.csv'))
seaborn.heatmap(train_data[8], cmap='Greys')

# %%
# Train 3 layer neywork
net = ThreeLayerNetwork(2, 2, 2)
