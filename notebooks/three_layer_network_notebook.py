# %%
%reload_ext autoreload
%autoreload 2

%aimport tempfile
%aimport os
%aimport numpy
%aimport scipy.special

from neural_net.three_layer_network import ThreeLayerNetwork

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
# Train 3 layer neywork
net = ThreeLayerNetwork(2, 2, 2)
