import numpy
import scipy.special

class ThreeLayerNetwork(object):

    def __init__(self, in_nodes, out_nodes, hidden_nodes):
        self.in_nodes = in_nodes
        self.result_nodes = result_nodes
        self.hidden_nodes = hidden_nodes
        self.hidden_weights = numpy.random.normal(0, pow(hidden_nodes, -0.5), (hidden_nodes, in_nodes))
        self.result_weights = numpy.random.normal(0, pow(result_nodes, -0.5), (result_nodes, hidden_nodes))

    def train(self, learning_rate, inputs, targets):
        hidden_out = self.layer_out(self.hidden_weights, inputs)
        result_out = self.layer_out(self.result_weights, hidden_out)
        result_errors = result_out - self.query(inputs)
        hidden_errors = numpy.dot(self.result_weights.T, result_errors)
        self.result_weights += self.delta_weights(learning_rate, result_errors, result_out, hidden_out)
        self.hidden_weights += self.delta_weights(learning_rate, hidden_errors, hidden_errors, inputs)

    def delta_weights(self, learning_rate, errors, inputs, outputs):
        return learning_rate * numpy.dot(errors * (outputs * (1.0 - outputs)), numpy.transpose(inputs))

    def query(self, inputs):
        hidden_out = self.layer_out(self.hidden_weights, inputs)
        return self.layer_out(self.result_weights, hidden_out)

    def layer_out(self, weights, inputs):
        layer_in = numpy.dot(weights, inputs)
        return self.activation_function(layer_in)

    def activation_function(self, x):
        return scipy.special.expit(x)
