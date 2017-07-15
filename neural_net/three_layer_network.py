import numpy
import scipy.special

class ThreeLayerNetwork(object):

    def __init__(self, in_nodes, result_nodes, hidden_nodes, learning_rate):
        self.in_nodes = in_nodes
        self.result_nodes = result_nodes
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.hidden_weights = numpy.random.normal(0, pow(hidden_nodes, -0.5), (hidden_nodes, in_nodes))
        self.result_weights = numpy.random.normal(0, pow(result_nodes, -0.5), (result_nodes, hidden_nodes))

    def train(self, inputs, targets):
        hidden_out = self.hidden_output(inputs)
        result_out = self.result_output(hidden_out)
        result_errors = self.errors_output(targets, result_out)
        hidden_errors = self.errors_hidden(result_errors)
        self.result_weights += self.delta_weights(result_errors, hidden_out, result_out)
        self.hidden_weights += self.delta_weights(hidden_errors, inputs, hidden_out)

    def errors_output(self, targets, outputs):
        return targets - outputs

    def errors_hidden(self, result_errors):
        return numpy.dot(self.result_weights.T, result_errors)

    def delta_weights(self, errors, inputs, outputs):
        return self.learning_rate * errors * (outputs * (1.0 - outputs)) * inputs.reshape((len(inputs), 1))

    def query(self, inputs):
        hidden_out = self.layer_out(self.hidden_weights, inputs)
        return self.layer_out(self.result_weights, hidden_out)

    def hidden_output(self, inputs):
        layer_in = numpy.dot(self.hidden_weights, inputs)
        return self.activation_function(layer_in)

    def result_output(self, inputs):
        layer_in = numpy.dot(self.result_weights, inputs)
        return self.activation_function(layer_in)

    def activation_function(self, x):
        return scipy.special.expit(x)
