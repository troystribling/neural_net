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

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
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
        delta = errors * (outputs * (1.0 - outputs))
        return self.learning_rate * numpy.dot(delta, inputs.T)

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_out = self.hidden_output(inputs)
        return self.result_output(hidden_out)

    def reverse_query(self, targets_list):
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_out = self.reverse_hidden_output(targets)
        return self.reverse_result_output(hidden_out)

    def hidden_output(self, inputs):
        layer_in = numpy.dot(self.hidden_weights, inputs)
        return self.activation_function(layer_in)

    def result_output(self, inputs):
        layer_in = numpy.dot(self.result_weights, inputs)
        return self.activation_function(layer_in)

    def reverse_hidden_output(self, outputs):
        inputs = self.inverse_activation_function(outputs)
        return self.scale_outputs(numpy.dot(self.result_weights.T, inputs))

    def reverse_result_output(self, outputs):
        inputs = self.inverse_activation_function(outputs)
        return self.scale_outputs(numpy.dot(self.hidden_weights.T, inputs))

    def scale_outputs(self, outputs):
        outputs -= numpy.min(outputs)
        outputs /= numpy.max(outputs)
        outputs *= 0.98
        outputs += 0.01
        return outputs


    def activation_function(self, x):
        return scipy.special.expit(x)

    def inverse_activation_function(self, x):
        return scipy.special.logit(x)
