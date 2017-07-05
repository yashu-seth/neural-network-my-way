import numpy as np

class FeedForward():
    """
    Feed Forward Neural Network.
    """
    def __init__(self, no_of_hidden_layers, no_of_hidden_units_per_layer):
        """
        Parameters
        ----------

        no_of_hidden_layers: Int
        The total number of hidden layers in the network.

        no_of_hidden_units_per_layer: List
        The number of neurons in each layer.

        Example
        -------

        >>> model = FeedForward(2, [3, 5])
        """
        self.no_of_hidden_layers = no_of_hidden_layers

        if no_of_hidden_units_per_layer == None:
            self.no_of_hidden_units_per_layer = []
        else:
            self.no_of_hidden_units_per_layer = no_of_hidden_units_per_layer

        self.layer_activations = None
        self.weights = None
        self.biases = None

    def initalize(self, data, labels, no_of_output_units):

        self.X = data
        self.no_of_samples = self.X.shape[0]
        self.no_of_input_units = self.X.shape[1]
        self.t = labels.T
        self.output = np.zeros(shape=(no_of_output_units, self.t.shape[1]))
        self.no_of_output_units = no_of_output_units

        if self.no_of_samples != self.t.shape[1] or self.t.shape[0] != no_of_output_units:
            raise ValueError

        self.random_initialize()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def deriv_sigmoid(self, x):
        return x * (1 - x)

    def random_initialize(self):

        # w_i dimension = (no of units in layer i + 1) X (no of units in layer i)
        # h_i dimension = (no of units in layer i) X (no of input samples)

        # weight matrix between input layer and first hidden layer
        self.weights = [np.random.uniform(size=(self.no_of_hidden_units_per_layer[0], self.no_of_input_units))]
        #weight matrices between hidden layers
        self.weights.extend([np.random.uniform(size=(self.no_of_hidden_units_per_layer[i],
                                                     self.no_of_hidden_units_per_layer[i-1]))
                            for i in range(1, self.no_of_hidden_layers)])
        #weight matrix between output layer and last hidden layer
        self.weights.append(np.random.uniform(size=(self.no_of_output_units,
                                                    self.no_of_hidden_units_per_layer[self.no_of_hidden_layers - 1],
                                                   )))

        self.biases = [np.random.uniform(size=(self.no_of_hidden_units_per_layer[i], 1))
                       for i in range(self.no_of_hidden_layers)]
        self.biases.append(np.random.uniform(size=(self.no_of_output_units, 1)))

    def _forward_prop(self):

        # The self.layer_activations stroes the output of each layer.
        # The first term is the input, the last term is the final output and the
        # terms in between are the hidden layer activations.
        self.layer_activations = [self.X.T]
        current_input = self.X.T

        for i in range(self.no_of_hidden_layers + 1):
            current_input = self.sigmoid(np.dot(self.weights[i], current_input) + self.biases[i])
            self.layer_activations.append(current_input)

        self.output = current_input

    def _backward_prop(self, lr):
        prev_activation_grad = self.t - self.output 

        deriv_weights = [0,] * (self.no_of_hidden_layers + 1)
        deriv_biases = [0,] * (self.no_of_hidden_layers + 1)
        

        # dE/dW_n = (y-t)y(1-y)h_n
        deriv_weights[-1] = np.dot(prev_activation_grad * self.deriv_sigmoid(self.output),
                                   self.layer_activations[-2].T)

        # dE/db_n = (y-t)y(1-y)
        deriv_biases[-1] = np.dot(prev_activation_grad *\
                                  self.deriv_sigmoid(self.output), np.ones(shape=(self.no_of_samples, 1)))


        for j in range(self.no_of_hidden_layers):
            i = self.no_of_hidden_layers - j - 1

            prev_deriv_hid_layer = self.deriv_sigmoid(self.layer_activations[i+1])
            
            prev_activation_grad = np.dot(self.weights[i+1].T,
                                          prev_activation_grad *\
                                          self.deriv_sigmoid(self.layer_activations[i+2]))

            # dE/dw_i = dE/dh_i+1 * (h_i+1)' * h_i
            deriv_weights[i] = np.dot(prev_activation_grad * prev_deriv_hid_layer,
                                      self.layer_activations[i].T)

            deriv_biases[i] = np.dot(prev_activation_grad *\
                                     prev_deriv_hid_layer, np.ones(shape=(self.no_of_samples, 1)))

        # updation
        for i in range(self.no_of_hidden_layers + 1):
            self.weights[i] += lr * deriv_weights[i]
            self.biases[i] += lr * deriv_biases[i]

    def fit(self, data, labels, no_of_output_units, epoch, lr):

        self.initalize(data, labels, no_of_output_units)

        for i in range(epoch):
            self._forward_prop()
            self._backward_prop(lr)

    def predict(self, data):
        self.X = data
        self._forward_prop()
        return self.output
