import numpy as np

class FeedForward():
    def __init__(self, data, labels, no_of_output_units = 1, no_of_hidden_layers = 1, no_of_hidden_units = None):

        self.X = data
        self.no_of_samples = self.X.shape[0]
        self.no_of_input_units = self.X.shape[1]
        self.Y = labels

        if self.no_of_samples != self.Y.shape[0] or self.Y.shape[1] != 1:
            raise ValueError

        self.no_of_hidden_layers = no_of_hidden_layers
        self.no_of_output_units = no_of_output_units

        if no_of_hidden_units == None:
            self.no_of_hidden_units = [self.no_of_input_units,] * no_of_hidden_layers
        else:
            self.no_of_hidden_units = no_of_hidden_units

        # weight matrix between input layer and first hidden layer
        self.weights = [np.zeros(shape=(self.no_of_input_units, self.no_of_hidden_units[0]))]
        #weight matrices between hidden layers
        self.weights.extend([np.zeros(shape=(self.no_of_hidden_units[i-1], self.no_of_hidden_units[i]))
                            for i in range(1, self.no_of_hidden_layers)])
        #weight matrix between output layer and last hidden layer
        self.weights.append(np.zeros(shape=(self.no_of_hidden_units[self.no_of_hidden_layers - 1], self.no_of_output_units)))

        self.biases = [np.zeros(self.no_of_hidden_units[i]) for i in range(self.no_of_hidden_layers)]
        self.biases.append([np.zeros(self.no_of_output_units)])

