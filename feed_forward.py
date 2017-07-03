import numpy as np

class FeedForward():
    def __init__(self, data, labels, no_of_output_units, no_of_hidden_layers = 1, no_of_hidden_units = None):

        self.X = data
        self.no_of_samples = self.X.shape[0]
        self.no_of_input_units = self.X.shape[1]
        self.t = labels.T
        self.output = np.zeros(shape=(no_of_output_units, self.t.shape[1]))

        if self.no_of_samples != self.t.shape[1] or self.t.shape[0] != no_of_output_units:
            raise ValueError

        self.no_of_hidden_layers = no_of_hidden_layers
        self.no_of_output_units = no_of_output_units

        if no_of_hidden_units == None:
            self.no_of_hidden_units = [self.no_of_input_units,] * no_of_hidden_layers
        else:
            self.no_of_hidden_units = no_of_hidden_units

        self.hidden_layer_activations = None
        self.weights = None
        self.biases = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def deriv_sigmoid(self, x):
        return x * (1 - x)

    def random_initialize(self):

        # w_i dimension = (no of units in layer i + 1) X (no of units in layer i)
        # h_i dimension = (no of units in layer i) X (no of input samples)

        # weight matrix between input layer and first hidden layer
        self.weights = [np.random.uniform(size=(self.no_of_hidden_units[0], self.no_of_input_units))]
        #weight matrices between hidden layers
        self.weights.extend([np.random.uniform(size=(self.no_of_hidden_units[i], self.no_of_hidden_units[i-1]))
                            for i in range(1, self.no_of_hidden_layers)])
        #weight matrix between output layer and last hidden layer
        self.weights.append(np.random.uniform(size=(self.no_of_output_units,
                                                    self.no_of_hidden_units[self.no_of_hidden_layers - 1],
                                                   )))

        self.biases = [np.random.uniform(size=(self.no_of_hidden_units[i], 1)) for i in range(self.no_of_hidden_layers)]
        self.biases.append(np.random.uniform(size=(self.no_of_output_units, 1)))

    def forward_prop(self):
        if self.weights == None:
            self.random_initialize()
            self.hidden_layer_activations = [self.X]

        current_input = self.X.T

        for i in range(self.no_of_hidden_layers + 1):
            current_input = self.sigmoid(np.dot(self.weights[i], current_input) + self.biases[i])
            self.hidden_layer_activations.append(current_input)

        self.output = current_input

#     def backward_prop(self, lr):
#         prev_activation_grad = self.t - self.output 

#         deriv_weights = [0,] * (self.no_of_hidden_layers + 1)
#         deriv_biases = [0,] * (self.no_of_hidden_layers + 1)
        

#         # dE/dW_n = (y-t)y(1-y)h_n

#         print(prev_activation_grad.shape)
#         print(self.deriv_sigmoid(self.output).shape)
#         print(self.hidden_layer_activations[-2].shape)

#         deriv_weights[-1] = np.dot(prev_activation_grad * self.deriv_sigmoid(self.output),
#                                    self.hidden_layer_activations[-2].T)

#         print(deriv_weights[-1].shape)
#         print(self.weights[-1].shape)

#         # dE/db_n = (y-t)y(1-y)
#         deriv_biases[-1] = prev_activation_grad *\
#                            self.deriv_sigmoid(self.output)


#         for j in range(self.no_of_hidden_layers):
#             i = self.no_of_hidden_layers - j - 1

#             prev_deriv_hid_layer = self.deriv_sigmoid(self.hidden_layer_activations[i+1])

#             # dE/dh_i+1 = dE/dh_i+2 * (h_i+1)' * w_i
#             # print(prev_activation_grad)
#             # print
#             # print
#             # print(self.deriv_sigmoid(self.hidden_layer_activations[i+2]))
#             prev_activation_grad = prev_activation_grad *\
#                                    self.deriv_sigmoid(self.hidden_layer_activations[i+2]) *\
#                                    self.weights[i]

#             # dE/dw_i = dE/dh_i+1 * (h_i+1)' * h_i
#             deriv_weights[i] = prev_activation_grad *\
#                                prev_deriv_hid_layer *\
#                                self.hidden_layer_activations[i]

#             deriv_biases[i] = prev_activation_grad *\
#                               prev_deriv_hid_layer

#         # updation
#         self.weights += lr * deriv_weights
#         self.biases += lr * deriv_biases



# x = np.array([[1,0,1,0], [1,0,1,1], [0,1,0,1], [1,0,1,0], [1,0,1,1], [0,1,0,1]])
# y = np.array([[1, 0], [1, 1], [0, 0], [1, 0], [1, 1], [0, 0]])

# # print(a.weights)

# w1 = np.array([[0.42, 0.88, 0.55],
#                [0.10, 0.73, 0.68],
#                [0.60, 0.18, 0.47],
#                [0.92, 0.11, 0.52]])

# w2 = np.array([[0.30], [0.25], [0.23]])

# b1 = np.array([0.46, 0.72, 0.08])
# b2 = np.array([0.69])

# a = FeedForward(x, y, 2, 2, [5, 3])

# # a.random_initialize()

# # print(a.weights)
# # print(a.biases)

# # a.weights = [w1, w2]
# # a.biases = [b1, b2]
# # a.hidden_layer_activations = [x]
# # print(a.output)

# a.forward_prop()

# # print(a.output)

# a.backward_prop(0.1)
