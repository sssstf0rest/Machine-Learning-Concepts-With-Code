import numpy as np

# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons, name=None):
        # Initialize weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.name = name or f"Dense({n_inputs}->{n_neurons})"

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
    @property
    def in_dim(self):
        return self.weights.shape[0]

    @property
    def out_dim(self):
        return self.weights.shape[1]

    def summary(self):
        params = self.weights.size + self.biases.size
        return {
            "type": "Dense",
            "name": self.name,
            "in": self.in_dim,
            "out": self.out_dim,
            "params": params,
        }

# class Layer_Dense:
#     def __init__(self, n_inputs, n_neurons, name=None):
#         self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
#         self.biases = np.zeros((1, n_neurons))
#         self.name = name or f"Dense({n_inputs}->{n_neurons})"

#     def forward(self, inputs):
#         self.inputs = inputs
#         self.output = np.dot(inputs, self.weights) + self.biases

#     def backward(self, dvalues):
#         self.dweights = np.dot(self.inputs.T, dvalues)
#         self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
#         self.dinputs = np.dot(dvalues, self.weights.T)

#     @property
#     def in_dim(self):
#         return self.weights.shape[0]

#     @property
#     def out_dim(self):
#         return self.weights.shape[1]

#     def summary(self):
#         params = self.weights.size + self.biases.size
#         return {
#             "type": "Dense",
#             "name": self.name,
#             "in": self.in_dim,
#             "out": self.out_dim,
#             "params": params,
#         }