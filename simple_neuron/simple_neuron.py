# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNeuron:
    def __init__(self, number_inputs: int, number_outputs: int):
        """
        Initialize a simple neuron model with given number of inputs and outputs.
        """
        self.model = nn.Linear(number_inputs, number_outputs)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def fit(self, input_data: list, output_data: list, epochs: int = 1000):
        """
        Fit the neuron to the input-output data.
        """
        x = torch.tensor(input_data, dtype=torch.float32)
        y = torch.tensor(output_data, dtype=torch.float32)

        for _ in range(epochs):
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def properties(self):
        """
        Print the weight and bias values.
        """
        with torch.no_grad():
            print("Weights:", self.model.weight.data.numpy())
            print("Bias:", self.model.bias.data.numpy())

    def set_weights(self, weights: list):
        """
        Manually set the weights.
        """
        with torch.no_grad():
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
            self.model.weight.copy_(weight_tensor)

    def set_bias(self, bias: list):
        """
        Manually set the bias.
        """
        with torch.no_grad():
            bias_tensor = torch.tensor(bias, dtype=torch.float32)
            self.model.bias.copy_(bias_tensor)

    def get_output(self, x: list):
        """
        Predict output for a new input.
        """
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(x_tensor)
        return output.numpy()

