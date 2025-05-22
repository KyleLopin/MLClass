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
        self.number_inputs = number_inputs
        self.model = nn.Linear(number_inputs, number_outputs)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, input_data: list, output_target: list, epochs: int = 1000):
        """
        Fit the neuron to the input-output data.
        Automatically reshapes 1D input/output lists to 2D tensors.
        """
        if isinstance(input_data[0], (int, float)) and self.number_inputs == 1:
            input_data = [[val] for val in input_data]
        if isinstance(output_target[0], (int, float)) and self.model.out_features == 1:
            output_target = [[val] for val in output_target]

        x = torch.tensor(input_data, dtype=torch.float32)
        y = torch.tensor(output_target, dtype=torch.float32)

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
        Accepts:
        - A single input sample as a flat list if number_inputs > 1
        - A list of numbers if number_inputs == 1 (will be reshaped)
        - A list of samples (list of lists)
        Returns:
        - A list of outputs if number_inputs == 1
        - A list of list outputs otherwise
        """
        is_single_input = isinstance(x[0], (int, float))

        if is_single_input:
            if self.number_inputs == 1:
                x = [[val] for val in x]  # reshape
            else:
                x = [x]  # one sample with multiple inputs

        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(x_tensor).numpy()

        if self.number_inputs == 1 and is_single_input:
            return output.flatten().tolist()
        return output.tolist()


if __name__ == '__main__':
    # lesson 1
    neuron = SimpleNeuron(number_inputs=1, number_outputs=1)
    neuron.set_weights([1])
    neuron.set_bias([1])
    output = neuron.get_output([0, 1, 2, 3])
    print(output)
    neuron.properties()
    # lesson 2:
    neuron.train(input_data=[0, 1, 2, 3], output_target=[2, 4, 6, 8])
    neuron.properties()
