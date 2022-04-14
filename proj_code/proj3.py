from torch import nn
import numpy as np
import pdb
import torch


def relu_forward(inputs):
    """

    :param inputs: inputs to the rectified linear layer
    :return: the output after applying the rectified linear activation function
    """
    return np.maximum(0,inputs)
    pass


def relu_backward(grad, inputs):
    """

    :param grad: the backpropagated gradients
    :param inputs: the inputs that were given to the rectified linear layer
    :return: the gradient with respect to the inputs
    """
    new = grad
    for i, grad_list in enumerate(grad,0):
        for j, gradient in enumerate(grad_list, 0):
            if inputs[i][j] > 0:
                new[i][j] = gradient
            else:
                new[i][j] = 0
    return new
    pass


def linear_forward(inputs, weights, biases):
    """

    :param inputs: inputs to the linear layer
    :param weights: the weight parameters
    :param biases: the bias parameters
    :return: the output after applying the linear transformation
    """
    return inputs@weights.T + biases
    pass


def linear_backward(grad, inputs, weights):
    """

    :param grad: the backpropagated gradient
    :param inputs: the inputs that were given to the linear layer
    :param weights: the weight parameters
    :return: the gradient with respect to the weights, biases, and inputs
    """
    print("grad shape: ", grad.shape)
    print("inputs shape: ", inputs.shape)
    print("weights shape: ", weights.shape)

    weight = np.matmul(grad, weights)
    input = np.dot(grad.T, inputs)
    bias = np.sum(grad, axis = 0)

    return input, bias, weight

    pass

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #values chosen are arbitrary besides input and output
        self.fc1 = nn.Linear(81, 40)
        self.fc2 = nn.Linear(40, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_nnet_model() -> nn.Module:
    """ Get the neural network model

    @return: neural network model
    """
    return Net()
    pass


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray):
    """

    :param nnet: The neural network
    :param states_nnet: states (inputs)
    :param outputs: the outputs
    :return: None
    """

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(nnet.parameters(), lr=0.001, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.996)

    #get values to compare results against and tensor of inputs
    inputs_tensor = torch.from_numpy(states_nnet).float()
    target = torch.from_numpy(outputs).float()

    #data loader, breaks into batches of size 100
    data = torch.utils.data.TensorDataset(inputs_tensor, target)
    dataLoader = torch.utils.data.DataLoader(
       data, batch_size=100, shuffle=True, num_workers=4)

    for epoch, toLoad in enumerate(dataLoader, 0):
        nnet.train()

        inputs, labels = toLoad
        pred = nnet(inputs)

        loss = criterion(pred, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        scheduler.step()

    pass

