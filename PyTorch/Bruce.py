# I worte most of this myself but copied a few things from YouTube's SentDex
# credits for this code: https://www.youtube.com/playlist?list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from keras.datasets import mnist, fashion_mnist
import numpy as np
import math

# grabbing MNIST and Fashion MNIST from keras just for conveinence
(mnist_training_img, mnist_training_answer),(mnist_testing_img, mnist_testing_answer) = mnist.load_data()
(f_mnist_training_img, f_mnist_training_answ), (f_mnist_testing_img, f_mnist_testing_answ) = fashion_mnist.load_data()

# normalize the inputs
mnist_training_img, mnist_testing_img = mnist_training_img/255, mnist_testing_img/255
f_mnist_training_img, f_mnist_testing_img = f_mnist_training_img/255, f_mnist_testing_img/255

# Creating the neural network
class ConvNet(nn.Module):
    def __init__(self, transfer_model=False):
        super(ConvNet, self).__init__()
        
        self.network = nn.ModuleList()
        
        self.network.append(nn.Conv2d(1, 32, kernel_size=3))
        self.network.append(nn.Conv2d(32, 32, kernel_size=3))
        self.network.append(nn.Conv2d(32, 32, kernel_size=3))
        
        self.conv_output_shape = self.get_output_shape()

        self.network.append(nn.Linear(self.conv_output_shape, 10))
        
        self.transfer_model = transfer_model
        
    def get_output_shape(self):
        test_data = torch.randn(28,28).view(-1,1,28,28)
        for layer in self.network:
            test_data = F.relu(layer(test_data))
        return test_data[0].shape[0] * test_data[0].shape[1] * test_data[0].shape[2]
    
    def forward(self, inputs):
        for i in range(len(self.network)-1):
            inputs = F.relu(self.network[i](inputs))
        x = self.network[-1](inputs.view(-1, self.conv_output_shape))
        return F.log_softmax(x, dim=1)
    
    # functions for manipulating the weights and biases of the network
    def get_weights(self):
        weights = [layer.weight.data.clone() for layer in self.network]
        return weights
    
    def get_biases(self):
        biases = [layer.bias.data.clone() for layer in self.network]
        return biases
    
    def set_weights(self, w):
        weights = w.copy()
        for i in range(len(weights)):
            self.network[i].weight.data = weights[i]
        if self.transfer_model:
            for i in range(len(self.network)-1):
                self.network[i].weight.requires_grad = False
    
    def set_biases(self, b):
        biases = b.copy()
        for i in range(len(biases)):
            self.network[i].bias.data = biases[i]
        if self.transfer_model:
            for i in range(len(self.network)-1):
                self.network[i].bias.requires_grad = False


# class to have the fit and evaluate functions in one conveinent place
class Agent:
    def __init__(self, transfer_model=False):
        self.net = ConvNet(transfer_model=transfer_model)
        self.optimizer = self.optimizer = optim.Adam(self.net.parameters(), lr=.001)
    
    def fit(self, training_data, target_output, epochs=1, batch_size=1):
        training_batches = list()
        target_batches = list()
        left_over = len(training_data) % batch_size
        
        for i in range(int(len(training_data) // batch_size)):
            training_batches.append(training_data[i*batch_size:(i+1)*batch_size])
            target_batches.append(target_output[i*batch_size:(i+1)*batch_size])
            
        if left_over != 0:
            training_batches.append(training_data[-left_over:])     
            target_batches.append(target_output[-left_over:])
        
        for epoch in range(epochs):
            for i in range(len(training_batches)):
                self.net.zero_grad()
                batch = torch.tensor(training_batches[i])
                output = self.net.forward(batch.view(-1,1,28,28).float())
                loss = F.nll_loss(output, torch.tensor(target_batches[i]))
                loss.backward()
                self.optimizer.step()
            print('Epoch', epoch+1, 'Loss:', loss)

    def evaluate(self, testing_data, target_output):
        testing_data = torch.tensor(testing_data)
        correct = 0
        total = 0

        with torch.no_grad():
            for index, data in enumerate(testing_data):
                output = self.net.forward(data.view(-1,1,28,28).float())
                if torch.argmax(output) == target_output[index]:
                    correct += 1
                total += 1
        print('Accuracy: ', correct/total)

agent = Agent()

# pre_training and testing the mnist model
agent.fit(mnist_training_img, mnist_training_answer, batch_size=64, epochs=4)
agent.evaluate(mnist_testing_img, mnist_testing_answer)

transfer_agent = Agent(transfer_model=True)

# grabbing the weights and biases from the pre-trained model
transfer_agent.net.set_weights(agent.net.get_weights())
transfer_agent.net.set_biases(agent.net.get_biases())

# training and testing the transfer learning model
transfer_agent.fit(f_mnist_training_img, f_mnist_training_answ, epochs=2, batch_size=64)
transfer_agent.evaluate(f_mnist_testing_img, f_mnist_testing_answ)

# showing that the weights are still the same
print(agent.net.get_weights()[0][0])
print(transfer_agent.net.get_weights()[0][0])