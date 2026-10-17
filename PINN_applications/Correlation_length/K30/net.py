import torch
import torch.optim
from collections import OrderedDict   #Ordered dictionary
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, seq_net, name='MLP', activation=torch.tanh):
        #Fully connected MLP: seq_net defines the network, with 2 inputs, 1 output, six hidden layers of 20 neurons, and tanh activation
        super().__init__() #Call the nn.Module initializer
        self.features = OrderedDict()
        for i in range(len(seq_net) - 1):
            self.features['{}_{}'.format(name, i)] = nn.Linear(seq_net[i], seq_net[i + 1], bias=True)
            #Format names as MLP_0, MLP_1, etc.; indices 0-6 represent seven connected MLP layers
        self.features = nn.ModuleDict(self.features)  #Assemble the network

        self.active = activation  #Use tanh activation between layers
        # Initialize biases to 0
        for m in self.modules():
            if isinstance(m, nn.Linear): #If the layer is fully connected (Linear)
                nn.init.constant_(m.bias, 0) #Fill each layer's bias values with 0

    def forward(self, x):
        # x = x.view(-1, 2)
        length = len(self.features)
        i = 0
        for name, layer in self.features.items():
            x = layer(x)
            if i == length - 1:
                break
            i += 1  #Six layers in total
            x = self.active(x)
        return x

