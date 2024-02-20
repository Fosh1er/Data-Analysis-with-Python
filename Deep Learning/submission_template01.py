import numpy as np
import torch
from torch import nn


def create_model():
    # Linear layer mapping from 784 features, so it should be 784->256->16->10

    class model(nn.Module):
        def __init__(self, dim=784):
            super(Net, self).__init__()

            self.fc1 = nn.Linear(dim, 256)
            self.tanh1 = nn.ReLU()

            self.fc2 = nn.Linear(256,16)
            self.tanh2 = nn.ReLU()

            self.fc3 = nn.Linear(16, 10)
            


        def forward(self, x):

            x = self.fc1(x)
            x = self.tanh1(x)

            x = self.fc2(x)
            x = self.tanh2(x)

            x = self.fc3(x)

            return x
    # return model instance (None is just a placeholder)

    return model()
    

def count_parameters(model):
    # your code here
    numpar=sum([p.numel() for p in model.parameters() if p.requires_grad])
    
    # верните количество параметров модели model
    return numpar
