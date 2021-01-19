## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import time



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # First Convo Layer
        self.conv1 = nn.Conv2d(1, 32, 3)
        # Second Convo Layer
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Third Convo Layer
        self.conv3 = nn.Conv2d(64, 128, 5)        
        
        nn.init.normal_(self.conv1.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=1.0)
        
        # Maxpool
        self.pool = nn.MaxPool2d(2, 2)
        self.poool = nn.MaxPool2d(3, 3)
        
        # Normalization
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        
        # Linear Layers 
        self.fc1 = nn.Linear(32768, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 136)
        
        # Dropout
        self.drop = nn.Dropout(p=0.1)
        self.droop = nn.Dropout(p=0.25)


        
    def forward(self, x):
                
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.norm1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.norm2(x)
        
        x = self.poool( F.relu(self.conv3(x)))
                
        x = self.norm3(x)
        
                       
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        
        x = self.drop(x)
               
        x = F.relu(self.fc2(x))
        
        x = self.droop(x)
                
        x = self.fc3(x)

        return x
