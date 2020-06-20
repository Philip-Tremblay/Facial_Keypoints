## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()       
        
        t = 64     # to manually scale the network
        
        # input tensor: (1, 224, 224)
        self.conv1 = nn.Conv2d(1, t, 5)
        # pooling
        # output tensor: (t, 110, 110)
        #self.conv1_bn = nn.BatchNorm2d(t)
        
        self.conv2 = nn.Conv2d(t, 2*t, 5)
        # pooling
        # output tensor: (2t, 54, 54)
        #self.conv2_bn = nn.BatchNorm2d(2*t)
        
        self.conv3 = nn.Conv2d(2*t, 4*t, 3)
        # pooling
        # output tensor: (4t, 26, 26)
        #self.conv3_bn = nn.BatchNorm2d(4*t)
        
        self.conv4 = nn.Conv2d(4*t, 8*t, 2)
        # pooling
        # output tensor: (8t, 12, 12) 
        #self.conv4_bn = nn.BatchNorm2d(8*t)
        
        self.conv5 = nn.Conv2d(8*t, 16*t, 2)
        # pooling
        # output tensor: (8t, 5, 5) 
        #self.conv5_bn = nn.BatchNorm2d(16*t)       
        
        
        self.pool = nn.MaxPool2d(2, 2)
        

        self.fc1 = nn.Linear(16*t*5*5, 4*t*5*5)
        self.fc2 = nn.Linear(4*t*5*5, 2*t*5*5)
        self.fc3 = nn.Linear(2*t*5*5, 136)
        
        
        # dropout with p=0.4
        self.dropout = nn.Dropout(p=0.4)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        '''
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.pool(F.relu(self.conv5_bn(self.conv5(x))))
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # flatten
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        

        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
