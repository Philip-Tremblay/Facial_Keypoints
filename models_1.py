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
        
        t = 32     # to manually scale the network
        
        # input tensor: (1, 224, 224)
        self.conv1 = nn.Conv2d(1, t, 5)
        # pooling
        # output tensor: (t, 110, 110)
        #self.conv1_bn = nn.BatchNorm2d(t)
        
        self.conv2 = nn.Conv2d(t, 2*t, 5)
        # pooling
        # output tensor: (6, 54, 54)
        #self.conv2_bn = nn.BatchNorm2d(2*t)
        
        self.conv3 = nn.Conv2d(2*t, 3*t, 3)
        # pooling
        # output tensor: (12, 26, 26)
        #self.conv3_bn = nn.BatchNorm2d(3*t)
        
        self.conv4 = nn.Conv2d(3*t, 4*t, 2)
        # pooling
        # output tensor: (12, 12, 12) 
        #self.conv4_bn = nn.BatchNorm2d(4*t)
        
        
        self.pool = nn.MaxPool2d(2, 2)
        
        cv_size=t*12*12
        
        self.fc1 = nn.Linear(4*cv_size, 2*cv_size)
        self.fc2 = nn.Linear(2*cv_size, cv_size)
        self.fc3 = nn.Linear(cv_size, 136)
        
        
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
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        
        # flatten
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        

        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
