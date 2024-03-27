from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import os
import numpy as np
import pickle
from scipy.spatial.distance import cosine
import random

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 10 channels in first convolution layer
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 20 channels in second conv. layer
        self.fc1 = nn.Linear(320, 50) # 50 hidden units in first fully-connected layer
        self.fc2 = nn.Linear(50, 10) # 10 output units

    def forward(self, x):
        # first convolutional layer
        h_conv1 = self.conv1(x)
        h_conv1 = F.relu(h_conv1)
        h_conv1_pool = F.max_pool2d(h_conv1, 2)

        # second convolutional layer
        h_conv2 = self.conv2(h_conv1_pool)
        h_conv2 = F.relu(h_conv2)
        h_conv2_pool = F.max_pool2d(h_conv2, 2)

        # fully-connected layer
        h_fc1 = h_conv2_pool.view(-1, 320)
        h_fc1 = self.fc1(h_fc1)
        h_fc1 = F.relu(h_fc1)

        # classifier output
        h_fc2 = self.fc2(h_fc1)
        output = F.log_softmax(h_fc2, dim=1)
        return output
    
class Net10(nn.Module):
    def __init__(self):
        super(Net10, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=2) # 5 channels in first convolution layer
        self.conv2 = nn.Conv2d(5, 5, kernel_size=1) # 5 channels in second conv. layer
        self.conv3 = nn.Conv2d(5, 10, kernel_size=2) # 10 channels in third conv. layer
        self.conv4 = nn.Conv2d(10, 10, kernel_size=1) # 10 channels in forth conv. layer
        self.conv5 = nn.Conv2d(10, 20, kernel_size=2) # 20 channels in fifth conv. layer
        self.conv6 = nn.Conv2d(20, 40, kernel_size=2) # 40 channels in sixth conv. layer
        self.conv7 = nn.Conv2d(40, 80, kernel_size=1) # 80 channels in seventh conv. layer
        self.fc1 = nn.Linear(320, 160) # 160 hidden units in first fully-connected layer
        self.fc2 = nn.Linear(160, 50) # 50 output units
        self.fc3 = nn.Linear(50, 10) # 10 output units

    def forward(self, x):
        # convolutional layer
        h_conv1 = self.conv1(x)
        h_conv1 = F.relu(h_conv1)

        h_conv2 = self.conv2(h_conv1)
        h_conv2 = F.relu(h_conv2)
        h_conv2_pool = F.max_pool2d(h_conv2, 2)

        h_conv3 = self.conv3(h_conv2_pool)
        h_conv3 = F.relu(h_conv3)

        h_conv4 = self.conv4(h_conv3)
        h_conv4 = F.relu(h_conv4)
        h_conv4_pool = F.max_pool2d(h_conv4, 2)

        h_conv5 = self.conv5(h_conv4_pool)
        h_conv5 = F.relu(h_conv5)

        h_conv6 = self.conv6(h_conv5)
        h_conv6 = F.relu(h_conv6)

        h_conv7 = self.conv7(h_conv6)
        h_conv7 = F.relu(h_conv7)
        h_conv7_pool = F.max_pool2d(h_conv7, 2)

        # fully-connected layer
        h_fc1 = h_conv7_pool.view(-1, 320)
        h_fc1 = self.fc1(h_fc1)
        h_fc1 = F.relu(h_fc1)

        h_fc2 = self.fc2(h_fc1)
        h_fc2 = F.relu(h_fc2)

        h_fc3 = self.fc3(h_fc2)
        output = F.log_softmax(h_fc3,dim=1)
        return h_conv1,h_conv2,h_conv3,h_conv4,h_conv5,h_conv6,h_conv7,h_fc1,h_fc2,h_fc3,output