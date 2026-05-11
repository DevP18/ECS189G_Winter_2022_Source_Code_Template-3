'''
Method module
'''

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------
# abstract method class
class method:

    method_name = None
    method_description = None

    def __init__(self, mName=None, mDescription=None):

        self.method_name = mName
        self.method_description = mDescription

    @abc.abstractmethod
    def run(self, trainData, trainLabel, testData):
        return

# -----------------------------------------------------
# cnn method class
class cnn_method(method):

    def __init__(self, input_channel=1, num_classes=10):

        super().__init__(
            mName='cnn_method',
            mDescription='CNN image classification model'
        )

        self.model = cnn_model(
            input_channel,
            num_classes
        )

    def run(self, trainData, trainLabel, testData):

        return self.model(testData)

# -----------------------------------------------------
# cnn model
class cnn_model(nn.Module):

    def __init__(self, input_channel, num_classes):

        super(cnn_model, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_channel,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # adaptive pooling works for all image sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7,7))

        self.fc1 = nn.Linear(
            64 * 7 * 7,
            128
        )

        self.fc2 = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))   # no pooling here

        x = self.adaptive_pool(x)   # IMPORTANT FIX

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x