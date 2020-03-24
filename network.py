import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as nprandom

'''
DNN based on Vera-Diaz et al. (2018) - Towards End-to-End Acoustic Localization using Deep Learning: from Audio Signal to Source Position Coordinates
Structure:
#     Layer       Output  Kernel  Activation  
---------------------------------------------
1     Conv1       96      7       ReLu
2     MaxPool1    96      7
3     Conv2       96      7       ReLu
4     Conv3       128     5       ReLu
5     MaxPool3    128     5
6     Conv4       128     5       ReLu
7     MaxPool4    128     5
8     Conv5       128     3       ReLu
9     FC          500             ReLu        
10    Output (FC) 1
'''
class SSLConvNet(nn.Module):

    def __init__(self, signal_length, do_cosine_output = False):
        super(SSLConvNet, self).__init__()

        # To record activations
        self.fc_activations = []

        self.conv1 = nn.Conv1d(2, 96, 7, padding=7//2)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.maxp1 = nn.MaxPool1d(7, stride=7)

        self.conv2 = nn.Conv1d(96, 96, 7, padding = 7//2)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv1d(96, 128, 5, padding=5//2)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.maxp3 = nn.MaxPool1d(5, stride=5)

        self.conv4 = nn.Conv1d(128, 128, 5, padding=5//2)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.maxp4 = nn.MaxPool1d(5, stride=5)

        self.conv5 = nn.Conv1d(128, 128, 3, padding=3//2)
        nn.init.xavier_uniform_(self.conv5.weight)

        self.fc = nn.Linear(self.compute_flattened_size(signal_length), 500)
        nn.init.xavier_uniform_(self.fc.weight)

        if do_cosine_output:
            self.out_x = nn.Linear(500, 1)
            nn.init.xavier_uniform_(self.out_x.weight)
            self.out_y = nn.Linear(500, 1)
            nn.init.xavier_uniform_(self.out_y.weight)
        else:
            self.out = nn.Linear(500, 1)
            nn.init.xavier_uniform_(self.out_y.weight)

        self.act = nn.ReLU()

    def compute_flattened_size(self, signal_length):
        new_size = (signal_length - 7)//7 + 1
        new_size = (new_size - 5)//5 + 1
        new_size = (new_size - 5)//5 + 1
        return 128 * new_size

    def forward(self, inL, inR, record_activations = False):
        # Add extra "channel" dimension for convolutional layers
        inL = inL.view(inL.size(0), 1, inL.size(1))
        inR = inR.view(inR.size(0), 1, inR.size(1))

        # Concatenate input signals
        x = torch.cat((inL, inR), dim = 1)
        x = self.act(self.conv1(x))
        x = self.maxp1(x)
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.maxp3(x)
        x = self.act(self.conv4(x))
        x = self.maxp4(x)
        x = self.act(self.conv5(x))

        x = x.view(x.size(0), -1)
        x = self.act(self.fc(x))
        
        if record_activations:
            self.fc_activations.append(torch.mean(x).detach().cpu().numpy())
        
        if self.do_cosine_output:
            return self.out(x)
        else:
            return self.out_x(x), self.out_y(x)


# class SSLConvNetCosLoss(nn.Module):
#     def __init__(self, signal_length):
#         super(SSLConvNetCosLoss, self).__init__()

#         self.conv1 = nn.Conv1d(2, 96, 7, padding=7//2)
#         nn.init.xavier_uniform_(self.conv1.weight)
#         self.maxp1 = nn.MaxPool1d(7, stride=7)

#         self.conv2 = nn.Conv1d(96, 96, 7, padding = 7//2)
#         nn.init.xavier_uniform_(self.conv2.weight)

#         self.conv3 = nn.Conv1d(96, 128, 5, padding=5//2)
#         nn.init.xavier_uniform_(self.conv3.weight)
#         self.maxp3 = nn.MaxPool1d(5, stride=5)

#         self.conv4 = nn.Conv1d(128, 128, 5, padding=5//2)
#         nn.init.xavier_uniform_(self.conv4.weight)
#         self.maxp4 = nn.MaxPool1d(5, stride=5)

#         self.conv5 = nn.Conv1d(128, 128, 3, padding=3//2)
#         nn.init.xavier_uniform_(self.conv5.weight)

#         self.fc = nn.Linear(self.compute_flattened_size(signal_length), 500)
#         nn.init.xavier_uniform_(self.fc.weight)
#         self.out = nn.Linear(500, 1)
#         nn.init.xavier_uniform_(self.out.weight)

#         self.act = nn.ReLU()
        
#     def compute_flattened_size(self, signal_length):
#         new_size = (signal_length - 7)//7 + 1
#         new_size = (new_size - 5)//5 + 1
#         new_size = (new_size - 5)//5 + 1
#         return 128 * new_size
        
#     def forward(self, inL, inR, debug=False):
#         # Add extra "channel" dimension for convolutional layers
#         inL = inL.view(inL.size(0), 1, inL.size(1))
#         inR = inR.view(inR.size(0), 1, inR.size(1))

#         # Concatenate input signals
#         x = torch.cat((inL, inR), dim=1)

#         x = self.act(self.conv1(x))
#         x = self.maxp1(x)
#         x = self.act(self.conv2(x))
#         x = self.act(self.conv3(x))
#         x = self.maxp3(x)
#         x = self.act(self.conv4(x))
#         x = self.maxp4(x)
#         x = self.act(self.conv5(x))

#         x = x.view(x.size(0), -1)
#         x = self.act(self.fc(x))

#         return self.out(x)

"""
BASELINE MODEL                   
"""
class SimpleNet(nn.Module):
    def __init__(self, signalLength, p=0.5):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(signalLength*2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.outX = nn.Linear(64, 1)
        self.outY = nn.Linear(64, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

    def forward(self, inL, inR):
        # Concatenate input signals
        x = torch.cat((inL, inR), dim = 1)

        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.dropout(self.act(self.fc3(x)))
        x = self.dropout(self.act(self.fc4(x)))
        x = self.dropout(self.act(self.fc5(x)))

        return self.outX(x), self.outY(x)