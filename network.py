import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as nprandom

#DNN based on Vera-Diaz et al. (2018) - Towards End-to-End Acoustic Localization using Deep Learning: from Audio Signal to Source Position Coordinates
'''
Structure:
#     Layer       Output  Kernel  Activation  Other
---------------------------------------------------------
1     Conv1       96      7       ReLu
2     MaxPool1    96      7
3     Conv2       96      7       ReLu
4     Conv3       128     5       ReLu
5     MaxPool3    128     5
6     Conv4       128     5       ReLu
7     MaxPool4    128     5
8     Conv5       128     3       ReLu
9     FC          500             ReLu        Dropout 0.5
10    Output (FC) 1
'''
class SSLConvNet(nn.Module):

    def __init__(self, signalLength):
        super(SSLConvNet, self).__init__()
        self.KERNEL_LAYER_12 = 7
        self.KERNEL_LAYER_34 = 5
        self.KERNEL_LAYER_5 = 3
        self.OUT_SIZE_LAYER_12 = 96
        self.OUT_SIZE_LAYER_345 = 128

        self.conv1 = nn.Conv1d(2, self.OUT_SIZE_LAYER_12, self.KERNEL_LAYER_12,
                               padding=int((self.KERNEL_LAYER_12 - 1)/2))
        nn.init.xavier_uniform_(self.conv1.weight)
        self.maxp1 = nn.MaxPool1d(self.KERNEL_LAYER_12, stride=self.KERNEL_LAYER_12)
        self.conv2 = nn.Conv1d(self.OUT_SIZE_LAYER_12, self.OUT_SIZE_LAYER_12, 
                               self.KERNEL_LAYER_12, padding = int((self.KERNEL_LAYER_12 - 1)/2))
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv1d(self.OUT_SIZE_LAYER_12, self.OUT_SIZE_LAYER_345, 
                               self.KERNEL_LAYER_34, padding=int((self.KERNEL_LAYER_34 - 1)/2))
        nn.init.xavier_uniform_(self.conv3.weight)
        self.maxp3 = nn.MaxPool1d(self.KERNEL_LAYER_34, stride=self.KERNEL_LAYER_34)
        self.conv4 = nn.Conv1d(self.OUT_SIZE_LAYER_345, self.OUT_SIZE_LAYER_345, 
                               self.KERNEL_LAYER_34, padding=int((self.KERNEL_LAYER_34 - 1)/2))
        nn.init.xavier_uniform_(self.conv4.weight)
        self.maxp4 = nn.MaxPool1d(self.KERNEL_LAYER_34, stride=self.KERNEL_LAYER_34)
        self.conv5 = nn.Conv1d(self.OUT_SIZE_LAYER_345, self.OUT_SIZE_LAYER_345, 
                               self.KERNEL_LAYER_5, padding=int((self.KERNEL_LAYER_5 - 1)/2))
        nn.init.xavier_uniform_(self.conv5.weight)
        self.fc = nn.Linear(self.computeConvFlatSize(signalLength), 500)
        nn.init.xavier_uniform_(self.fc.weight)
        self.outX = nn.Linear(500, 1)
        self.outY = nn.Linear(500, 1)
        nn.init.xavier_uniform_(self.outX.weight)
        nn.init.xavier_uniform_(self.outY.weight)
        self.act = nn.ReLU()

    def computeConvFlatSize(self, signLen):
        afterMaxP1 = int((signLen - self.KERNEL_LAYER_12)/self.KERNEL_LAYER_12) + 1
        afterMaxP3 = int((afterMaxP1 - self.KERNEL_LAYER_34)/self.KERNEL_LAYER_34) + 1
        afterMaxP4 = int((afterMaxP3 - self.KERNEL_LAYER_34)/self.KERNEL_LAYER_34) + 1
        return self.OUT_SIZE_LAYER_345 * afterMaxP4

    def forward(self, inL, inR, debug = False, isDropout1 = False, isDropout2 = False):
        # Add extra "channel" dimension for convolutional layers
        inL = inL.view(inL.size(0), 1, inL.size(1))
        inR = inR.view(inR.size(0), 1, inR.size(1))

        # Concatenate input signals
        x = torch.cat((inL, inR), dim = 1)
        if debug: 
            print("In = " + str(x.size()))

        x = self.act(self.conv1(x))
        if debug: 
            print("Conv1 = " + str(x.size()))

        x = self.maxp1(x)
        if debug: 
            print("Maxp1 = " + str(x.size()))

        x = self.act((self.conv2(x)))
        if debug: 
            print("Conv2 = " + str(x.size()))

        x = self.act((self.conv3(x)))
        if debug: 
            print("Conv3 = " + str(x.size()))

        x = self.maxp3(x)
        if debug: 
             print("Maxp3 = " + str(x.size()))

        x = self.act((self.conv4(x)))
        if debug: 
            print("Conv4 = " + str(x.size()))

        x = self.maxp4(x)
        if debug: 
            print("Maxp4 = " + str(x.size()))

        x = self.act((self.conv5(x)))
        if debug: 
            print("Conv5 = " + str(x.size()))

        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        if debug: 
            print("Reshape = " + str(x.size()))

        x = F.dropout(x, isDropout1)

        x = self.act(self.fc(x))
        if debug: 
            print("FC = " + str(x.size()))

        x = F.dropout(x, isDropout2)
        
        return self.outX(x), self.outY(x)


class SSLConvNetCosLoss(nn.Module):

    def __init__(self, signalLength):
        super(SSLConvNetCosLoss, self).__init__()
        self.KERNEL_LAYER_12 = 7
        self.KERNEL_LAYER_34 = 5
        self.KERNEL_LAYER_5 = 3
        self.OUT_SIZE_LAYER_12 = 96
        self.OUT_SIZE_LAYER_345 = 128

        self.conv1 = nn.Conv1d(2, self.OUT_SIZE_LAYER_12, self.KERNEL_LAYER_12,
                               padding=int((self.KERNEL_LAYER_12 - 1)/2))
        nn.init.xavier_uniform_(self.conv1.weight)
        self.maxp1 = nn.MaxPool1d(self.KERNEL_LAYER_12, stride=self.KERNEL_LAYER_12)
        self.conv2 = nn.Conv1d(self.OUT_SIZE_LAYER_12, self.OUT_SIZE_LAYER_12, 
                               self.KERNEL_LAYER_12, padding = int((self.KERNEL_LAYER_12 - 1)/2))
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv1d(self.OUT_SIZE_LAYER_12, self.OUT_SIZE_LAYER_345, 
                               self.KERNEL_LAYER_34, padding=int((self.KERNEL_LAYER_34 - 1)/2))
        nn.init.xavier_uniform_(self.conv3.weight)
        self.maxp3 = nn.MaxPool1d(self.KERNEL_LAYER_34, stride=self.KERNEL_LAYER_34)
        self.conv4 = nn.Conv1d(self.OUT_SIZE_LAYER_345, self.OUT_SIZE_LAYER_345, 
                               self.KERNEL_LAYER_34, padding=int((self.KERNEL_LAYER_34 - 1)/2))
        nn.init.xavier_uniform_(self.conv4.weight)
        self.maxp4 = nn.MaxPool1d(self.KERNEL_LAYER_34, stride=self.KERNEL_LAYER_34)
        self.conv5 = nn.Conv1d(self.OUT_SIZE_LAYER_345, self.OUT_SIZE_LAYER_345, 
                               self.KERNEL_LAYER_5, padding=int((self.KERNEL_LAYER_5 - 1)/2))
        nn.init.xavier_uniform_(self.conv5.weight)
        self.fc = nn.Linear(self.computeConvFlatSize(signalLength), 500)
        nn.init.xavier_uniform_(self.fc.weight)
        self.out = nn.Linear(500, 1)
        nn.init.xavier_uniform_(self.out.weight)
        self.act = nn.ReLU()
        
    def computeConvFlatSize(self, signLen):
        afterMaxP1 = int((signLen - self.KERNEL_LAYER_12)/self.KERNEL_LAYER_12) + 1
        afterMaxP3 = int((afterMaxP1 - self.KERNEL_LAYER_34)/self.KERNEL_LAYER_34) + 1
        afterMaxP4 = int((afterMaxP3 - self.KERNEL_LAYER_34)/self.KERNEL_LAYER_34) + 1
        return self.OUT_SIZE_LAYER_345 * afterMaxP4
        
    def forward(self, inL, inR, debug=False):
        # Add extra "channel" dimension for convolutional layers
        inL = inL.view(inL.size(0), 1, inL.size(1))
        inR = inR.view(inR.size(0), 1, inR.size(1))

        # Concatenate input signals
        x = torch.cat((inL, inR), dim=1)
        if debug:
            print("In = " + str(x.size()))

        x = self.act((self.conv1(x)))
        if debug:
            print("Conv1 = " + str(x.size()))

        x = self.maxp1(x)
        if debug:
            print("Maxp1 = " + str(x.size()))

        x = self.act((self.conv2(x)))
        if debug:
            print("Conv2 = " + str(x.size()))

        x = self.act((self.conv3(x)))
        if debug:
            print("Conv3 = " + str(x.size()))

        x = self.maxp3(x)
        if debug:
             print("Maxp3 = " + str(x.size()))

        x = self.act((self.conv4(x)))
        if debug:
            print("Conv4 = " + str(x.size()))

        x = self.maxp4(x)
        if debug:
            print("Maxp4 = " + str(x.size()))

        x = self.act((self.conv5(x)))
        if debug:
            print("Conv5 = " + str(x.size()))

        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        if debug:
            print("Reshape = " + str(x.size()))

        x = self.act(self.fc(x))
        if debug:
            print("FC = " + str(x.size()))

        angles = self.out(x)

        return angles

"""
ONE FREQUENCY AND AMP:
    Converges around 4000 epochs for:
        RAD = 50000                                         
        MIC_L_DIST = (20000, 10000)                         
        MIC_R_DIST = (18000, 12000)      
        ABSORPTION = 1.0
        MIN_FREQ = 50
        SAMPLE_RATE = MIN_FREQ*2.5                            
        TIME = 1                                            
        MIN_LENGTH = 900    

    Converges around 1100 epochs for:
        RAD = 1000                                         
        MIC_L_DIST = (0.4*RAD, 0.2*RAD)                         
        MIC_R_DIST = (0.36*RAD, 0.24*RAD)               
        ABSORPTION = 1.0
        MIN_FREQ = 50
        SAMPLE_RATE = MIN_FREQ*2.5                            
        TIME = 1                                           
        MIN_LENGTH = 350                                   
        TO_RAD = np.pi/180                                  
        TO_DEG = 180/np.pi      

                            
"""
class SimpleNet(nn.Module):

    def __init__(self, signalLength):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(signalLength*2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        # self.out = nn.Linear(32, 1)
        self.outX = nn.Linear(32, 1)
        self.outY = nn.Linear(32, 1)
        self.act = nn.ReLU()

    def forward(self, inL, inR, p = 0.5, isDropout = True):
        # Concatenate input signals
        x = torch.cat((inL, inR), dim = 1)

        x = F.dropout(self.act(self.fc1(x)), p = p, training = isDropout)
        x = F.dropout(self.act(self.fc2(x)), p = p, training = isDropout)
        x = F.dropout(self.act(self.fc3(x)), p = p, training = isDropout)
        x = F.dropout(self.act(self.fc4(x)), p = p, training = isDropout)
        x = F.dropout(self.act(self.fc5(x)), p = p, training = isDropout)
        x = F.dropout(self.act(self.fc6(x)), p = p, training = isDropout)
        # x = self.out(x)

        # return x
        return self.outX(x), self.outY(x)