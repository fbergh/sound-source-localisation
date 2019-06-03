#%%
# Initialise variables and imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from room_simulation import Simulation
from sine_dataset import SineData
from network import SSLConvNet as ConvNet
from network import SSLConvNetCosLoss as ConvNetCosLoss
from logger import Logger
import matplotlib.pyplot as plt
import math
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_PATH = "../models/"
BATCH_SIZE = 50
TEST_SIZE = 10000
EPOCHS = TEST_SIZE/BATCH_SIZE
NR_MICS = 2
RAD = 50
RADII = [50, 500, 5000]
ROOM_SIMS = []
DATASETS = []
MIC_L_DIST = (11, -10)
MIC_R_DIST = (11, 10)
ABSORPTION = 0.0
MAX_FREQ = 20000
SAMPLE_RATE = int(MAX_FREQ*2.2)
TIME = 1
MIN_LENGTH = 65000
MIN_LENGTH_MSECOS = 48000
TO_RAD = np.pi/180
TO_DEG = 180/np.pi

# Define custom loss function
class CosBorderLoss(torch.nn.Module):

    def __init__(self):
        super(CosBorderLoss, self).__init__()

    def forward(self, pred, target):
        radial = torch.abs(torch.cos(pred-target) - torch.cos(target-target))
        border = nn.functional.relu(pred-2*np.pi) + nn.functional.relu(-pred)
        return torch.sum(radial + border)

def calcCircumPos(x1, y1, r, theta):
    xPositions = torch.add(x1, torch.mul(r, torch.sin(theta)))
    yPositions = torch.add(y1, torch.mul(r, torch.add(torch.ones(theta.size()), -torch.cos(theta))))
    return xPositions, yPositions

def calcCircumAngle(cx, cy, x, y):
    theta = torch.add(torch.atan2(torch.add(y, -cy), torch.add(cx, -x)), np.pi/2)
    return theta

for rad in RADII:
    roomSim = Simulation(SAMPLE_RATE, rad, ABSORPTION,MIC_L_DIST, MIC_R_DIST, NR_MICS)
    ROOM_SIMS.append(roomSim)

    DATASETS.append(SineData(BATCH_SIZE, roomSim, TIME, MIN_LENGTH, MAX_FREQ))

roomSimMSECos = ROOM_SIMS[0]
datasetMSECos = SineData(BATCH_SIZE, roomSimMSECos, TIME, MIN_LENGTH_MSECOS, MAX_FREQ)

#%%
# Create networks
nets = []
for rad in RADII:
    net = ConvNet(MIN_LENGTH).double()
    net.load_state_dict(torch.load(MODELS_PATH+"ConvNet_Rad"+str(rad), map_location=device))
    net.eval()
    nets.append(net)

MSENet = ConvNet(MIN_LENGTH_MSECOS).double()
MSENet.load_state_dict(torch.load(MODELS_PATH+"ConvNet_MSELoss", map_location=device))
MSENet.eval()

CosNet = ConvNet(MIN_LENGTH_MSECOS).double()
CosNet.load_state_dict(torch.load(MODELS_PATH+"ConvNet_CosLoss", map_location=device))
CosNet.eval()

#%%
################
# EXPERIMENT 1 #
################

CosLoss = CosBorderLoss()
MSELoss = nn.MSELoss()
CosLosses = np.zeros((2, EPOCHS))
MSELosses = np.zeros((2, EPOCHS))
dataLoader = DataLoader(datasetMSECos) 
MSENet = MSENet.to(device)
CosNet = CosNet.to(device)

for i in range(EPOCHS):

    inL, inR, labelX, labelY, labelAzi = next(iter(dataLoader))
    inL = inL.double().to(device)
    inR = inR.double().to(device)
    labelX = labelX.double().to(device)
    labelY = labelY.double().to(device)
    labelAzi = labelAzi.double().to(device)

    outputXMSE, outputYMSE = MSENet(inL, inR)
    outputAziCos = CosNet(inL, inR)

    outputAziMSE = calcCircumAngle(50, 50, outputXMSE, outputYMSE)
    outputXCos, outputYCos = calcCircumPos(50, 0, 50, outputAziCos)

    CosLosses[0,i] = CosLoss(outputAziMSE)
    CosLosses[1,i] = CosLoss(outputAziCos)
    MSELosses[0,i] = MSELoss(outputXMSE, labelX) + MSELoss(outputYMSE, labelY)
    MSELosses[1,i] = MSELoss(outputXCos, labelX) + MSELoss(outputYCos, labelY)

#%%
################
# EXPERIMENT 2 #
################

MSELosses = np.zeros((len(DATASETS), len(nets), EPOCHS))

for i,dataset in enumerate(DATASETS):

    dataLoader = DataLoader(dataset)
    
    for j,net in enumerate(nets):

        net.to(device)

        for k in range(EPOCHS):
            
            inL, inR, labelX, labelY, _ = next(iter(dataLoader))
            inL = inL.double().to(device)
            inR = inR.double().to(device)
            labelX = labelX.double().to(device)
            labelY = labelY.double().to(device)

            outputX, outputY = net(inL, inR)

            MSELosses[i,j,k] = MSELoss(outputX, labelX) + MSELoss(outputY, labelY)

        del net, inL, inR, labelX, labelY

np.save("exp2_outcomes", MSELosses)