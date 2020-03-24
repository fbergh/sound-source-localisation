#%%
# Initialise variables and imports
import pyroomacoustics.doa as pra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from room_simulation import Simulation
from sine_dataset import SineData
from network import SSLConvNet as ConvNet
import matplotlib.pyplot as plt
import math
import gc

#%%
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
MODELS_PATH = "../"
DATA_PATH = "../thesis_data/"
BATCH_SIZE_EXP2 = 5
BATCH_SIZE_EXP1 = 25
TEST_SIZE = 10000
EPOCHS_EXP2 = int(TEST_SIZE/BATCH_SIZE_EXP2)
EPOCHS_EXP1 = int(TEST_SIZE/BATCH_SIZE_EXP1)
RAD = 50
RADII = [50, 500, 5000]
ROOM_SIMS = []
DATASETS_EXP2 = []
MIC_L_DIST = (11, -10)
MIC_R_DIST = (11, -10)
ABSORPTION = 0.0
MIN_FREQ = 20
MAX_FREQ = 20000
SAMPLE_RATE = int(MAX_FREQ*2.2)
TIME = 1
MIN_LENGTH_EXP2 = 65000
MIN_LENGTH_EXP1 = 48000
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

def get_circumference_pos(x1, y1, r, theta):
    x_positions = x1 + r*torch.sin(theta)
    x_positions = r*2-x_positions #Flip x-coordinates
    y_positions = y1 + r*(1-torch.cos(theta))
    y_positions = r*2-y_positions #Flip y-coordinates
    return x_positions, y_positions

def get_circumference_angle(cx, cy, x, y):
    theta = torch.atan2(cy-y, cx-x)
    theta = torch.remainder(torch.remainder(theta, 2*np.pi) + np.pi/2, 2*np.pi)
    return theta

def get_mean_ITD(inL, inR):
    non_zero_idcs_inL = np.argmax((inL > 0))
    non_zero_idcs_inR = np.argmax((inR > 0))
    
    return (non_zero_idcs_inL/SAMPLE_RATE - non_zero_idcs_inR/SAMPLE_RATE) * 1000

def get_mean_IID(inL, inR):
    peak_inL_idxs = pra.detect_peaks(inL)
    peak_inR_idxs = pra.detect_peaks(inR)

    len_diff = len(peak_inL_idxs) - len(peak_inR_idxs)
    if len_diff != 0:
        if len_diff > 0:
            peak_inL_idxs = peak_inL_idxs[len_diff:]
        else:
            peak_inR_idxs = peak_inR_idxs[abs(len_diff):]
    
    return np.mean((inL[peak_inL_idxs] - inR[peak_inR_idxs]))


#%%
for rad in RADII:
    print(rad)
    room_simulation = Simulation(SAMPLE_RATE, rad, ABSORPTION,MIC_L_DIST, MIC_R_DIST)
    ROOM_SIMS.append(room_simulation)

    DATASETS_EXP2.append(SineData(BATCH_SIZE_EXP2, room_simulation, TIME, MIN_LENGTH_EXP2, MIN_FREQ, MAX_FREQ))

roomSimMSECos = ROOM_SIMS[0]
datasetMSECos = SineData(BATCH_SIZE_EXP1, roomSimMSECos, TIME, MIN_LENGTH_EXP1, MIN_FREQ, MAX_FREQ)

room_simulation = Simulation(SAMPLE_RATE, 50, ABSORPTION,MIC_L_DIST, MIC_R_DIST)
datasat_low_frequency    = SineData(1, room_simulation, TIME, MIN_LENGTH_EXP1, 20, 1500)
dataset_medium_frequency = SineData(1, room_simulation, TIME, MIN_LENGTH_EXP1, 1500, 3000)
dataset_high_frequency   = SineData(1, room_simulation, TIME, MIN_LENGTH_EXP1, 3000, 20000)
DATASETS_EXP3 = [datasat_low_frequency, dataset_medium_frequency, dataset_high_frequency]


#%%
# Create networks
nets = []
for rad in RADII:
    print(rad)
    net = ConvNet(MIN_LENGTH_EXP2).double()
    net.load_state_dict(torch.load(MODELS_PATH+"ConvNet_Rad"+str(rad)+".pt", map_location="cpu"))
    net.eval()
    nets.append(net)

mse_net = ConvNet(MIN_LENGTH_EXP1).double()
mse_net.load_state_dict(torch.load(MODELS_PATH+"ConvNet_MSELoss.pt", map_location="cpu"))
mse_net.eval()

cos_net = ConvNet(MIN_LENGTH_EXP1, do_cosine_output=True).double()
cos_net.load_state_dict(torch.load(MODELS_PATH+"ConvNet_CosLoss.pt", map_location="cpu"))
cos_net.eval()


#%%
################
# EXPERIMENT 1 #
################

cos_loss = CosBorderLoss()
mse_loss = nn.MSELoss()
cos_losses = np.zeros((2, EPOCHS_EXP1))
mse_losses = np.zeros((2, EPOCHS_EXP1))
dataloader = DataLoader(datasetMSECos, batch_size=BATCH_SIZE_EXP1)

with torch.no_grad():
    for i in range(EPOCHS_EXP1):
        if i%100 == 0:
            print("Epoch "+str(i)+"/"+str(EPOCHS_EXP1))
        
        # Retrieve all input data and labels
        inL, inR, labelX, labelY, label_azimuth = next(iter(dataloader))
        inL = inL.double().to(device)
        inR = inR.double().to(device)
        labelX = labelX.double().to(device)
        labelY = labelY.double().to(device)
        label_azimuth = label_azimuth.double().to(device)

        # Compute MSE and Cos losses for coordinate network
        mse_net = mse_net.to(device)
        
        outputX_mse, outputY_mse = mse_net(inL, inR)
        outputX_mse = torch.squeeze(outputX_mse)
        outputY_mse = torch.squeeze(outputY_mse)
        
        output_azimuth_mse = get_circumference_angle(50, 50, outputX_mse, outputY_mse)
        
        cos_losses[0,i] = cos_loss(output_azimuth_mse, label_azimuth)
        mse_losses[0,i] = mse_loss(outputX_mse, labelX) + mse_loss(outputY_mse, labelY)
        
        
        # Compute MSE and Cos losses for angle network
        cos_net = cos_net.to(device)
        
        output_azimuth_cos = cos_net(inL, inR)
        output_azimuth_cos = torch.squeeze(output_azimuth_cos)
        outputX_cos, outputY_cos = get_circumference_pos(50, 0, 50, output_azimuth_cos)
        
        cos_losses[1,i] = cos_loss(output_azimuth_cos, label_azimuth)
        mse_losses[1,i] = mse_loss(outputX_cos, labelX) + mse_loss(outputY_cos, labelY)

del mse_net, cos_net, inL, inR, labelX, labelY, label_azimuth
gc.collect()

np.save(DATA_PATH+"exp1_MSE_outcomes", mse_losses)
np.save(DATA_PATH+"exp1_Cos_outcomes", cos_losses)


#%%
mse_losses_exp1 = np.load(DATA_PATH+"exp1_MSE_outcomes.npy")
cos_losses_exp1 = np.load(DATA_PATH+"exp1_Cos_outcomes.npy")

print("MSE        COS")
print("MSE Losses")
print(np.mean(mse_losses_exp1, axis = 1))
print(np.std(mse_losses_exp1, axis = 1))

print("Cos Losses")
print(np.mean(cos_losses_exp1, axis = 1))
print(np.std(cos_losses_exp1, axis = 1))


#%%
################
# EXPERIMENT 2 #
################

mse_losses = np.zeros((len(DATASETS_EXP2), len(nets), EPOCHS_EXP2))

with torch.no_grad():
    for i,dataset in enumerate(DATASETS_EXP2):
        print("Dataset "+str(i))

        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE_EXP2)
        
        for j,net in enumerate(nets):
            print("Network "+str(j))
            net.to(device)
            for k in range(EPOCHS_EXP2):
                if k%400 == 0:
                    print("Epoch "+str(k)+"/"+str(EPOCHS_EXP2))
                
                inL, inR, labelX, labelY, _ = next(iter(dataloader))
                inL = inL.double().to(device)
                inR = inR.double().to(device)
                labelX = labelX.double().to(device)
                labelY = labelY.double().to(device)

                outputX, outputY = net(inL, inR)
                outputX = torch.squeeze(outputX)
                outputY = torch.squeeze(outputY)

                mse_losses[i,j,k] = mse_loss(outputX, labelX) + mse_loss(outputY, labelY)

            del net, inL, inR, labelX, labelY
            gc.collect()

np.save(DATA_PATH+"exp2_outcomes", mse_losses)


#%%
losses_exp2 = np.load(DATA_PATH+"exp2_outcomes.npy")

print("50    500    5000")
print("RAD = 50")
print(np.mean(losses_exp2[0], axis=1))
print(np.std(losses_exp2[0], axis=1))
print("RAD = 500")
print(np.mean(losses_exp2[1], axis=1))
print(np.std(losses_exp2[1], axis=1))
print("RAD = 5000")
print(np.mean(losses_exp2[2], axis=1))
print(np.std(losses_exp2[2], axis=1))


#%%
################
# EXPERIMENT 3 #
################

BATCH_SIZE_EXP3 = 1
TEST_SIZE_EXP3 = 1500
EPOCHS_EXP3 = int(TEST_SIZE_EXP3/BATCH_SIZE_EXP3)
ITDs = np.zeros((len(DATASETS_EXP3), EPOCHS_EXP3))
IIDs = np.zeros((len(DATASETS_EXP3), EPOCHS_EXP3))
threshold = 2e-12
mse_net = ConvNet(MIN_LENGTH_EXP1).double()
mse_net.load_state_dict(torch.load(MODELS_PATH+"ConvNet_MSELoss", map_location="cpu"))
mse_net.eval()
mse_net = mse_net.to(device)

with torch.no_grad():
    for i,dataset in enumerate(DATASETS_EXP3):
        print("Dataset "+str(i))
        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE_EXP3)
        for j in range(EPOCHS_EXP3):
            ITDs_batch = []
            IIDs_batch = []
            
            if j%100 == 0:
                    print("Epoch "+str(j)+"/"+str(EPOCHS_EXP3))
                    
            inLs, inRs, _, _, _ = next(iter(dataloader))
            
            for signalIdx in range(inLs.size(0)):
                inL = inLs[signalIdx].numpy()
                inR = inRs[signalIdx].numpy()

                inL[np.where(np.abs(inL) < threshold)] = 0
                inR[np.where(np.abs(inR) < threshold)] = 0

                ITDs_batch.append(get_mean_ITD(inL, inR))
                IIDs_batch.append(get_mean_IID(inL, inR))
            
            ITDs[i,j] = np.mean(ITDs_batch)
            IIDs[i,j] = np.mean(IIDs_batch)
            
            inLs = inLs.double().to(device)
            inRs = inRs.double().to(device)
            _, _ = mse_net(inLs, inRs, record_acts = True)
            
            del inLs, inRs
            gc.collect()

del net 
gc.collect()       
        
import scipy.stats

np.save(DATA_PATH+"exp3_ITDs.npy", ITDs)
np.save(DATA_PATH+"exp3_IIDs.npy", IIDs)


#%%
ITDs = np.load(DATA_PATH+"exp3_ITDs.npy")
IIDs = np.load(DATA_PATH+"exp3_IIDs.npy")

for i,freq in enumerate(["LowFreq","MedFreq","HigFreq"]):
    print(freq+" ITD correlation = " + str(scipy.stats.pearsonr(mse_net.fc_activations[1500*i:1500*(i+1)], ITDs[i])))
    print(freq+" IID correlation = " + str(scipy.stats.pearsonr(mse_net.fc_activations[1500*i:1500*(i+1)], IIDs[i])))

print(np.min(ITDs, axis=1))
print(np.max(ITDs, axis=1))

print(np.min(IIDs, axis=1))
print(np.max(IIDs, axis=1)) 