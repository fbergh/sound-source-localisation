#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
import numpy as np

class SineData(Dataset):

    def __init__(self, batchSize, roomSim, time, minLength, maxFreq):
        self.batchSize = batchSize
        self.roomSim = roomSim
        self.time = time
        self.sampleRate = self.roomSim.getSampleRate()
        self.MIN_LENGTH = minLength # Different for different signals
        self.MAX_FREQ = maxFreq
        self.signalMean, self.signalStd = self.getMeanAndStd()

    def __len__(self):
        return self.batchSize

    def __getitem__(self, idx):
        azi = 2*np.pi*np.random.random_sample()
        amp = np.random.randint(1, 500 + 1)
        freq = np.random.randint(20, self.MAX_FREQ + 1)

        signalL, signalR, ssX, ssY = self.generateSignals(azi, amp, freq)
 
        # Standardise by sample mean and std (pyroomacoustics gives really small values sometimes)
        signalL = (signalL - self.signalMean) / self.signalStd
        signalR = (signalR - self.signalMean) / self.signalStd

        # Convert numpy arrays to torch tensors and return X and Y coordinates in meters
        return torch.from_numpy(signalL), torch.from_numpy(signalR), ssX * 100, ssY * 100

    def getMeanAndStd(self):
        nrOfSamples = 1000
        meansL = []
        meansR = []
        varsL = []
        varsR = []
        
        for _ in range(nrOfSamples):
            azi = 2*np.pi*np.random.random_sample()
            amp = np.random.randint(1, 500 + 1)
            freq = np.random.randint(20, self.MAX_FREQ + 1)

            signalL, signalR, _, _ = self.generateSignals(azi, amp, freq)
            meansL.append(np.mean(signalL))
            meansR.append(np.mean(signalR))
            varsL.append(np.var(signalL))
            varsR.append(np.var(signalR))

        # We can only take mean of means because they are all equally long
        meanL = np.mean(meansL)
        meanR = np.mean(meansR)
        meanVarL = np.mean(varsL)
        meanVarR = np.mean(varsR)

        # Because signals are symmetrical due to mic positions, 
        # we can average the means and variances of both signals
        mean = (meanL + meanR)/2
        # avg_std = sqrt(avg_variance)
        std = np.sqrt((meanVarL + meanVarR)/2)
        return mean, std

    def generateSignals(self, azi, amp, freq):
        sinWave = amp * np.sin(2 * np.pi * np.arange(self.sampleRate * self.time) * (freq / self.sampleRate))
        ssX, ssY = self.roomSim.setSoundSource(azi, sinWave)
        self.roomSim.simulateSoundPropagation()
        signalL, signalR = self.roomSim.getPropagatedSignals()

        # Make sure the signals have the same length as the others
        if len(signalL) < self.MIN_LENGTH:
            signalL = np.append(signalL, np.zeros(self.MIN_LENGTH-len(signalL)))
        if len(signalR) < self.MIN_LENGTH:
            signalR = np.append(signalR, np.zeros(self.MIN_LENGTH-len(signalR)))

        return signalL, signalR, ssX, ssY

    def gatherSignals(self):
        signalsL = []
        signalsR = []
        for azi in np.linspace(0,2*np.pi,360,endpoint=False):
            amp = 500#np.random.randint(1, 500 + 1)
            freq = self.MAX_FREQ#np.random.randint(20, self.MAX_FREQ)

            signalL, signalR, _, _ = self.generateSignals(azi, amp, freq)
            signalsL.append(signalL)
            signalsR.append(signalR)
        return signalsL, signalsR

    def computeErrorMatrixSignals(self):
        signalsL, signalsR = self.gatherSignals()
        errorMatrix = np.zeros((len(signalsL), len(signalsL)))

        for i in range(len(signalsL)):
            for j in range(i,len(signalsL)):
                normErrorL = self.normalisedError(signalsL[i], signalsL[j])
                normErrorR = self.normalisedError(signalsR[i], signalsR[j])
                errorMatrix[i,j] = normErrorL + normErrorR
                errorMatrix[j,i] = errorMatrix[i,j]
                
        return errorMatrix

    def normalisedError(self, signal1, signal2):
        return np.sum(np.square(signal1 - signal2))/np.sum(np.square(signal1))
