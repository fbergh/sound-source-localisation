import torch
from torch.utils.data import Dataset
import numpy as np

class SineData(Dataset):

    def __init__(self, batch_size, room_simulation, time, min_length, min_frequency, max_frequency):
        self.batch_size = batch_size
        self.room_simulation = room_simulation
        self.time = time
        self.sample_rate = self.room_simulation.get_sample_rate()
        self.MIN_LENGTH = min_length # Different for different signals
        self.MIN_FREQUENCY = min_frequency
        self.MAX_FREQUENCY = max_frequency
        self.signalMean, self.signalStd = self.get_mean_std() # Compute mean and std to standardise signals

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        # Generate sound source at random angle with amplitude in [1,500] and frequency in [MIN_FREQUENCY, MAX_FREQUENCY]
        azi = 2*np.pi*np.random.random_sample()
        amp = np.random.randint(1, 500 + 1)
        freq = np.random.randint(self.MIN_FREQUENCY, self.MAX_FREQUENCY + 1)

        signalL, signalR, ssX, ssY = self.generate_signals(azi, amp, freq)
 
        # Standardise by sample mean and std (pyroomacoustics gives really small values sometimes)
        signalL = (signalL - self.signalMean) / self.signalStd
        signalR = (signalR - self.signalMean) / self.signalStd

        # Convert numpy arrays to torch tensors and return X and Y coordinates in meters
        return torch.from_numpy(signalL), torch.from_numpy(signalR), ssX * 100, ssY * 100, azi

    def get_mean_std(self, n_samples=1000):
        meansL, varsL = [], []
        meansR, varsR = [], []
        
        for _ in range(n_samples):
            azi = 2*np.pi*np.random.random_sample()
            amp = np.random.randint(1, 500 + 1)
            freq = np.random.randint(self.MIN_FREQ, self.MAX_FREQ + 1)

            signalL, signalR, _, _ = self.generate_signals(azi, amp, freq)
            meansL.append(np.mean(signalL))
            meansR.append(np.mean(signalR))
            varsL.append(np.var(signalL))
            varsR.append(np.var(signalR))

        # We can only take mean of means because they are all equally long
        meanL = np.mean(meansL)
        meanR = np.mean(meansR)
        mean_varL = np.mean(varsL)
        mean_varR = np.mean(varsR)

        # Because signals are symmetrical due to mic positions, 
        # we can average the means and variances of both signals
        mean = (meanL + meanR)/2
        # avg_std = sqrt(avg_variance)
        std = np.sqrt((mean_varL + mean_varR)/2)
        return mean, std

    def generate_signals(self, azi, amp, freq):
        sin_wave = amp * np.sin(2 * np.pi * np.arange(self.sample_rate * self.time) * (freq / self.sample_rate))
        ssX, ssY = self.room_simulation.set_sound_source(azi, sin_wave)
        self.room_simulation.simulate_sound_propagation()
        signalL, signalR = self.room_simulation.get_propagated_signals()

        # Make sure the signals have the same length as the others
        if len(signalL) < self.MIN_LENGTH:
            signalL = np.append(signalL, np.zeros(self.MIN_LENGTH-len(signalL)))
        if len(signalR) < self.MIN_LENGTH:
            signalR = np.append(signalR, np.zeros(self.MIN_LENGTH-len(signalR)))

        return signalL, signalR, ssX, ssY

    def gather_signals(self):
        signalsL = []
        signalsR = []
        for azi in np.linspace(0,2*np.pi,360,endpoint=False):
            amp = 500
            freq = self.MAX_FREQUENCY

            signalL, signalR, _, _ = self.generate_signals(azi, amp, freq)
            signalsL.append(signalL)
            signalsR.append(signalR)
        return signalsL, signalsR

    def compute_error_matrix(self):
        signalsL, signalsR = self.gather_signals()
        error_matrix = np.zeros((len(signalsL), len(signalsL)))

        for i in range(len(signalsL)):
            for j in range(i,len(signalsL)):
                normErrorL = self.normalised_error(signalsL[i], signalsL[j])
                normErrorR = self.normalised_error(signalsR[i], signalsR[j])
                error_matrix[i,j] = normErrorL + normErrorR
                error_matrix[j,i] = error_matrix[i,j]
                
        return error_matrix

    def normalised_error(self, signal1, signal2):
        return np.sum(np.square(signal1 - signal2))/np.sum(np.square(signal1))
