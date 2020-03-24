import numpy as np
import math
import pyroomacoustics as pra
import matplotlib.pyplot as plt

class Simulation():
    def __init__(self, fs, radius, absorption, mic_distance_left, mic_distance_right):
        self.fs = fs # Sample rate
        # Convert to meters because pyroomacoustics uses meters
        self.radius = radius / 100
        self.mic_distance_left = (mic_distance_left[0]/100, mic_distance_left[1]/100)
        self.mic_distance_right = (mic_distance_right[0]/100, mic_distance_right[1]/100)
        self.absorption = absorption
        self.room = self.get_room()

    def get_room(self):
        #Create square room
        room = pra.ShoeBox([self.radius*2, self.radius*2],
                           fs=self.fs, absorption=self.absorption)

        #Mic initiliasation
        mics = np.c_[
            [self.radius-self.mic_distance_left[0], self.radius-self.mic_distance_left[1]],
            [self.radius+self.mic_distance_right[0], self.radius-self.mic_distance_right[1]]
                    ]
        mic_array = pra.MicrophoneArray(mics, self.fs)
        room.add_microphone_array(mic_array)

        return room

    def set_sound_source(self, azimuth, source_signal):
        # Get x and y coordinate of sound source given their angle
        ssX, ssY = self.get_circumference_pos(self.radius, 0, self.radius, azimuth)
        ssX = self.radius*2 - ssX  # Invert x coordinates since we work counter-clockwise
        ssY = self.radius*2 - ssY  # Invert y coordinates for some reason...
        self.room.sources = [pra.SoundSource([ssX, ssY], signal=source_signal)]
        return ssX, ssY

    def get_circumference_pos(self, x1, y1, r, theta):
        """Compute point (x2, y2) with angle theta on circumference of circle relative with radius r to a point (x1, y1)"""
        x2 = x1 + r * math.sin(theta)
        y2 = y1 + r * (1 - math.cos(theta))
        return x2, y2

    def get_circumference_angle(self, x1, y1, x2, y2, r):
        """Compute angle theta of point (x2, y2) on circumference of circle with radius r relative to a point (x1, y1)"""
        theta = math.degrees(math.acos((r - y2 + y1)/r))
        return np.round(theta)

    def plot_room(self, plot_sources = False):
        # Initialise sizes of different attributes
        mic_source_size = self.radius/40
        source_at_90_size = self.radius/20
        sound_source_circle_w = 1 if self.radius <= 159 else self.radius/160

        fig, ax = plt.subplots()

        # Draw ellipse where sources are placed on and microphones
        micL = plt.Circle(self.room.mic_array.R[:,0], mic_source_size, color='r')
        micR = plt.Circle(self.room.mic_array.R[:,1], mic_source_size, color='r')
        srcCircle = plt.Circle((self.radius, self.radius), self.radius, fill=False, linewidth=sound_source_circle_w, edgecolor = "blue")
        centerCircle = plt.Circle((self.radius, self.radius), mic_source_size, color = "yellow")
        ax.add_artist(centerCircle)
        ax.add_artist(srcCircle)
        ax.add_artist(micL)
        ax.add_artist(micR)

        if plot_sources:
            # Draw all possible sound sources
            blue_values = np.linspace(0, 1, 360)
            for azi, i in enumerate(np.linspace(0,2*np.pi,360)):
                sourceX, sourceY = self.get_circumference_pos(self.radius, 0, self.radius, azi)
                sourceX = self.radius*2 - sourceX # Invert x coordinates since we work counter-clockwise
                sourceY = self.radius*2 - sourceY # Invert y coordinates for some reason...
                # To indicate what is (counter-)clockwise
                if azi == 90:
                    ax.add_artist(plt.Circle((sourceX, sourceY), source_at_90_size, color=(blue_values[i], 0.5, 0.5)))
                else:
                    ax.add_artist(plt.Circle((sourceX, sourceY), mic_source_size, color=(blue_values[i], 0.5, 0.5)))

        # Set room boundaries
        ax.set_xlim(-self.radius/10, self.radius * 2 + self.radius/10)
        ax.set_ylim(-self.radius/10, self.radius * 2 + self.radius/10)

        plt.show()

    def simulate_sound_propagation(self):
        self.room.image_source_model() # Ensure proper sound propagation
        self.room.simulate(recompute_rir=True)

    def get_propagated_signals(self):
        signalL = self.room.mic_array.signals[0,:]
        signalR = self.room.mic_array.signals[1,:]
        return signalL, signalR

    def get_sample_rate(self):
        return self.fs
