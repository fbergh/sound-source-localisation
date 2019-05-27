import numpy as np
import math
import pyroomacoustics as pra
import matplotlib.pyplot as plt

class Simulation():
    def __init__(self, fs, radius, absorption, micDistL, micDistR, nrMics):
        self.fs = fs
        # Convert to meters because pyroomacoustics uses meters
        self.radius = radius / 100
        self.micDistL = (micDistL[0]/100, micDistL[1]/100)
        self.micDistR = (micDistR[0]/100, micDistR[1]/100)
        self.nrMics = nrMics
        self.absorption = absorption
        self.room = self.getRoom()

    def getRoom(self):
        #Create room with no reverb (absorption = 1.0)
        room = pra.ShoeBox([self.radius*2, self.radius*2],
                           fs=self.fs, absorption=self.absorption)

        #Mic initiliasation
        mics = np.c_[
            [self.radius-self.micDistL[0], self.radius-self.micDistL[1]],
            [self.radius+self.micDistR[0], self.radius-self.micDistR[1]]
                    ]
        mic_array = pra.MicrophoneArray(mics, self.fs)
        room.add_microphone_array(mic_array)

        return room

    def setSoundSource(self, azimuth, sourceSignal):
        ssX, ssY = self.calcCircumPos(self.radius, 0, self.radius, azimuth)
        ssX = self.radius*2 - ssX  # Invert x coordinates since we work counter-clockwise
        ssY = self.radius*2 - ssY  # Invert y coordinates for some reason...
        self.room.sources = [pra.SoundSource([ssX, ssY], signal=sourceSignal)]
        return ssX, ssY

    def calcCircumPos(self, x1, y1, r, theta):
        """Compute point (x2, y2) with angle theta on circumference of circle relative with radius r to a point (x1, y1)"""
        x2 = x1 + r * math.sin(theta)
        y2 = y1 + r * (1 - math.cos(theta))
        return x2, y2

    def calcCircumAngle(self, x1, y1, x2, y2, r):
        """Compute angle theta of point (x2, y2) on circumference of circle with radius r relative to a point (x1, y1)"""
        theta = math.degrees(math.acos((r - y2 + y1)/r))
        return np.round(theta)

    def plotRoom(self, plotSources = False):
        # Initialise sizes of different attributes
        micAndSrcSize = self.radius/40
        src90 = self.radius/20
        srcCircWidth = 1 if 160 > self.radius else self.radius/160

        fig, ax = plt.subplots()

        # Draw ellipse where sources are placed on and microphones
        micL = plt.Circle(self.room.mic_array.R[:,0], micAndSrcSize, color='r')
        micR = plt.Circle(self.room.mic_array.R[:,1], micAndSrcSize, color='r')
        srcCircle = plt.Circle((self.radius, self.radius), self.radius, fill=False, linewidth=srcCircWidth, edgecolor = "blue")
        centerCircle = plt.Circle((self.radius, self.radius), micAndSrcSize, color = "yellow")
        ax.add_artist(centerCircle)
        ax.add_artist(srcCircle)
        ax.add_artist(micL)
        ax.add_artist(micR)

        if plotSources:
            # Draw all possible sound sources
            blueValues = np.linspace(0, 1, 360)
            for azi, i in enumerate(np.linspace(0,2*np.pi,360)):
                sourceX, sourceY = self.calcCircumPos(self.radius, 0, self.radius, azi)
                # Invert x coordinates since we work counter-clockwise
                sourceX = self.radius*2 - sourceX
                # Invert y coordinates for some reason...
                sourceY = self.radius*2 - sourceY
                if azi == 90:
                    ax.add_artist(plt.Circle((sourceX, sourceY), src90, color=(blueValues[i], 0.5, 0.5)))
                else:
                    ax.add_artist(plt.Circle((sourceX, sourceY), micAndSrcSize, color=(blueValues[i], 0.5, 0.5)))

        # Set room boundaries
        ax.set_xlim(-self.radius/10, self.radius * 2 + self.radius/10)
        ax.set_ylim(-self.radius/10, self.radius * 2 + self.radius/10)

        plt.show()

    def simulateSoundPropagation(self):
        self.room.image_source_model()
        self.room.simulate(recompute_rir=True)

    def getPropagatedSignals(self):
        signalL = self.room.mic_array.signals[0,:]
        signalR = self.room.mic_array.signals[1,:]
        return signalL, signalR

    def getSampleRate(self):
        return self.fs
