import numpy as np
import matplotlib.pyplot as plt
import math
from room_simulation import Simulation
from sine_dataset import SineData

radians = [
           r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', 
           r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', 
           r'$2\pi$', r'$\frac{9\pi}{4}$', r'$\frac{5\pi}{2}$', r'$\frac{11\pi}{4}$', 
           r'$3\pi$', r'$\frac{13\pi}{4}$', r'$\frac{7\pi}{2}$', r'$\frac{15\pi}{4}$', 
           r'$4\pi$', r'$\frac{17\pi}{4}$', r'$\frac{9\pi}{2}$', r'$\frac{19\pi}{4}$',
           r'$5\pi$', r'$\frac{21\pi}{4}$', r'$\frac{11\pi}{2}$', r'$\frac{23\pi}{4}$',
           r'$6\pi$', r'$\frac{25\pi}{4}$', r'$\frac{13\pi}{2}$', r'$\frac{27\pi}{4}$',
           r'$7\pi$', r'$\frac{29\pi}{4}$', r'$\frac{15\pi}{2}$', r'$\frac{31\pi}{4}$',
           r'$8\pi$', r'$\frac{33\pi}{4}$', r'$\frac{17\pi}{2}$', r'$\frac{35\pi}{4}$',
           r'$9\pi$', r'$\frac{37\pi}{4}$', r'$\frac{19\pi}{2}$', r'$\frac{39\pi}{4}$',
           r'$10\pi$', r'$\frac{41\pi}{4}$', r'$\frac{21\pi}{2}$', r'$\frac{43\pi}{4}$',
           r'$11\pi$'
          ]

# Define custom loss function
def radial(pred, target):
    return abs(math.cos(pred-target) - math.cos(0))

def penalty(pred):
    return max(0, pred - math.radians(360)) + max(0, -pred)

def cossLoss(pred, target):
    return radial(pred, target) + penalty(pred)

def heatmap2d(arr, title="", xlabel="", ylabel="", xticks=None, xlabels=None, yticks=None, ylabels=None, cbar_label=""):
    plt.figure(figsize=(10,5))
    img = plt.imshow(arr, cmap='inferno')
    im_ratio = arr.shape[0]/arr.shape[1]
    cbar = plt.colorbar(img, fraction=0.046*im_ratio, pad=0.04)
    cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=15)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks, xlabels)
    plt.yticks(yticks, ylabels)
    plt.show()

# Plot radial
radialValues = np.zeros((360+1, 360+1))
for theta in range(360+1):
    for phi in range(theta, 360+1):
        radialLoss = radial(math.radians(theta), math.radians(phi))
        radialValues[theta, phi] = radialLoss
        radialValues[phi, theta] = radialLoss

xlabel = r'Predicted angles [radians]'
ylabel = r'Actual angles [radians]'
xticks = np.arange(0, 360+1, 90)
yticks = xticks
xlabels = radians[:len(xticks)*2][::2]
ylabels = xlabels
cbar_label = "Loss value [-]"
heatmap2d(radialValues, 
          r'Heatmap of Equation 3',
          xlabel,
          ylabel,
          xticks,
          xlabels,
          yticks,
          ylabels,
          cbar_label
          )
plt.savefig("../Images/heatmap_eq3.png")

# Plot penalty
penaltyValues = np.zeros((360+1, 1080+1))
for theta in range(360+1):
    for j in range(1080+1):
        penaltyLoss = penalty(math.radians(j))
        penaltyValues[theta, j] = penaltyLoss

xticks = np.arange(0, 1080+1, 90)
xlabels = radians[:len(xticks)*2][::2]
heatmap2d(penaltyValues,
          r'Heatmap of Equation 4',
          xlabel,
          ylabel,
          xticks,
          xlabels,
          yticks,
          ylabels,
          cbar_label
          )
plt.savefig("../Images/heatmap_eq4.png")

# Plot cosine loss
cosineValues = np.append(radialValues, 
                         np.append(radialValues + penaltyValues[:,360:721], radialValues + penaltyValues[:,720:], axis = 1), 
                         axis = 1)
heatmap2d(cosineValues,
          r'Heatmap of Equation 5',
          xlabel,
          ylabel,
          xticks,
          xlabels,
          yticks,
          ylabels,
          cbar_label
          )
plt.savefig("../Images/heatmap_eq5.png")

RAD = 50
MIC_L_DIST = (11, -10)
MIC_R_DIST = (11, -10)
ABSORPTION = 0.0
MIN_FREQ = 20
MAX_FREQ = 20000
SAMPLE_RATE = int(MAX_FREQ*2.2)
TIME = 1                                            
MIN_LENGTH = 48000

roomSim = Simulation(SAMPLE_RATE, RAD, ABSORPTION, MIC_L_DIST, MIC_R_DIST, 2)
sineDataset = SineData(1, roomSim, TIME, MIN_LENGTH, MIN_FREQ, MAX_FREQ)

amp = 1
freq = 20
azi = np.pi/2

signalMic1, signalMic2, _, _ = sineDataset.generateSignals(azi, amp, freq)

plt.plot(signalMic1)
plt.title("A (standardised) sine wave with amplitude of one and a frequency of 20Hz")
plt.ylabel("Amplitude [z-score]")
plt.xlabel("Samples (1s = 44000 data points) [-]")
plt.grid(True)
plt.show()
plt.savefig("../Images/input2_nn_arch.png")