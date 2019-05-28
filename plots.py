import numpy as np
import matplotlib.pyplot as plt
import math

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

def heatmap2d(arr, title="", xlabel="", ylabel="", xticks=None, xlabels=None, yticks=None, ylabels=None):
    img = plt.imshow(arr, cmap='inferno')
    im_ratio = arr.shape[0]/arr.shape[1]
    plt.colorbar(img, fraction=0.046*im_ratio, pad=0.04)
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

xlabel = r'Angles in radians'
ylabel = xlabel
xticks = np.arange(0, 360+1, 90)
yticks = xticks
xlabels = radians[:len(xticks)*2][::2]
ylabels = xlabels
heatmap2d(radialValues, 
          r'Heatmap of Equation 3',
          xlabel,
          ylabel,
          xticks,
          xlabels,
          yticks,
          ylabels)

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
          ylabels)

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
          ylabels)

#%%