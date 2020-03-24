# Sound Source Localisation in a Simulated Environment using Convolutional Neural Networks

## Abstract

Sound source localisation is a multifaceted problem where the location of sound sources is retrieved with the signals received at microphones. Many different variables complicate this problem, such as the environment, the number of microphones and the distance between sources and microphones. Contemporary neural network approaches use several pre-processing techniques on the signals before they are given to the network. In this presentation, I will demonstrate that the problem can also be approached without pre-processing the signals. Specifically, I will explore the effect of different distances between sound sources and microphones, and different location representations on the accuracy of sound source localisation. Moreover, I will explore whether the neural network learns to use the same cues, for sound source localization, as humans.

## Description

This is the repository of my bachelor thesis called "Sound Source Localisation in a a Simulated Environment using Convolutional Neural Networks" (2018-2019). The code is based on Python 3 and published under the 2-clause BSD license. To install the required packages run:
```
python3 -m pip install -r requirements.txt
```

## Structure

- `logger.py`: contains the `Logger` class to log the gradients and loss of a network during training, adapted from [this repository](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard>).
- `network.py`: contains the PyTorch-based network classes for the baseline and main neural networks.
- `plots.py`: script to create part of the figures in the thesis.
- `room_simulation.py`: contains the `Simulation` class for simulating a square room to perform sound source localisation in.
- `sine_dataset.py`: contains the `SineDataset` class for creating a PyTorch dataset with sine waves which are used as input for the networks.
- `test_nets.py`: script to run the experiments performed in the thesis.
- `train_nets.ipynb`: Jupyter notebook for training the networks that are tested in `test_nets.py`.