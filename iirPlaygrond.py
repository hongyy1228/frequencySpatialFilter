import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import mat73

#%% Load Data
mat_data = mat73.loadmat('/Users/hongyy/Documents/B1/20221025/GangulyServer/20221025/Robot3DArrow/100256/BCI_Fixed/Data0001.mat')

#%% Read Broadband Data
BroadBandData = mat_data['TrialData']['BroadbandData']
BroadBandData = np.concatenate(BroadBandData, axis = 0)
ChMap = mat_data['TrialData']['Params']['ChMap']

#%% predefine
mapSize = ChMap.shape
reMap = np.zeros(128)
reMap[ChMap.astype(int).flatten() - 1] = BroadBandData[0,:]
reMap2D = reMap.reshape(mapSize)

#%% design filter
