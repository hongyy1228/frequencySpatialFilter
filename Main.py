import mat73
import numpy as np
import matplotlib.pyplot as plt

#%% Read Data
mat_data = mat73.loadmat("H:/bravo1/20221107/GangulyServer/Robot3DArrow/103555/BCI_Fixed/Data0001.mat")

#%% Read Broadband Data
BroadBandData = mat_data['TrialData']['BroadbandData']
BroadBandData = np.concatenate(BroadBandData, axis = 0)
ChMap = mat_data['TrialData']['Params']['ChMap']

#%% predefine
mapSize = ChMap.shape
reMap = np.zeros(mapSize)
reMap[ChMap.astype(int) - 1] = BroadBandData[0,:]

#%% see Data
plt.plot(BroadBandData[:,0])
plt.show()

#%% show matrix as image
