import mat73
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import mne
from scipy import signal

#%% Read Data
mat_data = mat73.loadmat("H:/bravo1/20221107/GangulyServer/Robot3DArrow/103555/BCI_Fixed/Data0001.mat")

#%% Read Broadband Data
BroadBandData = mat_data['TrialData']['BroadbandData']
BroadBandData = np.concatenate(BroadBandData, axis = 0)
ChMap = mat_data['TrialData']['Params']['ChMap']

#%% predefine
mapSize = ChMap.shape
reMap = np.zeros(128)
reMap[ChMap.astype(int).flatten() - 1] = BroadBandData[0,:]
reMap2D = reMap.reshape(mapSize)

#%% see Data
plt.plot(BroadBandData[:,0])
plt.show()

#%% show matrix as image
plt.imshow(reMap2D, interpolation='nearest')
plt.show()

#%% Plot PSD
freqs, psd = signal.welch(BroadBandData[:,0], 1000, nperseg=2048)
plt.plot(freqs[0:300], psd[0:300])
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()
plt.show()