import mat73
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy import signal
import fooof

#%% Read Data
#mat_data = mat73.loadmat("H:/bravo1/20221107/GangulyServer/Robot3DArrow/103555/BCI_Fixed/Data0001.mat")
# for mac
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

#%% see Data
plt.plot(BroadBandData[:,0])
plt.show()

#%% show matrix as image
plt.imshow(reMap2D, interpolation='nearest')
plt.show()

#%% Plot PSD
freqs, psd = signal.welch(BroadBandData[:,0], 1000, nperseg=2048)
plt.plot(freqs, psd)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()
plt.xlim([0,40])
plt.show()

#%% fitting fooof, fitting oscilations and 1/f
fm = fooof.FOOOFGroup()
fm.fit(freqs, psd)