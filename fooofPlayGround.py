# Import the FOOOF object
from fooof import FOOOF
import mat73
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# Import a utility to download and load example data
from fooof.utils.download import load_fooof_data

#%% Download example data files needed for this example
freqs = load_fooof_data('freqs.npy', folder='data')
spectrum = load_fooof_data('spectrum.npy', folder='data')

#%%

# Initialize a FOOOF object
fm = FOOOF()

# Set the frequency range to fit the model
freq_range = [2, 40]

# Report: fit the model, print the resulting parameters, and plot the reconstruction
fm.report(freqs, spectrum, freq_range)
plt.show()

# Alternatively, just fit the model with FOOOF.fit() (without printing anything)


# After fitting, plotting and parameter fitting can be called independently:
# fm.print_results()
# fm.plot()



#%%
# for mac
mat_data = mat73.loadmat('/Users/hongyy/Documents/B1/20221025/GangulyServer/20221025/Robot3DArrow/100256/BCI_Fixed/Data0001.mat')

#%%
BroadBandData = mat_data['TrialData']['BroadbandData']
BroadBandData = np.concatenate(BroadBandData, axis = 0)
ChMap = mat_data['TrialData']['Params']['ChMap']
#%% predefine
mapSize = ChMap.shape
reMap = np.zeros(128)
reMap[ChMap.astype(int).flatten() - 1] = BroadBandData[0,:]
reMap2D = reMap.reshape(mapSize)

#%%
freqs_mat, psd_mat = signal.welch(BroadBandData[:,0], 1000, nperseg=1048)
plt.plot(freqs_mat, psd_mat)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()
plt.xlim([0,40])
plt.show()

#%%
# Set the frequency range to fit the model
freq_range = [0.5, 50]

# Report: fit the model, print the resulting parameters, and plot the reconstruction
fm.report(freqs_mat, psd_mat, freq_range)
plt.show()