import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import mat73
import ssd
from functools import partial
from joblib import Parallel, delayed

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

#%% ssd calculation
bin_width = 0.5
peak = 8.82
nr_components = 3

signal_bp = [peak - bin_width, peak + bin_width]
noise_bp = [peak - (bin_width + 2), peak + (bin_width + 2)]
noise_bs = [peak - (bin_width + 1), peak + (bin_width + 1)]

#filters, patterns = ssd.run_ssd(BroadBandData, peak, bin_width)



#%% design filter
sfreq = 1000
iir_params = dict(order = 2, ftype = 'butter', output = 'sos')
l_freq=signal_bp[0],
h_freq=signal_bp[1],

kind = 'bandstop'
ftype = 'butter'
output = iir_params.get('output', 'sos')
l_stop, h_stop = l_freq, h_freq
f_pass = [l_freq, h_freq]
f_stop = [l_freq, h_freq]

btype = 'bandpass'
f_pass = np.atleast_1d(f_pass)
Wp = f_pass / (float(sfreq) / 2)
output = 'sos'

kwargs = dict(N=iir_params['order'], Wn=Wp, btype=btype,
                          ftype=ftype, output=output)
for key in ('rp', 'rs'):
    if key in iir_params:
        kwargs[key] = iir_params[key]
system = signal.iirfilter(**kwargs)
cutoffs = signal.sosfreqz(system, worN=Wp * np.pi)[1]

max_try = 100000

kind = 'sos'
sos = system
zi = [[0.] * 2] * len(sos)

n_per_chunk = 1000
n_chunks_max = int(np.ceil(max_try / float(n_per_chunk)))
x = np.zeros(n_per_chunk)
x[0] = 1
last_good = n_per_chunk
thresh_val = 0

for ii in range(n_chunks_max):
    h, zi = signal.sosfilt(sos, x, zi=zi)
    x[0] = 0  # for subsequent iterations we want zero input
    h = np.abs(h)
    thresh_val = max(0.001 * np.max(h), thresh_val)
    idx = np.where(np.abs(h) > thresh_val)[0]
    if len(idx) > 0:
        last_good = idx[-1]
    else:  # this iteration had no sufficiently lange values
        idx = (ii - 1) * n_per_chunk + last_good
        break

padlen = idx
iir_params.update(dict(padlen=padlen))
iir_params.update(sos=system)
# either -6 dB or -12 dB for dB_cutoff

#%% Start filtering
padlen = min(iir_params['padlen'], x.shape[-1] - 1)
fun = partial(signal.sosfiltfilt, sos=iir_params['sos'], padlen=padlen,
                      axis=-1)

y_sos = signal.sosfilt(iir_params['sos'],BroadBandData)


#%% Covariance
observation_data = [y_sos[:,56],y_sos[:,55],y_sos[:,48],y_sos[:,52]]
cov_signal = np.cov(observation_data)
#%% Docu_testing
sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
x = signal.unit_impulse(700)
y_sos = signal.sosfilt(sos, x)
plt.plot(y_sos, 'k', label='SOS')
plt.legend(loc='best')
plt.show()


#%% Docu method testing

x = BroadBandData[:,0]
y_sos = signal.sosfilt(iir_params['sos'],BroadBandData)
plt.plot(y_sos, 'k', label='SOS')
plt.legend(loc='best')
plt.show()