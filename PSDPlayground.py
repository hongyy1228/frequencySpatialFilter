import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, compute_source_psd

print(__doc__)

data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_raw.fif'
fname_inv = meg_path / 'sample_audvis-meg-oct-6-meg-inv.fif'
fname_label = meg_path / 'labels' / 'Aud-lh.label'

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname, verbose=False)
events = mne.find_events(raw, stim_channel='STI 014')
inverse_operator = read_inverse_operator(fname_inv)
raw.info['bads'] = ['MEG 2443', 'EEG 053']

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True,
                       stim=False, exclude='bads')

tmin, tmax = 0, 120  # use the first 120s of data
fmin, fmax = 1, 150  # look at frequencies between 4 and 100Hz
n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2
label = mne.read_label(fname_label)

stc = compute_source_psd(raw, inverse_operator, lambda2=1. / 9., method="dSPM",
                         tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
                         pick_ori="normal", n_fft=n_fft, label=label,
                         dB=True)

stc.save('psd_dSPM', overwrite=True)

plt.plot(stc.times, stc.data.T)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB)')
plt.title('Source Power Spectrum (PSD)')
plt.show()