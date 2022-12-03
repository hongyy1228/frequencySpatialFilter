import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# Seed the random number generator
np.random.seed(0)

time_step = .01
time_vec = np.arange(0, 70, time_step)

# A signal with a small frequency chirp
sig = np.sin(0.5 * np.pi * time_vec * (1 + .1 * time_vec))

plt.plot(time_vec, sig)
plt.show()

freqs, psd = signal.welch(sig)

plt.figure(figsize=(5, 4))
plt.plot(freqs, psd)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.show()
