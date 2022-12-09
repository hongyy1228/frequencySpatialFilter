import numpy as np
from scipy.linalg import eig
from scipy import signal

def run_ssd(BroadBandData, peak, band_width):
    """Wrapper for compute_ssd with standard settings for definining filters.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance with signals to be spatially filtered.
    peak : float
        Peak frequency of the desired signal contribution.
    band_width : float
        Spectral bandwidth for the desired signal contribution.

    Returns
    -------
    filters : array, 2-D
        Spatial filters as computed by SSD, each column = 1 spatial filter.
    patterns : array, 2-D
        Spatial patterns, with each pattern being a column vector.
    """

    signal_bp = [peak - band_width, peak + band_width]
    noise_bp = [peak - (band_width + 2), peak + (band_width + 2)]
    noise_bs = [peak - (band_width + 1), peak + (band_width + 1)]

    filters, patterns = compute_ssd(BroadBandData, signal_bp, noise_bp, noise_bs)

    return filters, patterns


def compute_ssd(BroadBandData, signal_bp, noise_bp, noise_bs):
    """Compute SSD for a specific peak frequency.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance with signals to be spatially filtered.
    signal_bp : tuple
        Pass-band for defining the signal contribution. E.g. (8, 13)
    noise_bp : tuple
        Pass-band for defining the noise contribution.
    noise_bs : tuple
        Stop-band for defining the noise contribution.


    Returns
    -------
    filters : array, 2-D
        Spatial filters as computed by SSD, each column = 1 spatial filter.
    patterns : array, 2-D
        Spatial patterns, with each pattern being a column vector.
    """

    iir_params = dict(order=2, ftype="butter", output="sos")



    # bandpass filter for signal
    sfreq = 1000
    iir_params = dict(order=2, ftype='butter', output='sos')
    l_freq = signal_bp[0],
    h_freq = signal_bp[1],

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

# Bandpass, l_stop, h_stop = l_freq, h_freq
    raw_signal = signal.sosfilt(iir_params['sos'], BroadBandData)

    # bandpass filter
    l_freq = signal_bp[0],
    h_freq = signal_bp[1],

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
    raw_noise = raw.copy().filter(
        l_freq=noise_bp[0],
        h_freq=noise_bp[1],
        method="iir",
        iir_params=iir_params,
        verbose=False,
    )

    # bandstop filter
    raw_noise = raw_noise.filter(
        l_freq=noise_bs[1],
        h_freq=noise_bs[0],
        method="iir",
        iir_params=iir_params,
        verbose=False,
    )

    # compute covariance matrices for signal and noise contributions

    cov_signal = np.cov(raw_signal._data)
    cov_noise = np.cov(raw_noise._data)

    # compute spatial filters
    filters = compute_ged(cov_signal, cov_noise)

    # compute spatial patterns
    patterns = compute_patterns(cov_signal, filters)

    return filters, patterns

def compute_ged(cov_signal, cov_noise):
    """Compute a generatlized eigenvalue decomposition maximizing principal
    directions spanned by the signal contribution while minimizing directions
    spanned by the noise contribution.

    Parameters
    ----------
    cov_signal : array, 2-D
        Covariance matrix of the signal contribution.
    cov_noise : array, 2-D
        Covariance matrix of the noise contribution.

    Returns
    -------
    filters : array
        SSD spatial filter matrix, columns are individual filters.

    """

    nr_channels = cov_signal.shape[0]

    # check for rank-deficiency
    [lambda_val, filters] = eig(cov_signal)
    idx = np.argsort(lambda_val)[::-1]
    filters = np.real(filters[:, idx])
    lambda_val = np.real(lambda_val[idx])
    tol = lambda_val[0] * 1e-6
    r = np.sum(lambda_val > tol)

    # if rank smaller than nr_channels make expansion
    if r < nr_channels:
        print("Warning: Input data is not full rank")
        M = np.matmul(filters[:, :r], np.diag(lambda_val[:r] ** -0.5))
    else:
        M = np.diag(np.ones((nr_channels,)))

    cov_signal_ex = (M.T @ cov_signal) @ M
    cov_noise_ex = (M.T @ cov_noise) @ M

    [lambda_val, filters] = eig(cov_signal_ex, cov_signal_ex + cov_noise_ex)

    # eigenvalues should be sorted by size already, but double checking
    idx = np.argsort(lambda_val)[::-1]
    filters = filters[:, idx]
    filters = np.matmul(M, filters)

    return filters


def apply_filters(raw, filters, prefix="ssd"):
    """Apply spatial filters on continuous data.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance with signals to be spatially filtered.
    filters : array, 2-D
        Spatial filters as computed by SSD.
    prefix : string | None
        Prefix for renaming channels for disambiguation. If None: "ssd"
        is used.

    Returns
    -------
    raw_projected : instance of Raw
        Raw instance with projected signals as traces.
    """

    raw_projected = raw.copy()
    components = filters.T @ raw.get_data()
    nr_components = filters.shape[1]
    raw_projected._data = components

    ssd_channels = ["%s%i" % (prefix, i + 1) for i in range(nr_components)]
    mapping = dict(zip(raw.info["ch_names"], ssd_channels))
    mne.channels.rename_channels(raw_projected.info, mapping)
    raw_projected.drop_channels(raw_projected.info["ch_names"][nr_components:])

    return raw_projected


def compute_patterns(cov_signal, filters):
    """Compute spatial patterns for a specific covariance matrix.

    Parameters
    ----------
    cov_signal : array, 2-D
        Covariance matrix of the signal contribution.
    filters : array, 2-D
        Spatial filters as computed by SSD.
    Returns
    -------
    patterns : array, 2-D
        Spatial patterns.
    """

    top = cov_signal @ filters
    bottom = (filters.T @ cov_signal) @ filters
    patterns = top @ np.linalg.pinv(bottom)

    return patterns