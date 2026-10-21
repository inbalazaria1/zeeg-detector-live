# Cell 15 of spm_extraction_annotated.ipynb, code unchanged.  pipeline.py runs
# it with np, pd and sp_stats already defined.
#
# extract_epochs_features_mne: the features of every 250 ms epoch of one channel.
# raw_chan_to_feat: the same, starting from an mne.Raw (not used by the build).
from mne_features.univariate import get_univariate_funcs, compute_pow_freq_bands
from mne_features.feature_extraction import extract_features
import mne_features.univariate

window_size = 250  # samples per epoch, 250 ms at 1000 Hz


def extract_epochs_features_mne(epochs, subj, sr):
    """
    Features of each epoch, from mne_features.

    epochs: (n_epochs, n_samples_per_epoch); subj goes in the 'subj' column;
    sr is the sampling rate in Hz.  One row per epoch.
    """
    data = np.array(epochs)

    # one value per epoch
    ptp_amp = mne_features.univariate.compute_ptp_amp(data)
    hjorth_mobility = mne_features.univariate.compute_hjorth_mobility(data)
    hjorth_complexity = mne_features.univariate.compute_hjorth_complexity(data)
    samp_entropy = mne_features.univariate.compute_samp_entropy(data, emb=2, metric='chebyshev')
    compute_spect_slope = mne_features.univariate.compute_spect_slope(
        sr, data, fmin=0.1, fmax=50, with_intercept=True, psd_method='welch', psd_params=None
    )

    # frequency bands in Hz
    bands_dict = {'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 16), 'beta': (16, 30), 'gamma': (30, 100),
                  'fast': (100, 300)}

    energy_freq_bands = mne_features.univariate.compute_energy_freq_bands(sr, data, freq_bands=bands_dict)

    app_entropy = mne_features.univariate.compute_app_entropy(data, emb=2, metric='chebyshev')

    decorr_time = mne_features.univariate.compute_decorr_time(sr, data)

    higuchi_fd = mne_features.univariate.compute_higuchi_fd(data, kmax=10)

    hjorth_complexity_spect = mne_features.univariate.compute_hjorth_complexity_spect(sr, data, normalize=False,
                                                                                      psd_method='welch',
                                                                                      psd_params=None)

    hjorth_mobility_spect = mne_features.univariate.compute_hjorth_mobility_spect(sr, data, normalize=False,
                                                                                  psd_method='welch', psd_params=None)

    hurst_exp = mne_features.univariate.compute_hurst_exp(data)

    katz_fd = mne_features.univariate.compute_katz_fd(data)

    kurtosis = mne_features.univariate.compute_kurtosis(data)

    line_length = mne_features.univariate.compute_line_length(data)

    mean = mne_features.univariate.compute_mean(data)

    quantile = mne_features.univariate.compute_quantile(data, q=0.75)

    rms = mne_features.univariate.compute_rms(data)

    skewness = mne_features.univariate.compute_skewness(data)

    spect_entropy = mne_features.univariate.compute_spect_entropy(sr, data, psd_method='welch', psd_params=None)

    std = mne_features.univariate.compute_std(data)

    svd_entropy = mne_features.univariate.compute_svd_entropy(data, tau=2, emb=10)

    svd_fisher_info = mne_features.univariate.compute_svd_fisher_info(data, tau=2, emb=10)

    variance = mne_features.univariate.compute_variance(data)

    zero_crossings = mne_features.univariate.compute_zero_crossings(data, threshold=2.220446049250313e-16)

    abspow = mne_features.univariate.compute_pow_freq_bands(sr, data, {'total': (0.1, 500)}, False,
                                                            psd_method='multitaper')

    # the one-value features as columns
    stacked_arrays = np.column_stack(
        (hjorth_mobility, hjorth_complexity, samp_entropy, ptp_amp, app_entropy, decorr_time, higuchi_fd,
         hjorth_complexity_spect, hjorth_mobility_spect, hurst_exp, katz_fd, kurtosis, line_length, mean,
         quantile, rms, skewness, spect_entropy, std, svd_entropy, svd_fisher_info,
         variance, zero_crossings, abspow))

    df = pd.DataFrame(stacked_arrays)

    df.columns = ['hjorth_mobility', 'hjorth_complexity', 'samp_entropy', 'ptp_amp', 'app_entropy', 'decorr_time',
                  'higuchi_fd', 'hjorth_complexity_spect', 'hjorth_mobility_spect', 'hurst_exp', 'katz_fd', 'kurtosis',
                  'line_length', 'mean', 'quantile', 'rms', 'skewness', 'spect_entropy',
                  'std', 'svd_entropy', 'svd_fisher_info', 'variance', 'zero_crossings', 'abspow_']

    # label columns, dropped before the model sees the matrix
    df['subj'] = np.full(df.shape[0], subj)
    df['epoch_id'] = list(range(0, len(epochs)))

    # features that come back flattened, reshaped into one column each
    kaiser = mne_features.univariate.compute_teager_kaiser_energy(data)
    kaiser_df = pd.DataFrame(
        np.array(kaiser).reshape(-1, 12),
        columns=['teager_kaiser_energy_0_mean', 'teager_kaiser_energy_0_std',
                 'teager_kaiser_energy_1_mean', 'teager_kaiser_energy_1_std',
                 'teager_kaiser_energy_2_mean', 'teager_kaiser_energy_2_std',
                 'teager_kaiser_energy_3_mean', 'teager_kaiser_energy_3_std',
                 'teager_kaiser_energy_4_mean', 'teager_kaiser_energy_4_std',
                 'teager_kaiser_energy_5_mean', 'teager_kaiser_energy_5_std']
    )

    pow_freq_bands = mne_features.univariate.compute_pow_freq_bands(data=epochs, sfreq=sr,
                                                                    freq_bands=bands_dict,
                                                                    normalize=True, ratios=None,
                                                                    ratios_triu=False,
                                                                    psd_method='multitaper', log=False,
                                                                    psd_params=None)
    pow_freq_bands_df = pd.DataFrame(np.array(pow_freq_bands).reshape(-1, 6),
                                     columns=['pow_freq_bands_theta', 'pow_freq_bands_alpha', 'pow_freq_bands_sigma',
                                              'pow_freq_bands_beta', 'pow_freq_bands_gamma', 'pow_freq_bands_fast'])

    # every pairwise power ratio (theta/alpha, alpha/theta, ...)
    for band1 in ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']:
        for band2 in ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']:
            if band1 == band2:
                continue
            else:
                pow_freq_bands_df[f'pow_freq_bands_{band1}/{band2}'] = pow_freq_bands_df[f'pow_freq_bands_{band1}'] / \
                                                                       pow_freq_bands_df[f'pow_freq_bands_{band2}']

    # spectral slope: intercept, slope, MSE, R2
    spect_slope = pd.DataFrame(
        np.array(compute_spect_slope).reshape(-1, 4),
        columns=['spect_slope_intercept', 'spect_slope_slope', 'spect_slope_MSE', 'spect_slope_R2']
    )

    # band energies, one column per band
    energy_freq_bands = pd.DataFrame(
        np.array(energy_freq_bands).reshape(-1, 6),
        columns=['energy_freq_bands_theta', 'energy_freq_bands_alpha', 'energy_freq_bands_sigma',
                 'energy_freq_bands_beta', 'energy_freq_bands_gamma', 'energy_freq_bands_fast']
    )
    # two energy ratios
    energy_freq_bands['energy_freq_bands_gf'] = energy_freq_bands['energy_freq_bands_gamma'] / energy_freq_bands[
        'energy_freq_bands_fast']
    energy_freq_bands['energy_freq_bands_bg'] = energy_freq_bands['energy_freq_bands_beta'] / energy_freq_bands[
        'energy_freq_bands_gamma']

    wavelet_coef_energy = mne_features.univariate.compute_wavelet_coef_energy(data, wavelet_name='db4')
    wavelet_coef_energy_df = pd.DataFrame(
        np.array(wavelet_coef_energy).reshape(-1, 5),
        columns=['wavelet_coef_energy_0', 'wavelet_coef_energy_1', 'wavelet_coef_energy_2', 'wavelet_coef_energy_3',
                 'wavelet_coef_energy_4'])

    subj_features = pd.concat(
        [df, spect_slope, energy_freq_bands, kaiser_df, pow_freq_bands_df, wavelet_coef_energy_df], axis=1)

    # every pairwise energy ratio, named by first letters (e.g. 'ab')
    for band1 in ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']:
        for band2 in ['theta', 'alpha', 'sigma', 'beta', 'gamma', 'fast']:
            if band1 == band2:
                continue
            else:
                subj_features[f'energy_freq_bands_{band1[0]}{band2[0]}'] = subj_features[f'energy_freq_bands_{band1}'] / \
                                                                           subj_features[f'energy_freq_bands_{band2}']
    # the columns and order model 1 was trained on
    subj_features = subj_features[
        ['subj', 'epoch_id', 'app_entropy', 'decorr_time', 'energy_freq_bands_theta', 'energy_freq_bands_alpha',
         'energy_freq_bands_sigma', 'energy_freq_bands_beta', 'energy_freq_bands_gamma', 'energy_freq_bands_fast',
         'higuchi_fd', 'hjorth_complexity', 'hjorth_complexity_spect', 'hjorth_mobility', 'hjorth_mobility_spect',
         'hurst_exp', 'katz_fd', 'kurtosis', 'line_length', 'mean', 'pow_freq_bands_theta', 'pow_freq_bands_alpha',
         'pow_freq_bands_sigma', 'pow_freq_bands_beta', 'pow_freq_bands_gamma', 'pow_freq_bands_fast',
         'pow_freq_bands_theta/alpha', 'pow_freq_bands_theta/sigma', 'pow_freq_bands_theta/beta',
         'pow_freq_bands_theta/gamma', 'pow_freq_bands_theta/fast', 'pow_freq_bands_alpha/theta',
         'pow_freq_bands_alpha/sigma', 'pow_freq_bands_alpha/beta', 'pow_freq_bands_alpha/gamma',
         'pow_freq_bands_alpha/fast', 'pow_freq_bands_sigma/theta', 'pow_freq_bands_sigma/alpha',
         'pow_freq_bands_sigma/beta', 'pow_freq_bands_sigma/gamma', 'pow_freq_bands_sigma/fast',
         'pow_freq_bands_beta/theta', 'pow_freq_bands_beta/alpha', 'pow_freq_bands_beta/sigma',
         'pow_freq_bands_beta/gamma', 'pow_freq_bands_beta/fast', 'pow_freq_bands_gamma/theta',
         'pow_freq_bands_gamma/alpha', 'pow_freq_bands_gamma/sigma', 'pow_freq_bands_gamma/beta',
         'pow_freq_bands_gamma/fast', 'pow_freq_bands_fast/theta', 'pow_freq_bands_fast/alpha',
         'pow_freq_bands_fast/sigma', 'pow_freq_bands_fast/beta', 'pow_freq_bands_fast/gamma', 'ptp_amp', 'quantile',
         'rms', 'samp_entropy', 'skewness', 'spect_entropy', 'spect_slope_intercept', 'spect_slope_slope',
         'spect_slope_MSE', 'spect_slope_R2', 'std', 'svd_entropy', 'svd_fisher_info', 'teager_kaiser_energy_0_mean',
         'teager_kaiser_energy_0_std', 'teager_kaiser_energy_1_mean', 'teager_kaiser_energy_1_std',
         'teager_kaiser_energy_2_mean', 'teager_kaiser_energy_2_std', 'teager_kaiser_energy_3_mean',
         'teager_kaiser_energy_3_std', 'teager_kaiser_energy_4_mean', 'teager_kaiser_energy_4_std',
         'teager_kaiser_energy_5_mean', 'teager_kaiser_energy_5_std', 'variance', 'wavelet_coef_energy_0',
         'wavelet_coef_energy_1', 'wavelet_coef_energy_2', 'wavelet_coef_energy_3', 'wavelet_coef_energy_4',
         'zero_crossings', 'abspow_', 'energy_freq_bands_ab', 'energy_freq_bands_ag', 'energy_freq_bands_as',
         'energy_freq_bands_af', 'energy_freq_bands_at', 'energy_freq_bands_bt', 'energy_freq_bands_bs',
         'energy_freq_bands_bg', 'energy_freq_bands_bf', 'energy_freq_bands_st', 'energy_freq_bands_sg',
         'energy_freq_bands_sf', 'energy_freq_bands_gt', 'energy_freq_bands_gf', 'energy_freq_bands_ft']]
    feat = subj_features
    return feat


def raw_chan_to_feat(raw, chan, subj):
    """
    Features of one channel of an mne.Raw, the original notebook's way.

    Whole-channel z-score, 250 ms epochs (skipping any with NaN), then
    chan_ptp / chan_skew / chan_kurt over the whole channel on every row.
    Empty DataFrame if `chan` is not in `raw`.
    """
    epochs = []
    if chan not in raw.ch_names:
        return pd.DataFrame()
    chan_raw = raw.copy().pick([chan]).get_data(reject_by_annotation='nan').flatten()
    # z-score over the whole channel
    chan_norm = (chan_raw - np.nanmean(chan_raw)) / np.nanstd(chan_raw)
    # every 250 ms epoch except the last second
    for i in range(0, len(chan_norm) - 4 * window_size, window_size):
        if not np.isnan(chan_norm[i: i + window_size]).any():
            epochs.append(chan_norm[i: i + window_size])
    epochs = np.array(epochs)

    curr_feat = extract_epochs_features_mne(epochs, subj, raw.info['sfreq'])
    clean_chan = [item for sublist in epochs for item in sublist]
    chan_feat = {
        'chan_name': chan,
        'chan_ptp': np.ptp(clean_chan),
        'chan_skew': sp_stats.skew(clean_chan),
        'chan_kurt': sp_stats.kurtosis(clean_chan),
    }

    for feat in chan_feat.keys():
        curr_feat[feat] = chan_feat[feat]

    return curr_feat
