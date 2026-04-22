"""
nn_loading_detection.py

Dataset generation and neural network pipeline for automatic identification
of magnetospheric loading events from MESSENGER magnetometer data.

Each training example is a fixed-length window of (ΔBx, ΔBz) resampled to
a uniform cadence.  The label is 1 if any part of the window overlaps a
human-labelled loading interval, 0 otherwise.

Only orbits that contain at least one loading event are used.

Usage
-----
    python nn_loading_detection.py --build              # build dataset.npz
    python nn_loading_detection.py --build --window 300 --step 30 --hz 1
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from MESSENGER_analysis import (
    load_bowers_data_pkl,
    filter_orbit_segment,
    get_kt17_along_track,
)

# ── paths ─────────────────────────────────────────────────────────────────────
_DIR                = os.path.dirname(os.path.abspath(__file__))
LOADING_LABELS_JSON = os.path.join(_DIR, 'human_loading_labels.json')
DATASET_NPZ         = os.path.join(_DIR, 'nn_dataset.npz')

# ── defaults (all overridable via CLI) ────────────────────────────────────────
DEFAULT_WINDOW_SEC = 2*60   # length of each example window in seconds
DEFAULT_STEP_SEC   = 1    # stride between consecutive window starts
DEFAULT_SAMPLE_HZ  = 0.1     # cadence after resampling (samples per second)

# ── editable hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE    = 12
N_EPOCHS      = 32
TRAIN_RATIO   = 0.75    # fraction of orbits used for training (rest = validation)
LEARNING_RATE = 8e-4
CNN_WIDTH     = 6     # number of filters in each conv layer / nodes in dense layer
DROPOUT       = 0.3     # dropout probability after each conv ReLU and dense ReLU
WEIGHT_DECAY  = 1e-4    # L2 regularisation coefficient for Adam
NOISE_SIGMA   = 0.05    # std of Gaussian noise added to augmented copies (normalised units)
NOISE_COPIES  = 1       # number of noisy duplicates per training window (0 = no augmentation)
PLOT_INTERVAL = 1       # show example diagnostic plot every N epochs (0 = never)

# ── hyperparameter sweep grid (edit these lists, then run --sweep) ─────────────
#SWEEP_WINDOW_SEC = [60, 90, 120, 150, 180]   # window lengths to try (seconds)
#SWEEP_SAMPLE_HZ  = [0.1, 0.2, 1]       # resample rates to try (Hz)
#SWEEP_CNN_WIDTH  = [4, 8, 16, 24]      # CNN filter widths to try

SWEEP_WINDOW_SEC = [90, 600]   # window lengths to try (seconds)
SWEEP_SAMPLE_HZ  = [0.1, 1]       # resample rates to try (Hz)
SWEEP_CNN_WIDTH  = [4, 24]      # CNN filter widths to try

# ── per-orbit processing ──────────────────────────────────────────────────────

def _loading_mask(t_s: np.ndarray, loading_events: list) -> np.ndarray:
    """
    Boolean mask — True where a sample falls inside any loading interval.

    Parameters
    ----------
    t_s            : float64 (N,)  seconds since orbit start (uniform grid)
    loading_events : list of {'start': str, 'stop': str}
    t_epoch        : pd.Timestamp  origin of t_s (= t_s[0] in wall-clock time)
    """
    # attached separately — see process_orbit
    raise NotImplementedError

def process_orbit(orb: int, loading_events: list,
                  window_sec: int, step_sec: int,
                  sample_hz: float) -> dict | None:
    """
    Load one orbit, resample to a uniform grid, slide a window over it, and
    return all (window, label) pairs.

    Parameters
    ----------
    orb            : orbit number
    loading_events : list of {'start': str, 'stop': str} dicts
    window_sec     : window length in seconds
    step_sec       : stride between window starts in seconds
    sample_hz      : samples per second on the uniform grid

    Returns
    -------
    dict with:
        'windows' : float32 (K, 2, W)  — K windows, 2 channels (ΔBx, ΔBz),
                                         W = window_sec * sample_hz samples
        'labels'  : int8    (K,)        — 1 if window overlaps loading, else 0
        'times'   : int64   (K,)        — UTC ns of each window's first sample
        'orbit'   : int
    or None if data is unavailable / too short.
    """
    # ── load & model ──────────────────────────────────────────────────────────
    full_df = load_bowers_data_pkl(orbit_number=orb)
    orb_df  = filter_orbit_segment(full_df)
    if orb_df is None or orb_df.empty:
        return None

    _, Bxm, Bym, Bzm = get_kt17_along_track(df=orb_df)

    t_obs  = pd.to_datetime(orb_df['time'])
    dBx    = (orb_df['magx'].to_numpy(dtype='float64') - Bxm)
    dBz    = (orb_df['magz'].to_numpy(dtype='float64') - Bzm)

    # normalise by model field magnitude so features are dimensionless
    Bmag_mod = np.sqrt(Bxm**2 + Bym**2 + Bzm**2)
    Bmag_mod = np.where(Bmag_mod > 1e-6, Bmag_mod, np.nan)  # avoid /0
    dBx = dBx / Bmag_mod
    dBz = dBz / Bmag_mod

    # ΔBmag/|B_mod|: hemisphere-invariant field strengthening feature
    Bmag_obs = np.sqrt(orb_df['magx'].to_numpy(dtype='float64')**2 +
                       orb_df['magy'].to_numpy(dtype='float64')**2 +
                       orb_df['magz'].to_numpy(dtype='float64')**2)
    dBmag = (Bmag_obs - np.sqrt(Bxm**2 + Bym**2 + Bzm**2)) / Bmag_mod

    # ── loading events as (start_s, stop_s, duration_s) in seconds since orbit start ──
    t_orbit_start = t_obs.iloc[0]
    ev_intervals = []
    for ev in loading_events:
        s0  = (pd.Timestamp(ev['start']) - t_orbit_start).total_seconds()
        s1  = (pd.Timestamp(ev['stop'])  - t_orbit_start).total_seconds()
        dur = s1 - s0
        if dur > 0:
            ev_intervals.append((s0, s1, dur))

    # ── low-pass filter then downsample to sample_hz ─────────────────────────
    # Estimate the native cadence from the median sample interval.
    # Apply a boxcar (uniform) filter of width 1/sample_hz seconds before
    # picking samples, so no aliasing occurs on the downsampled grid.
    t_s_raw    = (t_obs - t_orbit_start).dt.total_seconds().to_numpy()
    dt_native  = float(np.median(np.diff(t_s_raw)))          # seconds/sample
    half_win   = max(1, int(round(0.5 / (sample_hz * dt_native))))  # half-width in native samples

    def _lowpass_interp(arr):
        """Boxcar smooth then interpolate to the uniform grid."""
        from numpy.lib.stride_tricks import sliding_window_view
        win = 2 * half_win + 1
        # reflect-pad to avoid edge shrinkage
        padded   = np.pad(arr, half_win, mode='reflect')
        smoothed = np.convolve(padded, np.ones(win) / win, mode='valid')
        n_out    = int(np.floor(t_s_raw[-1] * sample_hz)) + 1
        t_out    = np.linspace(0.0, t_s_raw[-1], n_out)
        return np.interp(t_out, t_s_raw, smoothed).astype('float32'), t_out

    dBx_u,   t_uniform = _lowpass_interp(dBx)
    dBz_u,   _         = _lowpass_interp(dBz)
    dBmag_u, _         = _lowpass_interp(dBmag)
    n_samples           = len(t_uniform)

    # NaN mask on the uniform grid: True where KT17 was unavailable
    # np.interp propagates NaN from any channel into the resampled arrays
    nan_u = ~np.isfinite(dBx_u) | ~np.isfinite(dBz_u) | ~np.isfinite(dBmag_u)

    # UTC timestamps for the uniform grid
    t0_ns  = t_orbit_start.value   # nanoseconds
    t_ns_u = (t0_ns + (t_uniform * 1e9).astype('int64'))

    # ── sliding window ────────────────────────────────────────────────────────
    W      = max(1, int(window_sec * sample_hz))   # samples per window
    stride = max(1, int(step_sec   * sample_hz))   # samples per step

    if n_samples < W:
        return None   # orbit too short for even one window

    windows, labels, win_times = [], [], []

    for start in range(0, n_samples - W + 1, stride):
        end    = start + W

        # skip any window that contains a NaN sample (KT17 unavailable)
        if nan_u[start:end].any():
            continue

        win_s0 = t_uniform[start]
        win_s1 = t_uniform[end - 1]

        win = np.stack([dBx_u[start:end],
                        dBz_u[start:end],
                        dBmag_u[start:end]], axis=0)   # (3, W)

        # score = fraction of the window occupied by loading
        # sum overlaps across all events, cap at 1
        win_dur  = win_s1 - win_s0
        total_overlap = 0.0
        for ev_s0, ev_s1, ev_dur in ev_intervals:
            total_overlap += max(0.0, min(win_s1, ev_s1) - max(win_s0, ev_s0))
        lbl = min(total_overlap / win_dur, 1.0) if win_dur > 0 else 0.0

        windows.append(win)
        labels.append(lbl)
        win_times.append(t_ns_u[start])

    return {
        'windows': np.stack(windows, axis=0).astype('float32'),  # (K, 2, W)
        'labels':  np.array(labels,    dtype='float32'),           # (K,) fraction in [0, 1]
        'times':   np.array(win_times, dtype='int64'),            # (K,)
        'orbit':   orb,
    }

# ── dataset assembly ──────────────────────────────────────────────────────────

def build_dataset(labels_path:  str   = LOADING_LABELS_JSON,
                  out_path:     str   = DATASET_NPZ,
                  window_sec:   int   = DEFAULT_WINDOW_SEC,
                  step_sec:     int   = DEFAULT_STEP_SEC,
                  sample_hz:    float = DEFAULT_SAMPLE_HZ) -> None:
    """
    Build the full sliding-window dataset and save to a .npz archive.

    Only orbits with ≥1 loading event are included.

    Saved arrays
    ------------
    windows  : float32 (N, 2, W)  — N examples, 2 channels, W time-steps
    labels   : int8    (N,)        — 0 = background, 1 = loading
    orbits   : int32   (N,)        — source orbit for each example
    times    : int64   (N,)        — UTC nanoseconds of window start
    meta     : JSON string with hyperparameters
    """
    with open(labels_path) as f:
        raw = json.load(f)

    # orbits with ≥1 loading event
    orb_events = {
        int(k): v['loading_events']
        for k, v in raw.items()
        if isinstance(v, dict) and v.get('loading_events')
    }
    print(f'Found {len(orb_events)} orbits with loading events.')
    print(f'Window: {window_sec} s   Step: {step_sec} s   '
          f'Sample rate: {sample_hz} Hz   '
          f'({int(window_sec * sample_hz)} samples/window)\n')

    all_windows, all_labels, all_orbits, all_times = [], [], [], []

    for orb, events in sorted(orb_events.items()):
        print(f'  Orbit {orb:5d}  ({len(events)} event(s)) … ',
              end='', flush=True)
        result = process_orbit(orb, events, window_sec, step_sec, sample_hz)
        if result is None:
            print('skipped (insufficient data)')
            continue

        K     = len(result['labels'])
        n_pos = int(result['labels'].sum())
        all_windows.append(result['windows'])
        all_labels.append(result['labels'])
        all_orbits.append(np.full(K, orb, dtype='int32'))
        all_times.append(result['times'])
        print(f'{K} windows,  {n_pos} positive '
              f'({100 * n_pos / K:.1f}%)')

    windows = np.concatenate(all_windows, axis=0)
    labels  = np.concatenate(all_labels,  axis=0).astype('float32')
    orbits  = np.concatenate(all_orbits,  axis=0)
    times   = np.concatenate(all_times,   axis=0)

    # ── balance classes: downsample background to match positive count ─────────
    pos_idx = np.where(labels > 0)[0]
    neg_idx = np.where(labels == 0)[0]
    rng     = np.random.default_rng(seed=0)
    neg_keep = rng.choice(neg_idx, size=len(pos_idx), replace=False)
    keep    = np.sort(np.concatenate([pos_idx, neg_keep]))
    windows = windows[keep]
    labels  = labels[keep]
    orbits  = orbits[keep]
    times   = times[keep]
    print(f'\nAfter balancing  : {len(labels):,} windows  '
          f'({len(pos_idx):,} pos + {len(neg_keep):,} neg)')

    meta = json.dumps({
        'window_sec': window_sec,
        'step_sec':   step_sec,
        'sample_hz':  sample_hz,
        'W':          int(window_sec * sample_hz),
        'channels':   ['dBx', 'dBz', 'dBmag'],
        'n_orbits':   len(orb_events),
    })

    np.savez_compressed(out_path,
                        windows=windows,
                        labels=labels,
                        orbits=orbits,
                        times=times,
                        meta=np.array(meta))

    n_tot     = len(labels)
    n_nonzero = int((labels > 0).sum())
    print(f'\nTotal windows     : {n_tot:,}')
    print(f'  Any loading     : {n_nonzero:,}  ({100 * n_nonzero / n_tot:.2f}%)')
    print(f'  Pure background : {n_tot - n_nonzero:,}  '
          f'({100 * (n_tot - n_nonzero) / n_tot:.2f}%)')
    print(f'  Mean label      : {labels.mean():.4f}')
    print(f'  Shape           : {windows.shape}  '
          f'({windows.nbytes / 1e6:.1f} MB)')
    print(f'Saved → {out_path}')


# ── dataset visualisation ───────────────────────────────────────────────────────

def plot_orbit_labels(orb:        int   = None,
                      window_sec: int   = DEFAULT_WINDOW_SEC,
                      step_sec:   int   = DEFAULT_STEP_SEC,
                      sample_hz:  float = DEFAULT_SAMPLE_HZ,
                      labels_path: str  = LOADING_LABELS_JSON) -> None:
    """
    Three-panel plot for one orbit:
      1. Bx / By / Bz  — observed (solid) and KT17 model (dashed)
      2. FIPS H+ differential flux spectrogram
      3. Window label (0/1) plotted at each window's centre time

    Parameters
    ----------
    orb         : orbit number.  If None, a random orbit from
                  human_loading_labels.json (with ≥1 event) is chosen.
    window_sec  : sliding-window length (must match the desired dataset config)
    step_sec    : stride between windows
    sample_hz   : resampling rate
    labels_path : path to human_loading_labels.json
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors
    from MESSENGER_analysis import (
        get_kt17_along_track,
        _fips_espec_path_for_date,
        load_fips_espec_tab,
        _fips_time_edges,
        _fips_bin_edges,
    )

    # ── pick orbit ────────────────────────────────────────────────────────────
    with open(labels_path) as f:
        raw = json.load(f)

    orb_pool = [int(k) for k, v in raw.items()
                if isinstance(v, dict) and v.get('loading_events')]

    if orb is None:
        orb = int(np.random.choice(orb_pool))
        print(f'Randomly selected orbit {orb}')

    loading_events = raw.get(str(orb), {}).get('loading_events', [])

    # ── load data ─────────────────────────────────────────────────────────────
    full_df = load_bowers_data_pkl(orbit_number=orb)
    orb_df  = filter_orbit_segment(full_df)
    if orb_df is None or orb_df.empty:
        print(f'No data for orbit {orb}'); return

    _, Bxm_raw, Bym_raw, Bzm_raw = get_kt17_along_track(df=orb_df)
    t_obs_raw = pd.to_datetime(orb_df['time'])

    # keep full-res arrays for top panel
    Bx_obs_raw = orb_df['magx'].to_numpy(dtype='float32')
    By_obs_raw = orb_df['magy'].to_numpy(dtype='float32')
    Bz_obs_raw = orb_df['magz'].to_numpy(dtype='float32')

    # ── low-pass filter then downsample to sample_hz ─────────────────────────
    t_s_raw   = (t_obs_raw - t_obs_raw.iloc[0]).dt.total_seconds().to_numpy()
    dt_native = float(np.median(np.diff(t_s_raw)))
    half_win  = max(1, int(round(0.5 / (sample_hz * dt_native))))
    win       = 2 * half_win + 1
    n_out     = int(np.floor(t_s_raw[-1] * sample_hz)) + 1
    t_uniform = np.linspace(0.0, t_s_raw[-1], n_out)

    def _lp_interp(arr):
        padded   = np.pad(arr.astype('float64'), half_win, mode='reflect')
        smoothed = np.convolve(padded, np.ones(win) / win, mode='valid')
        return np.interp(t_uniform, t_s_raw, smoothed).astype('float32')

    Bx_obs = _lp_interp(Bx_obs_raw)
    By_obs = _lp_interp(By_obs_raw)
    Bz_obs = _lp_interp(Bz_obs_raw)
    Bxm    = _lp_interp(Bxm_raw)
    Bym    = _lp_interp(Bym_raw)
    Bzm    = _lp_interp(Bzm_raw)

    # dataset features: ΔBx/|Bmod|, ΔBz/|Bmod| on the resampled grid
    Bmag_mod  = np.sqrt(Bxm**2 + Bym**2 + Bzm**2)
    Bmag_mod  = np.where(Bmag_mod > 1e-6, Bmag_mod, np.nan)
    dBx_norm  = (Bx_obs - Bxm) / Bmag_mod
    dBz_norm  = (Bz_obs - Bzm) / Bmag_mod

    t0_ns  = t_obs_raw.iloc[0].value
    t_obs  = pd.to_datetime((t0_ns + (t_uniform * 1e9).astype('int64')))

    # ── compute window labels ─────────────────────────────────────────────────
    result = process_orbit(orb, loading_events, window_sec, step_sec, sample_hz)
    if result is None:
        print(f'Orbit {orb} too short for window_sec={window_sec}'); return

    half_ns      = int(window_sec * 0.5 * 1e9)
    centre_times = (result['times'] + half_ns).astype('datetime64[ns]')
    win_labels   = result['labels'].astype(float)

    # ── FIPS H+ ───────────────────────────────────────────────────────────────
    fips_ok = False
    try:
        fips_path = _fips_espec_path_for_date(t_obs[0])
        fips_data = load_fips_espec_tab(fips_path)
        t_fips    = fips_data['t'].astype('datetime64[ns]')
        t0_ns     = np.datetime64(t_obs[0].to_datetime64(), 'ns')
        t1_ns     = np.datetime64(t_obs[-1].to_datetime64(), 'ns')
        fmask     = (t_fips >= t0_ns) & (t_fips <= t1_ns)
        if fmask.sum() >= 2:
            t_fw      = t_fips[fmask]
            flux_hp   = fips_data['H+_flux'][fmask]        # (M, 64)
            energy_hp = fips_data['H+_energy']             # (64,)
            t_edges   = _fips_time_edges(t_fw.astype('int64'))
            e_edges   = _fips_bin_edges(energy_hp)
            fips_ok   = True
    except Exception as e:
        print(f'  FIPS unavailable: {e}')

    # ── plot ──────────────────────────────────────────────────────────────────
    n_rows    = 4 if fips_ok else 3
    h_ratios  = [2.5, 1.5, 1.5, 1] if fips_ok else [2.5, 1.5, 1]
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2 + 1.6 * n_rows),
                             sharex=True,
                             gridspec_kw={'hspace': 0.06,
                                          'height_ratios': h_ratios})

    ax_b    = axes[0]
    ax_ds   = axes[1]
    ax_lbl  = axes[-1]
    ax_fp   = axes[2] if fips_ok else None

    # panel 1 — full-res raw B field + KT17
    Bmag_obs_raw = np.sqrt(Bx_obs_raw**2 + By_obs_raw**2 + Bz_obs_raw**2)
    Bmag_mod_raw = np.sqrt(Bxm_raw**2    + Bym_raw**2    + Bzm_raw**2)
    for vals, mod, color, name in [
        (Bx_obs_raw,  Bxm_raw,  'red',   'Bx'),
        (By_obs_raw,  Bym_raw,  'green', 'By'),
        (Bz_obs_raw,  Bzm_raw,  'blue',  'Bz'),
        (Bmag_obs_raw, Bmag_mod_raw, 'black', '|B|'),
    ]:
        ax_b.plot(t_obs_raw, vals, color=color, lw=0.6, label=name)
        ax_b.plot(t_obs_raw, mod,  color=color, lw=0.6, ls='--', alpha=0.6)
    ax_b.set_ylabel('B (nT)')
    ax_b.legend(fontsize=8, loc='upper right',
                title='solid=obs  dashed=KT17', title_fontsize=7)
    ax_b.grid(True, alpha=0.25)

    # shade loading intervals on B panel
    for ev in loading_events:
        ax_b.axvspan(pd.Timestamp(ev['start']), pd.Timestamp(ev['stop']),
                     color='limegreen', alpha=0.25, zorder=0)

    # panel 2 — resampled, normalised dataset features
    Bmag_obs_rs = np.sqrt(Bx_obs**2 + By_obs**2 + Bz_obs**2)
    Bmag_mod_rs = np.sqrt(Bxm**2   + Bym**2   + Bzm**2)
    Bmag_mod_rs = np.where(Bmag_mod_rs > 1e-6, Bmag_mod_rs, np.nan)
    dBmag_norm  = (Bmag_obs_rs - Bmag_mod_rs) / Bmag_mod_rs

    ax_ds.plot(t_obs, dBx_norm,  color='red',    lw=0.8, label='ΔBx/|B_mod|')
    ax_ds.plot(t_obs, dBz_norm,  color='blue',   lw=0.8, label='ΔBz/|B_mod|')
    ax_ds.plot(t_obs, dBmag_norm, color='black',  lw=0.8, label='ΔBmag/|B_mod|')
    ax_ds.axhline(0, color='k', lw=0.4, alpha=0.4)
    ax_ds.set_ylabel('ΔB / |B_mod|')
    ax_ds.legend(fontsize=8, loc='upper right')
    ax_ds.grid(True, alpha=0.25)
    for ev in loading_events:
        ax_ds.axvspan(pd.Timestamp(ev['start']), pd.Timestamp(ev['stop']),
                      color='limegreen', alpha=0.25, zorder=0)

    # panel 3 — FIPS H+
    if fips_ok:
        cmap = plt.cm.nipy_spectral.copy()
        T, E = np.meshgrid(t_edges, e_edges)
        ax_fp.pcolormesh(T, E, flux_hp.T, cmap=cmap,
                         norm=mcolors.LogNorm(vmin=1e5, vmax=1e9),
                         shading='flat')
        ax_fp.set_yscale('log')
        ax_fp.set_ylabel('H+ (keV)')
        ax_fp.text(0.005, 0.97, 'H+', transform=ax_fp.transAxes,
                   fontsize=8, va='top', color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', fc='k', alpha=0.4))

    # panel last — window labels
    ct = centre_times.astype('datetime64[ms]').astype(object)  # for matplotlib
    ax_lbl.fill_between(ct, win_labels, step='mid',
                        color='limegreen', alpha=0.7, linewidth=0)
    ax_lbl.set_yticks([0, 1])
    ax_lbl.set_yticklabels(['bkg', 'load'], fontsize=8)
    ax_lbl.set_ylabel('Label')
    ax_lbl.set_ylim(-0.05, 1.15)
    ax_lbl.grid(True, alpha=0.25)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].set_xlabel(f'UTC  {t_obs[0].strftime("%Y-%m-%d")}')

    date_str = t_obs[0].strftime('%Y-%m-%d')
    fig.suptitle(f'Orbit {orb}  —  {date_str}  '
                 f'(window {window_sec} s, step {step_sec} s)',
                 fontsize=11)
    plt.tight_layout()
    plt.show()


# ── model / training / evaluation ─────────────────────────────────────────────


def _load_npz_dataset(npz_path: str = DATASET_NPZ):
    """Load the .npz archive produced by build_dataset()."""
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d['meta']))
    return d['windows'], d['labels'], d['orbits'], d['times'], meta


class LoadingDataset:
    """
    PyTorch Dataset wrapping the sliding-window .npz archive.

    Split is done by *orbit* so that no orbit appears in both train and val.

    Parameters
    ----------
    npz_path    : path to nn_dataset.npz
    split       : 'train' or 'val'
    train_ratio : fraction of orbits assigned to train
    seed        : RNG seed for the orbit split
    """

    def __init__(self, npz_path: str = DATASET_NPZ,
                 split: str = 'train',
                 train_ratio: float = TRAIN_RATIO,
                 noise_sigma: float = NOISE_SIGMA,
                 noise_copies: int  = NOISE_COPIES,
                 seed: int = 42):
        import torch

        windows, labels, orbits, times, meta = _load_npz_dataset(npz_path)

        # ── global per-channel z-score normalisation ──────────────────────────
        # Use all data so train and val are on the same scale.
        # windows: (N, 2, W) — channel axis = 1
        for c in range(windows.shape[1]):
            mu  = float(windows[:, c, :].mean())
            sig = float(windows[:, c, :].std())
            if sig > 1e-8:
                windows[:, c, :] = (windows[:, c, :] - mu) / sig

        # ── orbit-based split ─────────────────────────────────────────────────
        unique_orbs = np.unique(orbits)
        rng         = np.random.default_rng(seed)
        rng.shuffle(unique_orbs)
        n_train     = max(1, int(len(unique_orbs) * train_ratio))

        if split == 'train':
            keep = set(unique_orbs[:n_train].tolist())
        else:
            keep = set(unique_orbs[n_train:].tolist())

        mask = np.isin(orbits, list(keep))
        wins = windows[mask]
        lbls = labels[mask]
        tms  = times[mask]    # UTC ns of each window start

        # ── Gaussian noise augmentation (training split only) ─────────────────
        if split == 'train' and noise_copies > 0 and noise_sigma > 0:
            copies_w, copies_l, copies_t = [wins], [lbls], [tms]
            for _ in range(noise_copies):
                noise = rng.normal(0.0, noise_sigma, size=wins.shape).astype('float32')
                copies_w.append(wins + noise)
                copies_l.append(lbls)
                copies_t.append(tms)   # same timestamps for noisy copies
            wins = np.concatenate(copies_w, axis=0)
            lbls = np.concatenate(copies_l, axis=0)
            tms  = np.concatenate(copies_t, axis=0)

        self.windows   = torch.tensor(wins, dtype=torch.float32)
        self.labels    = torch.tensor(lbls, dtype=torch.float32)
        self.times_ns  = tms          # int64 (N,) UTC nanoseconds of window start
        self.sample_hz = meta['sample_hz']
        self.meta      = meta

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


class LoadingCNN:
    """
    1-D convolutional network for loading event detection.

    Architecture
    ------------
    Input  : (batch, 2, W)  — 2 channels (ΔBx, ΔBz), W time-steps
    Conv1 → ReLU → MaxPool(2)        32 filters, kernel 7
    Conv2 → ReLU → AdaptiveAvgPool   32 filters, kernel 5
    Flatten → Linear(256→32) → ReLU → Linear(32→1) → Sigmoid
    Output : (batch,)  score in [0, 1]
    """

    @staticmethod
    def build(W: int, cnn_width: int = CNN_WIDTH):
        """
        Parameters
        ----------
        W         : number of time-steps per window  (= window_sec × sample_hz)
        cnn_width : number of filters per conv layer / nodes in the dense head
        """
        import torch.nn as nn

        C = cnn_width

        class _CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(3, C, kernel_size=7, padding=3), nn.ReLU(),
                    nn.Dropout(p=DROPOUT),
                    nn.MaxPool1d(2),
                    nn.Conv1d(C, C, kernel_size=5, padding=2), nn.ReLU(),
                    nn.Dropout(p=DROPOUT),
                    nn.AdaptiveAvgPool1d(8),   # → (batch, C, 8)
                )
                self.head = nn.Sequential(
                    nn.Flatten(),              # → (batch, C*8)
                    nn.Linear(C * 8, C),
                    nn.ReLU(),
                    nn.Dropout(p=DROPOUT),
                    nn.Linear(C, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.head(self.conv_layers(x)).squeeze(1)

        return _CNN()


def _plot_examples(model, ds_train, ds_val, device, epoch, n=4):
    """
    2×n grid of random examples: top row = train, bottom row = val.
    Each panel shows ΔBx and ΔBz (normalised), green background shaded
    by the truth label, and titles with truth / predicted scores.
    """
    import torch
    import matplotlib.pyplot as plt

    model.eval()
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 5),
                             sharex=True, sharey=False,
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.35})

    import matplotlib.dates as mdates

    rng = np.random.default_rng()

    for row, (ds, split_name) in enumerate([(ds_train, 'train'),
                                             (ds_val,   'val')]):
        idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
        x_batch  = ds.windows[idx].to(device)
        y_batch  = ds.labels[idx].numpy()
        t0s_ns   = ds.times_ns[idx]          # UTC ns of each window start
        sample_hz = ds.sample_hz

        with torch.no_grad():
            preds = model(x_batch).cpu().numpy()

        for col in range(n):
            ax  = axes[row, col]
            win = x_batch[col].cpu().numpy()   # (3, W)
            W   = win.shape[1]
            lbl = float(y_batch[col])
            prd = float(preds[col])

            # time axis: seconds within the window
            t_sec = np.arange(W) / sample_hz

            # green background proportional to truth label
            if lbl > 0:
                ax.axvspan(t_sec[0], t_sec[-1], color='limegreen',
                           alpha=0.15 + 0.35 * lbl, zorder=0)

            ax.plot(t_sec, win[0], color='red',    lw=0.9, label='ΔBx')
            ax.plot(t_sec, win[1], color='blue',   lw=0.9, label='ΔBz')
            ax.plot(t_sec, win[2], color='black',  lw=0.9, label='ΔBmag')
            ax.axhline(0, color='k', lw=0.4, alpha=0.4)

            ax.set_title(f'{split_name}  truth={lbl:.2f}  pred={prd:.2f}',
                         fontsize=7.5)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel('norm. ΔB/|B_mod|', fontsize=7)
            if row == 1:
                ax.set_xlabel('time (s)', fontsize=7)

    axes[0, 0].legend(fontsize=6, loc='upper right')
    fig.suptitle(f'Epoch {epoch} — random examples', fontsize=9)
    plt.tight_layout()
    out_dir = os.path.join(_DIR, 'figures', 'training_examples')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'epoch_{epoch:04d}.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def train_model(npz_path:    str   = DATASET_NPZ,
                batch_size:  int   = BATCH_SIZE,
                n_epochs:    int   = N_EPOCHS,
                train_ratio: float = TRAIN_RATIO,
                lr:          float = LEARNING_RATE,
                cnn_width:   int   = CNN_WIDTH,
                seed:        int   = 42,
                _save:       bool  = True,
                _plot:       bool  = True) -> float:
    """
    Train the LoadingCNN on the pre-built dataset and plot the loss history.

    Parameters
    ----------
    npz_path    : path to nn_dataset.npz
    batch_size  : mini-batch size
    n_epochs    : number of training epochs
    train_ratio : fraction of orbits used for training
    lr          : Adam learning rate
    cnn_width   : number of filters per conv layer (overrides CNN_WIDTH global)
    _save       : save model weights and history JSON (set False during sweeps)
    _plot       : show training summary plot (set False during sweeps)

    Returns
    -------
    Final validation MSE.
    """
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── datasets ──────────────────────────────────────────────────────────────
    ds_train = LoadingDataset(npz_path, split='train',
                              train_ratio=train_ratio, seed=seed)
    ds_val   = LoadingDataset(npz_path, split='val',
                              train_ratio=train_ratio, seed=seed)

    print(f'Train examples : {len(ds_train):,}')
    print(f'Val   examples : {len(ds_val):,}')

    loader_train = DataLoader(ds_train, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size,
                              shuffle=False, drop_last=False)

    # ── model ─────────────────────────────────────────────────────────────────
    W     = ds_train.windows.shape[-1]
    model = LoadingCNN.build(W, cnn_width=cnn_width).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {total:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.BCELoss()

    train_losses, val_losses = [], []

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, n_epochs + 1):
        # — train —
        model.train()
        running = 0.0
        for x, y in loader_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
        train_losses.append(running / len(ds_train))

        # — validate —
        model.eval()
        running = 0.0
        with torch.no_grad():
            for x, y in loader_val:
                x, y = x.to(device), y.to(device)
                pred  = model(x)
                loss  = criterion(pred, y)
                running += loss.item() * len(y)
        val_losses.append(running / max(len(ds_val), 1))

        saved_str = ''
        if PLOT_INTERVAL > 0 and epoch % PLOT_INTERVAL == 0:
            _plot_examples(model, ds_train, ds_val, device, epoch)
            model.train()   # restore train mode after eval inside _plot_examples
            saved_str = '  [examples saved]'

        if epoch % max(1, n_epochs // 10) == 0 or epoch == 1:
            print(f'  Epoch {epoch:4d}/{n_epochs}  '
                  f'train BCE={train_losses[-1]:.4f}  '
                  f'val BCE={val_losses[-1]:.4f}{saved_str}')

    # ── gather validation predictions for scatter ────────────────────────────
    model.eval()
    val_loader_sc = DataLoader(ds_val, batch_size=256, shuffle=False)
    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader_sc:
            all_pred.append(model(xb.to(device)).cpu().numpy())
            all_true.append(yb.numpy())
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    history = {
        'train_losses': train_losses,
        'val_losses':   val_losses,
        'val_true':     all_true.tolist(),
        'val_pred':     all_pred.tolist(),
        'batch_size':   batch_size,
        'lr':           lr,
        'n_epochs':     n_epochs,
        'cnn_width':    cnn_width,
    }

    if _save:
        out_pt   = os.path.join(_DIR, 'loading_cnn.pt')
        out_hist = os.path.join(_DIR, 'loading_cnn_history.json')
        torch.save(model.state_dict(), out_pt)
        print(f'Model saved → {out_pt}')
        with open(out_hist, 'w') as f:
            json.dump(history, f)
        print(f'History saved → {out_hist}')

    if _plot:
        plot_training_summary(history)

    return val_losses[-1]


def plot_training_summary(history=None, hist_path=None):
    """
    Plot loss curves + validation scatter from a saved history dict or JSON file.

    Parameters
    ----------
    history   : dict  (as returned / saved by train_model)
    hist_path : str   path to loading_cnn_history.json  (used if history is None)
    """
    if history is None:
        if hist_path is None:
            hist_path = os.path.join(_DIR, 'loading_cnn_history.json')
        with open(hist_path) as f:
            history = json.load(f)

    train_losses = history['train_losses']
    val_losses   = history['val_losses']
    all_true     = np.array(history['val_true'])
    all_pred     = np.array(history['val_pred'])
    batch_size   = history['batch_size']
    lr           = history['lr']
    n_epochs     = history['n_epochs']

    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(train_losses) + 1)
    fig, (ax, ax_sc) = plt.subplots(1, 2, figsize=(13, 4))

    ax.semilogy(epochs, train_losses, label='Train BCE', color='steelblue', lw=1.8)
    ax.semilogy(epochs, val_losses,   label='Val BCE',   color='tomato',    lw=1.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE loss  (log scale)')
    ax.set_title(f'LoadingCNN training  (batch={batch_size}, lr={lr}, {n_epochs} epochs)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    import matplotlib.colors as mcolors
    h, xe, ye, img = ax_sc.hist2d(all_true, all_pred, bins=40,
                                   range=[[0, 1], [0, 1]],
                                   cmap='plasma',
                                   norm=mcolors.LogNorm(vmin=1))
    plt.colorbar(img, ax=ax_sc, label='Count (log scale)')
    ax_sc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label='y = x')
    ax_sc.set_xlabel('True label')
    ax_sc.set_ylabel('Model output')
    ax_sc.set_title('Validation set predictions')
    ax_sc.set_xlim(0, 1)
    ax_sc.set_ylim(0, 1)
    ax_sc.set_aspect('equal')
    ax_sc.legend(fontsize=9)

    plt.tight_layout()
    plt.show()



def run_sweep(npz_base:    str   = DATASET_NPZ,
              n_epochs:    int   = N_EPOCHS,
              batch_size:  int   = BATCH_SIZE,
              train_ratio: float = TRAIN_RATIO,
              lr:          float = LEARNING_RATE,
              labels_path: str   = LOADING_LABELS_JSON,
              step_sec:    int   = DEFAULT_STEP_SEC) -> None:
    """
    Grid search over all permutations of SWEEP_WINDOW_SEC × SWEEP_SAMPLE_HZ × SWEEP_CNN_WIDTH.

    For each combination a fresh dataset is built (saved to a temp .npz), the CNN
    is trained, and the final validation MSE is recorded.  A ranked summary is
    printed at the end.

    Sweep lists are set at the top of the file:
        SWEEP_WINDOW_SEC, SWEEP_SAMPLE_HZ, SWEEP_CNN_WIDTH
    """
    import itertools

    combos = list(itertools.product(SWEEP_WINDOW_SEC, SWEEP_SAMPLE_HZ, SWEEP_CNN_WIDTH))
    n_combos = len(combos)
    print(f"\n{'='*60}")
    print(f"Hyperparameter sweep: {n_combos} combinations")
    print(f"  window_sec : {SWEEP_WINDOW_SEC}")
    print(f"  sample_hz  : {SWEEP_SAMPLE_HZ}")
    print(f"  cnn_width  : {SWEEP_CNN_WIDTH}")
    print(f"{'='*60}\n")

    results = []
    for i, (window_sec, sample_hz, cnn_width) in enumerate(combos, 1):
        tag = f"win{window_sec}_hz{sample_hz}_cnn{cnn_width}"
        remaining = n_combos - i
        print(f"\n{'─'*60}")
        print(f"[{i}/{n_combos}]  window={window_sec}s  hz={sample_hz}  cnn_width={cnn_width}"
              f"  ({remaining} remaining after this)")
        print(f"{'─'*60}")

        # build dataset to a temp file
        tmp_npz = os.path.join(os.path.dirname(npz_base), f'_sweep_{tag}.npz')
        try:
            build_dataset(labels_path=labels_path, out_path=tmp_npz,
                          window_sec=window_sec, step_sec=step_sec,
                          sample_hz=sample_hz)
        except Exception as e:
            print(f"  build failed: {e}")
            results.append((tag, window_sec, sample_hz, cnn_width, float('nan')))
            continue

        try:
            val_mse = train_model(npz_path=tmp_npz,
                                  batch_size=batch_size,
                                  n_epochs=n_epochs,
                                  train_ratio=train_ratio,
                                  lr=lr,
                                  cnn_width=cnn_width,
                                  _save=False,
                                  _plot=False)
        except Exception as e:
            print(f"  train failed: {e}")
            val_mse = float('nan')
        finally:
            try:
                os.remove(tmp_npz)
            except OSError:
                pass

        print(f"\n  DONE [{i}/{n_combos}]  window={window_sec}s  hz={sample_hz}"
              f"  cnn_width={cnn_width}  =>  val BCE = {val_mse:.4f}"
              f"  ({remaining} remaining)\n")
        results.append((tag, window_sec, sample_hz, cnn_width, val_mse))

    # ── summary ───────────────────────────────────────────────────────────────
    results.sort(key=lambda r: (float('inf') if r[4] != r[4] else r[4]))
    print(f"\n{'='*60}")
    print(f"Sweep complete — ranked by validation BCE")
    print(f"{'='*60}")
    print(f"  {'Rank':>4}  {'window_sec':>10}  {'sample_hz':>9}  {'cnn_width':>9}  {'val_BCE':>8}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*8}")
    for rank, (tag, win, hz, cw, mse) in enumerate(results, 1):
        marker = '  <-- best' if rank == 1 else ''
        print(f"  {rank:>4}  {win:>10}  {hz:>9.2f}  {cw:>9}  {mse:>8.4f}{marker}")
    print(f"{'='*60}\n")

    # ── retrain best configuration and save weights / history ─────────────────
    best_tag, best_win, best_hz, best_cw, best_mse = results[0]
    if not (best_mse != best_mse):   # skip if best is NaN
        print(f"Re-training best configuration ({best_tag}) to save weights …")
        tmp_npz = os.path.join(os.path.dirname(npz_base), f'_sweep_{best_tag}.npz')
        build_dataset(labels_path=labels_path, out_path=tmp_npz,
                      window_sec=best_win, step_sec=step_sec,
                      sample_hz=best_hz)
        train_model(npz_path=tmp_npz,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    train_ratio=train_ratio,
                    lr=lr,
                    cnn_width=best_cw,
                    _save=True,
                    _plot=True)
        try:
            os.remove(tmp_npz)
        except OSError:
            pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--build',  action='store_true',
                        help='Build sliding-window dataset and save to .npz')
    parser.add_argument('--plot',   action='store_true',
                        help='Plot B field, FIPS H+, and window labels for one orbit')
    parser.add_argument('--train',   action='store_true',
                        help='Train the CNN on nn_dataset.npz and plot loss history')
    parser.add_argument('--summary', action='store_true',
                        help='Replot training summary from saved loading_cnn_history.json')
    parser.add_argument('--sweep',   action='store_true',
                        help='Grid search over SWEEP_WINDOW_SEC × SWEEP_SAMPLE_HZ × SWEEP_CNN_WIDTH')
    parser.add_argument('--orbit',  type=int, default=None,
                        help='Orbit number for --plot (default: random)')
    parser.add_argument('--labels', default=LOADING_LABELS_JSON,
                        help='Path to human_loading_labels.json')
    parser.add_argument('--out',    default=DATASET_NPZ,
                        help='Output .npz path')
    parser.add_argument('--window', type=int,   default=DEFAULT_WINDOW_SEC,
                        help=f'Window length in seconds (default {DEFAULT_WINDOW_SEC})')
    parser.add_argument('--step',   type=int,   default=DEFAULT_STEP_SEC,
                        help=f'Stride in seconds (default {DEFAULT_STEP_SEC})')
    parser.add_argument('--hz',     type=float, default=DEFAULT_SAMPLE_HZ,
                        help=f'Resample rate in Hz (default {DEFAULT_SAMPLE_HZ})')
    parser.add_argument('--epochs', type=int,   default=N_EPOCHS,
                        help=f'Number of training epochs (default {N_EPOCHS})')
    parser.add_argument('--batch',  type=int,   default=BATCH_SIZE,
                        help=f'Mini-batch size (default {BATCH_SIZE})')
    parser.add_argument('--split',  type=float, default=TRAIN_RATIO,
                        help=f'Train/val orbit split ratio (default {TRAIN_RATIO})')
    args = parser.parse_args()

    if args.build:
        build_dataset(labels_path=args.labels, out_path=args.out,
                      window_sec=args.window, step_sec=args.step,
                      sample_hz=args.hz)
    elif args.plot:
        plot_orbit_labels(orb=args.orbit,
                          window_sec=args.window, step_sec=args.step,
                          sample_hz=args.hz, labels_path=args.labels)
    elif args.train:
        train_model(npz_path=args.out,
                    batch_size=args.batch,
                    n_epochs=args.epochs,
                    train_ratio=args.split)
    elif args.summary:
        plot_training_summary()
    elif args.sweep:
        run_sweep(labels_path=args.labels, step_sec=args.step,
                  n_epochs=args.epochs, batch_size=args.batch,
                  train_ratio=args.split)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
