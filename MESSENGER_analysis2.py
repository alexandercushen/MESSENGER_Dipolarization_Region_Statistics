#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MESSENGER magnetometer and FIPS particle data — core utilities.

Functions
---------
load_bowers_data_pkl        Load MAG data (time, Bx/By/Bz, ephemeris, region)
get_kt17_along_track        Evaluate KT17 model field at MESSENGER positions
transform_to_fac            Rotate observed B into field-aligned coordinates
set_ephemeris_ticklabels    Add UT / lat / lon / alt tick labels to a plot axis
plot_quick_look             Overview plot: Bx/By/Bz + optional FIPS spectrogram
load_fips_espec_tab         Load a PDS FIPS ESPEC TAB file into a dict of flux arrays
plot_fips_espec_spectrogram Plot FIPS differential-flux spectrogram(s)
plot_fips_for_orbit         Convenience wrapper: FIPS spectrogram for one orbit number
download_all_fips_espec     Download the full mission FIPS ESPEC dataset from PDS
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import pandas as pd
import spiceypy as spice
import urllib.request
import json
import KT17

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KERNEL_DIR = os.path.expanduser('~/mercury_dipolarizations/messenger_kernels')
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()

_DI_CSV = os.path.join(_SCRIPT_DIR, 'orb_num_start_ut_rhel_di.csv')

# ---------------------------------------------------------------------------
# DistIndex lookup table (loaded once; used by get_kt17_along_track)
# ---------------------------------------------------------------------------
_DI_TABLE = None

def _get_di_table():
    """Load and cache the orbit DistIndex table."""
    global _DI_TABLE
    if _DI_TABLE is not None:
        return _DI_TABLE
    df = pd.read_csv(_DI_CSV, parse_dates=['start_ut'])
    df = df.dropna(subset=['di']).sort_values('start_ut').reset_index(drop=True)
    _DI_TABLE = df
    return _DI_TABLE

def _lookup_dist_index(t, default=50.0):
    """DistIndex for the orbit whose start_ut is the latest time <= t."""
    tbl = _get_di_table()
    before = tbl[tbl['start_ut'] <= t]
    if before.empty:
        return default
    return float(before.iloc[-1]['di'])

# ---------------------------------------------------------------------------
# SPICE kernels
# ---------------------------------------------------------------------------
def load_messenger_kernels():
    """Load all SPICE kernels found in KERNEL_DIR."""
    kernel_files = [f for f in os.listdir(KERNEL_DIR)
                    if f.endswith(('.bsp', '.tls', '.tpc', '.tf', '.tsc'))]
    if not kernel_files:
        raise FileNotFoundError(
            f'No kernel files found in {KERNEL_DIR}.\n'
            f'Download from: https://naif.jpl.nasa.gov/pub/naif/pds/data/'
            f'mess-e_v_h-spice-6-v1.0/messsp_1000/data/')
    for f in sorted(kernel_files):
        spice.furnsh(os.path.join(KERNEL_DIR, f))
        print(f'  Loaded: {f}')

# ---------------------------------------------------------------------------
# MAG data
# ---------------------------------------------------------------------------
def load_bowers_data_pkl(trange=None, orbit_number=None, filename=None):
    """
    Load the Bowers MESSENGER dataset filtered to a time range or orbit.

    Columns: time, ephx, ephy, ephz (R_M MSM), magx, magy, magz, magamp (nT),
             Transition, Type_num, orbit_number.

    On first call the pickle is converted to Parquet for fast future loads.

    Parameters
    ----------
    trange       : optional [start, end] as 'YYYY-MM-DD/HH:MM:SS' strings or
                   anything pd.Timestamp accepts
    orbit_number : optional int or list of ints
    filename     : optional explicit path to the .pkl file
    """
    import pyarrow.parquet as pq

    if filename is None:
        filename = os.path.expanduser(
            '~/mercury_dipolarizations/MESSENGER_Full_Data_Ab_MSM.pkl')

    parquet_path = os.path.splitext(filename)[0] + '.parquet'

    if not os.path.exists(parquet_path):
        print('First run: converting pickle -> Parquet (one-time, may be slow)...')
        with open(filename, 'rb') as f:
            df_full = pickle.load(f)
        df_full['time'] = pd.to_datetime(df_full['time'])
        df_full = df_full.sort_values('time').reset_index(drop=True)
        df_full.to_parquet(parquet_path, index=False, row_group_size=50_000)
        print(f'Saved: {parquet_path}  ({os.path.getsize(parquet_path)/1e6:.1f} MB)')
        del df_full

    fmt = '%Y-%m-%d/%H:%M:%S'
    def _to_ts(v):
        return pd.Timestamp(datetime.strptime(v, fmt)) if isinstance(v, str) else pd.Timestamp(v)

    if orbit_number is not None:
        orbs = [orbit_number] if np.isscalar(orbit_number) else list(orbit_number)
        return pq.read_table(parquet_path, filters=[('orbit_number', 'in', orbs)]).to_pandas()
    elif trange is not None:
        t0, t1 = _to_ts(trange[0]), _to_ts(trange[1])
        return pq.read_table(parquet_path,
                             filters=[('time', '>=', t0), ('time', '<=', t1)]).to_pandas()
    else:
        return pq.read_table(parquet_path).to_pandas()

# ---------------------------------------------------------------------------
# Human-labelled loading periods
# ---------------------------------------------------------------------------
def load_human_loading_labels(json_path=None, smooth_sec=30.0):
    """
    Read the human-reviewed loading-event labels and return a DataFrame.

    For each event the partition time (peak smoothed |ΔBx|) is computed by
    loading the orbit's MAG data and KT17 model once per orbit.

    Parameters
    ----------
    json_path  : str, optional
        Path to human_loading_labels.json.  Defaults to the copy in the
        same directory as this file.
    smooth_sec : float
        Smoothing window passed to partition_loading_event (seconds).

    Returns
    -------
    pd.DataFrame with columns:
        orbit     : int
        start     : pd.Timestamp
        partition : pd.Timestamp  (peak |ΔBx|; NaT if not found)
        stop      : pd.Timestamp
    One row per loading event, sorted by start time.
    """
    if json_path is None:
        json_path = os.path.join(_SCRIPT_DIR, 'human_loading_labels.json')

    with open(json_path) as f:
        labels = json.load(f)

    # Group events by orbit so each orbit's data is loaded only once
    by_orbit = {}
    for orb_str, entry in labels.items():
        if not isinstance(entry, dict):
            continue
        if not entry.get('reviewed') or not entry.get('loading_events'):
            continue
        orb = int(orb_str)
        by_orbit[orb] = [{'start': pd.Timestamp(ev['start']),
                           'stop':  pd.Timestamp(ev['stop'])}
                         for ev in entry['loading_events']]

    rows = []
    for orb, events in by_orbit.items():
        try:
            orb_df        = load_bowers_data_pkl(orbit_number=orb)
            t_obs         = pd.to_datetime(orb_df['time'])
            _, Bxm, _, _  = get_kt17_along_track(df=orb_df)
            dBx           = orb_df['magx'].to_numpy() - Bxm
        except Exception:
            for ev in events:
                rows.append({'orbit': orb, 'start': ev['start'],
                             'partition': pd.NaT, 'stop': ev['stop']})
            continue

        for ev in events:
            t_part = partition_loading_event(ev['start'], ev['stop'],
                                             t_obs, dBx,
                                             smooth_sec=smooth_sec)
            rows.append({
                'orbit':     orb,
                'start':     ev['start'],
                'partition': pd.Timestamp(t_part) if t_part is not None else pd.NaT,
                'stop':      ev['stop'],
            })

    return (pd.DataFrame(rows, columns=['orbit', 'start', 'partition', 'stop'])
              .sort_values('start')
              .reset_index(drop=True))

def partition_loading_event(t_start, t_stop, t_obs, dBx,
                            smooth_sec=60.0):
    """
    Partition one loading event into a loading phase and an unloading phase by
    locating the peak of smoothed |ΔBx| (= |Bx_obs − Bx_mod|) within the
    event window.

    Parameters
    ----------
    t_start, t_stop : pd.Timestamp
        Start and end of the loading event.
    t_obs : pd.Series or array-like of Timestamps (N,)
        Full-orbit time axis (does NOT need to be pre-masked to the event).
    dBx : array-like (N,)
        Full-orbit ΔBx residual (Bx_obs − Bx_mod, nT), aligned with t_obs.
    smooth_sec : float
        Window (seconds) for the rolling-mean smoother applied to |ΔBx|.
        Default 60 s.

    Returns
    -------
    t_partition : pd.Timestamp or None
        Timestamp of peak smoothed |ΔBx|.
        Loading phase  : [t_start, t_partition]
        Unloading phase: [t_partition, t_stop]
        Returns None if the event contains fewer than 2 samples.
    """
    t_obs = pd.Series(pd.to_datetime(t_obs)) if not isinstance(t_obs, pd.Series) else pd.to_datetime(t_obs)
    dBx   = np.asarray(dBx)

    mask = (t_obs >= t_start) & (t_obs <= t_stop)
    if mask.sum() < 2:
        return None

    t_ev   = t_obs[mask].reset_index(drop=True)
    dbx_ev = np.abs(dBx[mask.to_numpy() if hasattr(mask, 'to_numpy') else mask])

    # rolling smooth
    t_s    = (t_ev - t_ev.iloc[0]).dt.total_seconds().to_numpy()
    dt_med = float(np.median(np.diff(t_s))) if len(t_s) > 1 else 1.0
    win    = max(3, int(round(smooth_sec / dt_med)))
    smooth = (pd.Series(dbx_ev)
              .rolling(win, center=True, min_periods=1)
              .mean()
              .to_numpy())

    return t_ev.iloc[int(np.argmax(smooth))]

# ---------------------------------------------------------------------------
# Current-sheet crossing segment
# ---------------------------------------------------------------------------
def filter_orbit_segment(orb_df):
    """
    Trim an orbit DataFrame to the nightside current-sheet crossing segment.

    Keeps the first continuous run of points satisfying:
      - ephx < 0          (nightside)
      - -2 < ephz < 1.25  (near current sheet)
      - r < 3 R_M         (within inner magnetosphere)
      - azimuth fully within 90°–270° (nightside hemisphere)

    Returns a slice of orb_df, or an empty DataFrame if no segment qualifies.
    """
    empty = orb_df.iloc[0:0]

    criteria = (
        (orb_df['ephx'] < 0.0) &
        (orb_df['ephz'] > -2.0) &
        (orb_df['ephz'] < 1.25) &
        ((orb_df['ephx']**2 + orb_df['ephy']**2 + orb_df['ephz']**2) < 3**2)
    ).to_numpy()

    starts = np.where(np.diff(criteria.astype(int)) == 1)[0] + 1
    if criteria[0]:
        starts = np.concatenate([[0], starts])
    if len(starts) == 0:
        return empty

    seg_start = starts[0]
    ends = np.where(np.diff(criteria.astype(int)) == -1)[0] + 1
    ends = ends[ends > seg_start]
    seg_end = ends[0] - 1 if len(ends) > 0 else len(criteria) - 1

    seg_x = orb_df['ephx'].to_numpy()[seg_start:seg_end + 1]
    seg_y = orb_df['ephy'].to_numpy()[seg_start:seg_end + 1]
    phi = np.degrees(np.arctan2(seg_y, seg_x)) % 360
    if not np.all((phi >= 90) & (phi <= 270)):
        return empty

    return orb_df.iloc[seg_start:seg_end + 1]

# ---------------------------------------------------------------------------
# KT17 model field
# ---------------------------------------------------------------------------
def get_kt17_along_track(trange=None, df=None, **kt17_kwargs):
    """
    Evaluate the KT17 model field at MESSENGER's observed positions.

    Parameters
    ----------
    trange       : optional [start, end] strings; derived automatically if df given
    df           : optional pre-loaded Bowers DataFrame
    **kt17_kwargs : forwarded to KT17.ModelField (e.g. Rsm, DistIndex)

    Returns
    -------
    time       : np.ndarray of datetime64
    Bx, By, Bz : np.ndarray (nT), model field in MSM
    """
    from astropy.time import Time
    import astropy.units as u
    from astropy.coordinates import get_body, solar_system_ephemeris

    if df is None:
        if trange is None:
            raise ValueError('Either trange or df must be provided.')
        df = load_bowers_data_pkl(trange=trange)

    x = df['ephx'].to_numpy()
    y = df['ephy'].to_numpy()
    z = df['ephz'].to_numpy()

    t_obs = pd.to_datetime(df['time'])
    t0    = Time(t_obs.iloc[0].to_pydatetime())
    with solar_system_ephemeris.set('builtin'):
        mercury = get_body('mercury', t0)
        sun     = get_body('sun',     t0)
    rsun = mercury.separation_3d(sun).to(u.AU).value

    if 'DistIndex' not in kt17_kwargs:
        kt17_kwargs = dict(kt17_kwargs)
        kt17_kwargs['DistIndex'] = _lookup_dist_index(t_obs.iloc[0])

    T = KT17.ModelField(x, y, z, Rsun=rsun, **kt17_kwargs)
    return df['time'].to_numpy(), T[0], T[1], T[2]

# ---------------------------------------------------------------------------
# Field-aligned coordinates
# ---------------------------------------------------------------------------
def transform_to_fac(bx_meas, by_meas, bz_meas, bx_mod, by_mod, bz_mod, rx, ry, rz):
    """
    Rotate observed B into a Field-Aligned Coordinate (FAC) system.

    Basis vectors:
      b_hat    -- along model field (parallel)
      phi_hat  -- cross(b_hat, R)  (azimuthal)
      perp_hat -- cross(phi_hat, b_hat)  (meridional, completes right-hand set)

    Returns
    -------
    B_perp, B_phi, B_par : each shape (N,), nT
    """
    B_meas = np.column_stack([bx_meas, by_meas, bz_meas])
    B_mod  = np.column_stack([bx_mod,  by_mod,  bz_mod])
    R      = np.column_stack([rx,      ry,      rz])

    b_hat    = B_mod  / np.linalg.norm(B_mod,  axis=1, keepdims=True)
    phi_vec  = np.cross(b_hat, R)
    phi_hat  = phi_vec / np.linalg.norm(phi_vec, axis=1, keepdims=True)
    perp_hat = np.cross(phi_hat, b_hat)

    return (np.sum(B_meas * perp_hat, axis=1),
            np.sum(B_meas * phi_hat,  axis=1),
            np.sum(B_meas * b_hat,    axis=1))

# ---------------------------------------------------------------------------
# Ephemeris tick labels
# ---------------------------------------------------------------------------
def set_ephemeris_ticklabels(ax, df, fontsize=15, coords='latlon'):
    """
    Replace x-axis tick labels with multi-row ephemeris information.

    Parameters
    ----------
    ax     : matplotlib Axes
    df     : DataFrame with columns 'time', 'ephx', 'ephy', 'ephz'
    coords : 'latlon'  -> UT / E.Lon / Lat / Alt_MSO
             'xyz'     -> UT / X / Y / Z  (R_M)
    """
    t_obs = pd.to_datetime(df['time'])
    tick_locs  = ax.get_xticks()
    tick_times = [mdates.num2date(t).replace(tzinfo=None) for t in tick_locs]

    t_arr = t_obs.values.astype('datetime64[ns]')
    x_arr = df['ephx'].to_numpy()
    y_arr = df['ephy'].to_numpy()
    z_arr = df['ephz'].to_numpy()

    if coords == 'latlon':
        r_arr       = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
        lat_arr     = np.degrees(np.arcsin(np.clip(z_arr / r_arr, -1, 1)))
        lon_arr     = np.degrees(np.arctan2(y_arr, x_arr)) % 360
        alt_mso_arr = np.sqrt(x_arr**2 + y_arr**2 + (z_arr + 0.2)**2) - 1.0
        labels = []
        for tt in tick_times:
            idx = int(np.clip(np.searchsorted(t_arr, np.datetime64(tt, 'ns')),
                              0, len(t_arr) - 1))
            labels.append(
                f'{tt.strftime("%H:%M:%S")}\n{lon_arr[idx]:.1f}\n'
                f'{lat_arr[idx]:+.1f}\n{alt_mso_arr[idx]:.3f}')
        row_labels = ['UT', 'E.Lon', 'Lat', 'Alt$_{MSO}$ (R$_M$)']
    else:
        labels = []
        for tt in tick_times:
            idx = int(np.clip(np.searchsorted(t_arr, np.datetime64(tt, 'ns')),
                              0, len(t_arr) - 1))
            labels.append(
                f'{tt.strftime("%H:%M:%S")}\n{x_arr[idx]:.3f}\n'
                f'{y_arr[idx]:.3f}\n{z_arr[idx]:.3f}')
        row_labels = ['UT', 'X (R$_M$)', 'Y (R$_M$)', 'Z (R$_M$)']

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(labels, fontsize=fontsize)
    for i, rl in enumerate(row_labels):
        ax.annotate(rl, xy=(1.01, -0.06 * i),
                    xycoords=('axes fraction', 'axes fraction'),
                    fontsize=fontsize * 0.9, va='top', ha='left',
                    annotation_clip=False)
    return ax

# ---------------------------------------------------------------------------
# Quick-look plot
# ---------------------------------------------------------------------------
def plot_quick_look(t0=None, t1=None, species=('H+',), figsize=(14, 5), smooth_sec=1,
                    orbit=None, only_cs=False, show_loading=True, show_kt17=False,
                    save_path=None):
    """
    Overview plot for an arbitrary time window or orbit number.

    Rows (top to bottom):
      - Bx / By / Bz / |B| (nT, optionally smoothed)
      - ΔBx / ΔBy / ΔBz (obs − KT17) with zero line  [if show_kt17=True]
      - Region colour bar (from Type_num column, if present)
      - One FIPS differential-flux spectrogram per entry in *species*

    Parameters
    ----------
    t0, t1        : anything pd.Timestamp accepts (required if orbit not given)
    orbit         : int orbit number; overrides t0/t1 if given
    only_cs       : if True (and orbit given), restrict the window to the
                    nightside current-sheet crossing segment identified by
                    filter_orbit_segment; ignored when t0/t1 are given directly
    show_loading  : if True, overlay vertical lines for any human-labelled
                    loading events whose start/stop overlap the plot window:
                      green  dashed  — event start
                      orange dashed  — partition (peak |ΔBx|)
                      red    dashed  — event stop
    show_kt17     : if True, add a residual panel (obs − KT17) below the mag panel
    species       : FIPS species to show, e.g. ('H+',) or ('H+', 'He++').
                    Pass () for mag-only.
    smooth_sec    : boxcar smoothing window in seconds (0 or None for raw)

    Returns matplotlib Figure.
    """
    from matplotlib.colors import LogNorm

    if orbit is not None:
        orb_df = load_bowers_data_pkl(orbit_number=orbit)
        if only_cs:
            orb_df = filter_orbit_segment(orb_df)
            if orb_df.empty:
                raise ValueError(f'No current-sheet crossing found for orbit {orbit}.')
        t_obs  = pd.to_datetime(orb_df['time'])
        t0, t1 = t_obs.iloc[0], t_obs.iloc[-1]
    elif t0 is None or t1 is None:
        raise ValueError('Provide either orbit= or both t0 and t1.')

    t0 = pd.Timestamp(t0)
    t1 = pd.Timestamp(t1)

    df  = load_bowers_data_pkl(trange=[t0, t1])
    t   = pd.to_datetime(df['time'])

    dt_s = (float(np.median(np.diff((t - t.iloc[0]).dt.total_seconds().to_numpy())))
            if len(t) > 1 else 1.0)
    win  = int(round(smooth_sec / dt_s)) if smooth_sec else 1

    if win > 1:
        kernel = np.ones(win) / win
        def _boxcar(arr):
            pad = win // 2
            return np.convolve(np.pad(arr.astype(float), pad, mode='reflect'),
                               kernel, mode='valid')[:len(arr)]
        Bx = _boxcar(df['magx'].to_numpy())
        By = _boxcar(df['magy'].to_numpy())
        Bz = _boxcar(df['magz'].to_numpy())
    else:
        Bx, By, Bz = df['magx'].to_numpy(), df['magy'].to_numpy(), df['magz'].to_numpy()

    # KT17 model field (computed here so residuals are ready before subplot layout)
    dBx = dBy = dBz = dBmag = None
    Bxm_plot = Bym_plot = Bzm_plot = Bmagm_plot = None
    if show_kt17:
        try:
            _, Bxm, Bym, Bzm = get_kt17_along_track(df=df)
            Bmagm = np.sqrt(Bxm**2 + Bym**2 + Bzm**2)
            if win > 1:
                Bxm_plot   = _boxcar(Bxm)
                Bym_plot   = _boxcar(Bym)
                Bzm_plot   = _boxcar(Bzm)
                Bmagm_plot = _boxcar(Bmagm)
                dBx   = _boxcar(df['magx'].to_numpy()   - Bxm)
                dBy   = _boxcar(df['magy'].to_numpy()   - Bym)
                dBz   = _boxcar(df['magz'].to_numpy()   - Bzm)
                dBmag = _boxcar(df['magamp'].to_numpy() - Bmagm)
            else:
                Bxm_plot, Bym_plot, Bzm_plot, Bmagm_plot = Bxm, Bym, Bzm, Bmagm
                dBx   = df['magx'].to_numpy()   - Bxm
                dBy   = df['magy'].to_numpy()   - Bym
                dBz   = df['magz'].to_numpy()   - Bzm
                dBmag = df['magamp'].to_numpy() - Bmagm
        except Exception as e:
            print(f'KT17 unavailable: {e}')
            show_kt17 = False

    _region_colors = {
        1: '#4477AA', 2: '#66CCEE', 3: '#CCBB44', 4: '#EE6677', 5: '#AA3377',
    }
    _region_labels = {1: 'MS', 2: 'MSH', 3: 'SW', 4: 'BS', 5: 'MP'}
    has_region = 'Type_num' in df.columns

    n_fips   = len(species)
    nrows    = 1 + int(show_kt17) + int(has_region) + n_fips
    h_ratios = [3] + ([2] if show_kt17 else []) + ([0.12] if has_region else []) + [1] * n_fips
    fig, axes = plt.subplots(nrows, 1, sharex=True,
                             figsize=(figsize[0], figsize[1] + int(show_kt17) * 2 + n_fips * 1.5),
                             gridspec_kw={'hspace': 0.05, 'height_ratios': h_ratios})
    axes = list(np.atleast_1d(axes))

    if win > 1:
        Bmag = _boxcar(df['magamp'].to_numpy())
    else:
        Bmag = df['magamp'].to_numpy()

    ax_mag = axes[0]
    ax_mag.plot(t, Bx,   color='red',   lw=0.7, label='Bx')
    ax_mag.plot(t, By,   color='green', lw=0.7, label='By')
    ax_mag.plot(t, Bz,   color='blue',  lw=0.7, label='Bz')
    ax_mag.plot(t, Bmag, color='black', lw=0.7, label='|B|')
    if show_kt17 and Bxm_plot is not None:
        ax_mag.plot(t, Bxm_plot,   color='red',   lw=0.7, ls='--', alpha=0.6, label='Bx KT17')
        ax_mag.plot(t, Bym_plot,   color='green', lw=0.7, ls='--', alpha=0.6, label='By KT17')
        ax_mag.plot(t, Bzm_plot,   color='blue',  lw=0.7, ls='--', alpha=0.6, label='Bz KT17')
        ax_mag.plot(t, Bmagm_plot, color='black', lw=0.7, ls='--', alpha=0.6, label='|B| KT17')
    ax_mag.axhline(0, color='k', lw=0.4, alpha=0.4)
    ax_mag.set_ylabel('B (nT)')
    ax_mag.legend(loc='upper right', fontsize=8)
    ax_mag.grid(True, alpha=0.3)
    B_all = np.concatenate([Bx, By, Bz])
    B_all = B_all[np.isfinite(B_all)]
    if len(B_all):
        pad = 0.05 * (B_all.max() - B_all.min()) or 1.0
        ax_mag.set_ylim(B_all.min() - pad, B_all.max() + pad)

    # Trim x-axis to actual data extent (no blank margins)
    ax_mag.set_xlim(t.iloc[0], t.iloc[-1])

    if show_loading:
        try:
            lbl_df = load_human_loading_labels()
            for _, row in lbl_df.iterrows():
                ev_start = pd.Timestamp(row['start'])
                ev_stop  = pd.Timestamp(row['stop'])
                if ev_stop < t0 or ev_start > t1:
                    continue
                for ax in axes:
                    ax.axvline(ev_start, color='green',  linestyle='--', lw=1.0, alpha=0.8)
                    ax.axvline(ev_stop,  color='red',    linestyle='--', lw=1.0, alpha=0.8)
                    if pd.notna(row.get('partition', pd.NaT)):
                        ax.axvline(pd.Timestamp(row['partition']),
                                   color='orange', linestyle='--', lw=1.0, alpha=0.8)
        except Exception:
            pass

    if show_kt17 and dBx is not None:
        ax_res = axes[1]
        ax_res.plot(t, dBx,   color='red',   lw=0.7, label='ΔBx')
        ax_res.plot(t, dBy,   color='green', lw=0.7, label='ΔBy')
        ax_res.plot(t, dBz,   color='blue',  lw=0.7, label='ΔBz')
        ax_res.plot(t, dBmag, color='black', lw=0.7, label='Δ|B|')
        ax_res.axhline(0, color='k', lw=0.4, alpha=0.4)
        ax_res.set_ylabel('ΔB (nT)', fontsize=8)
        ax_res.legend(loc='upper right', fontsize=7)
        ax_res.grid(True, alpha=0.3)
        dB_all = np.concatenate([dBx, dBy, dBz, dBmag])
        dB_all = dB_all[np.isfinite(dB_all)]
        if len(dB_all):
            pad = 0.05 * (dB_all.max() - dB_all.min()) or 1.0
            ax_res.set_ylim(dB_all.min() - pad, dB_all.max() + pad)

    if has_region:
        from matplotlib.patches import Patch
        ax_bar   = axes[1 + int(show_kt17)]
        type_arr = df['Type_num'].to_numpy()
        t_arr    = t.to_numpy().astype('datetime64[ns]').astype('int64')
        dt_h     = (t_arr[1] - t_arr[0]) // 2 if len(t_arr) > 1 else int(5e8)
        edges        = np.empty(len(t_arr) + 1, dtype='int64')
        edges[0]     = t_arr[0]  - dt_h
        edges[-1]    = t_arr[-1] + dt_h
        edges[1:-1]  = (t_arr[:-1] + t_arr[1:]) // 2
        t_edges      = edges.astype('datetime64[ns]')
        for k in range(len(type_arr)):
            rtype = int(type_arr[k]) if np.isfinite(type_arr[k]) else 0
            ax_bar.axvspan(t_edges[k], t_edges[k + 1],
                           color=_region_colors.get(rtype, 'lightgrey'), lw=0)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_yticks([])
        ax_bar.tick_params(left=False, bottom=False)
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        seen    = sorted(set(int(v) for v in type_arr if np.isfinite(v)))
        handles = [Patch(facecolor=_region_colors.get(r, 'lightgrey'),
                         label=_region_labels.get(r, str(r))) for r in seen]
        ax_bar.legend(handles=handles, loc='center left', fontsize=6,
                      ncol=len(handles), framealpha=0.0,
                      borderpad=0.2, handlelength=1, handleheight=0.8)

    if species:
        try:
            t_fips_ref = t0 + (t1 - t0) / 2  # midpoint avoids date-boundary failures
            fips_path = _fips_espec_path_for_date(t_fips_ref)
            fips      = load_fips_espec_tab(fips_path)
            t_fips    = fips['t'].astype('datetime64[ns]')
            t0_ns, t1_ns = np.datetime64(t0, 'ns'), np.datetime64(t1, 'ns')
            fmask    = (t_fips >= t0_ns) & (t_fips <= t1_ns)
            t_edges  = _fips_time_edges(t_fips[fmask].astype('int64'))

            fips_cmap = plt.cm.nipy_spectral.copy()
            fips_cmap.set_under('black')

            fips_axes = axes[1 + int(show_kt17) + int(has_region):]
            for ax_f, sp in zip(fips_axes, species):
                if fmask.sum() < 2:
                    ax_f.text(0.5, 0.5, f'No FIPS data ({sp})',
                              ha='center', va='center', transform=ax_f.transAxes)
                    continue
                flux    = fips[f'{sp}_flux'][fmask]
                energy  = fips[f'{sp}_energy']
                e_edges = _fips_bin_edges(energy)
                T, E    = np.meshgrid(t_edges, e_edges)
                ax_f.pcolormesh(T, E, flux.T,
                                cmap=fips_cmap,
                                norm=LogNorm(vmin=np.nanpercentile(flux[flux > 0], 5)
                                             if (flux > 0).any() else 1e-3),
                                shading='flat')
                ax_f.set_yscale('log')
                ax_f.set_ylabel(f'{sp}\nE (keV)', fontsize=8)
                ax_f.grid(True, alpha=0.2, color='white', lw=0.4)
        except Exception as e:
            ax_fips0 = axes[1 + int(show_kt17) + int(has_region)]
            ax_fips0.text(0.5, 0.5, f'FIPS unavailable: {e}',
                          ha='center', va='center',
                          transform=ax_fips0.transAxes, fontsize=7)

    smooth_str = f'  |  {smooth_sec}s smooth' if win > 1 else '  |  raw'
    fig.suptitle(
        f'{t0.strftime("%Y-%m-%d %H:%M")} - {t1.strftime("%H:%M")} UTC{smooth_str}',
        fontsize=10)

    # Ephemeris tick labels: UTC / X / Y / Z (R_M) on the bottom axis
    from matplotlib.dates import num2date
    ax_bot = axes[-1]
    fig.canvas.draw()
    xlim = ax_bot.get_xlim()
    tick_locs = [tk for tk in ax_bot.get_xticks() if xlim[0] <= tk <= xlim[1]]
    if tick_locs and {'ephx', 'ephy', 'ephz'}.issubset(df.columns):
        t_ns = t.astype('int64').to_numpy()
        ex   = df['ephx'].to_numpy(dtype=float)
        ey   = df['ephy'].to_numpy(dtype=float)
        ez   = df['ephz'].to_numpy(dtype=float)
        labels = []
        for tk in tick_locs:
            tk_ts = pd.Timestamp(num2date(tk).replace(tzinfo=None))
            idx   = int(np.argmin(np.abs(t_ns - np.int64(tk_ts.value))))
            labels.append(
                f"{tk_ts.strftime('%H:%M')}\n"
                f"X={ex[idx]:.2f}\n"
                f"Y={ey[idx]:.2f}\n"
                f"Z={ez[idx]:.2f}"
            )
        ax_bot.set_xticks(tick_locs)
        ax_bot.set_xticklabels(labels, fontsize=7, ha='center')
    ax_bot.set_xlabel(r'UTC  /  $X\ Y\ Z\ (R_M)$', fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig

# ---------------------------------------------------------------------------
# FIPS energy grid (instrument constant, 64 channels, low->high keV)
# ---------------------------------------------------------------------------
_FIPS_ENERGY_KEV = np.array([
    0.04572, 0.05004, 0.05478, 0.05996, 0.06563, 0.07183, 0.07861,
    0.08603, 0.09413, 0.10300, 0.11270, 0.12330, 0.13500, 0.14770,
    0.16160, 0.17680, 0.19350, 0.21180, 0.23180, 0.25360, 0.27760,
    0.30380, 0.33240, 0.36380, 0.39810, 0.43570, 0.47680, 0.52190,
    0.57120, 0.62520, 0.68420, 0.74880, 0.81960, 0.89720, 0.98220,
    1.07500, 1.17700, 1.28800, 1.40900, 1.54300, 1.68900, 1.84900,
    2.02400, 2.21600, 2.42600, 2.65500, 2.90600, 3.18100, 3.48200,
    3.81200, 4.17300, 4.56800, 5.00000, 5.47700, 5.99600, 6.56400,
    7.18700, 7.86700, 8.60900, 9.42500, 10.3200, 11.3000, 12.3700,
    13.5400,
], dtype='float32')

_FIPS_TAB_SPECIES = ['H+', 'He++', 'He+', 'Na-group', 'O-group']

# ---------------------------------------------------------------------------
# FIPS file path / download helpers
# ---------------------------------------------------------------------------
_FIPS_ESPEC_DIR     = os.path.join(_SCRIPT_DIR, 'FIPS')
_FIPS_METADEX_BASE  = 'https://pds-ppi.igpp.ucla.edu/metadex/product/select/'
_FIPS_DATA_BASE     = 'https://pds-ppi.igpp.ucla.edu'
_FIPS_COLLECTION_ID = 'urn:nasa:pds:mess-epps-fips-derived:data-espec'


def _fips_metadex_query(q, rows=10, fl=None):
    """Query the PPI metadex Solr API and return the docs list."""
    import urllib.parse
    params = {'q': q, 'version': '2.2', 'start': '0',
              'rows': str(rows), 'indent': 'on', 'wt': 'json'}
    if fl:
        params['fl'] = fl
    url = _FIPS_METADEX_BASE + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data['response']['docs']


def _fips_tab_info_for_tag(yyyydoy):
    """Query metadex for a day's TAB URL and UTC start time.

    Returns (tab_url, utc_start_str) or (None, None) on failure.
    """
    import datetime as _dt
    year = int(yyyydoy[:4])
    doy  = int(yyyydoy[4:])
    date = _dt.date(year, 1, 1) + _dt.timedelta(days=doy - 1)
    t0   = date.strftime('%Y-%m-%dT00:00:00Z')
    t1   = date.strftime('%Y-%m-%dT23:59:59Z')
    q    = (f'collection_id:"{_FIPS_COLLECTION_ID}"'
            f' AND start_date_time:[{t0} TO {t1}]')
    try:
        docs = _fips_metadex_query(q, rows=5, fl='slot,data_file,start_date_time')
        if docs and docs[0].get('slot') and docs[0].get('data_file'):
            d       = docs[0]
            tab_url = _FIPS_DATA_BASE + d['slot'] + '/' + d['data_file']
            return tab_url, d.get('start_date_time', '')
    except Exception:
        pass
    return None, None


def _fips_save_anchor(tab_local, utc_start_str):
    """Save <tab>.utc sidecar with the UTC of the first record."""
    with open(tab_local + '.utc', 'w') as f:
        f.write(utc_start_str.rstrip('Z'))


def _fips_espec_path_for_date(date):
    """Return local path for the FIPS ESPEC TAB file covering *date*.

    Downloads from PDS via the metadex API if the file is not already present.
    *date* may be a datetime, Timestamp, or datetime64.
    """
    dt  = pd.Timestamp(date)
    doy = dt.day_of_year
    tag = f'{dt.year}{doy:03d}'

    os.makedirs(_FIPS_ESPEC_DIR, exist_ok=True)
    existing = [f for f in os.listdir(_FIPS_ESPEC_DIR)
                if f.upper().startswith(f'FIPS_ESPEC_{tag}') and f.upper().endswith('.TAB')]
    if existing:
        return os.path.join(_FIPS_ESPEC_DIR, existing[0])

    tab_url, utc_str = _fips_tab_info_for_tag(tag)
    if tab_url is None:
        raise FileNotFoundError(
            f'Could not resolve PDS download URL for FIPS ESPEC day {tag}.')
    fname = tab_url.split('/')[-1]
    local = os.path.join(_FIPS_ESPEC_DIR, fname)
    print(f'Downloading {fname} ...', end=' ', flush=True)
    try:
        urllib.request.urlretrieve(tab_url, local)
        print('done.')
    except Exception as e:
        if os.path.exists(local):
            os.remove(local)
        raise FileNotFoundError(f'Download failed for {tab_url}: {e}') from e
    if utc_str:
        _fips_save_anchor(local, utc_str)
    return local


def download_all_fips_espec(overwrite=False):
    """Download the full mission FIPS ESPEC dataset from PDS into FIPS/.

    Skips files already present unless overwrite=True.
    """
    os.makedirs(_FIPS_ESPEC_DIR, exist_ok=True)
    print('Querying PDS metadex for full product list ...')
    docs = _fips_metadex_query(
        f'collection_id:"{_FIPS_COLLECTION_ID}"',
        rows=2000,
        fl='product_id,slot,data_file,start_date_time',
    )
    print(f'Found {len(docs)} products.')

    for i, doc in enumerate(docs, 1):
        slot      = doc.get('slot', '')
        data_file = doc.get('data_file', '')
        utc_str   = doc.get('start_date_time', '')
        if not slot or not data_file:
            print(f'[{i}/{len(docs)}] Missing slot/data_file - skipping.')
            continue
        local = os.path.join(_FIPS_ESPEC_DIR, data_file)
        if os.path.exists(local) and not overwrite:
            if utc_str and not os.path.exists(local + '.utc'):
                _fips_save_anchor(local, utc_str)
            print(f'[{i}/{len(docs)}] {data_file} - already present, skipping.')
            continue
        tab_url = _FIPS_DATA_BASE + slot + '/' + data_file
        print(f'[{i}/{len(docs)}] Downloading {data_file} ...', end=' ', flush=True)
        try:
            urllib.request.urlretrieve(tab_url, local)
            print('done.')
            if utc_str:
                _fips_save_anchor(local, utc_str)
        except Exception as e:
            print(f'FAILED: {e}')
            if os.path.exists(local):
                os.remove(local)
    print('Download complete.')

# ---------------------------------------------------------------------------
# FIPS data loading
# ---------------------------------------------------------------------------
def _fips_time_edges(t_ns):
    """(N+1,) datetime64[ns] bin edges from (N,) int64 nanosecond centres."""
    edges = np.empty(len(t_ns) + 1, dtype='int64')
    edges[1:-1] = (t_ns[:-1] + t_ns[1:]) // 2
    dt = int(np.median(np.diff(t_ns))) if len(t_ns) > 1 else int(60e9)
    edges[0]  = t_ns[0]  - dt // 2
    edges[-1] = t_ns[-1] + dt // 2
    return edges.astype('datetime64[ns]')


def _fips_bin_edges(centres):
    """(N+1,) log-spaced energy bin edges from (N,) bin centres."""
    log_c = np.log10(centres)
    dlog  = np.diff(log_c)
    edges = np.empty(len(centres) + 1)
    edges[1:-1] = 10 ** (0.5 * (log_c[:-1] + log_c[1:]))
    edges[0]    = 10 ** (log_c[0]  - 0.5 * dlog[0])
    edges[-1]   = 10 ** (log_c[-1] + 0.5 * dlog[-1])
    return edges


def load_fips_espec_tab(path):
    """
    Load a MESSENGER FIPS ESPEC DDR TAB file (PDS product).

    MET is converted to UTC via SPICE if available; otherwise falls back to
    a per-file UTC anchor sidecar written by _fips_espec_path_for_date.

    Returns
    -------
    dict with keys:
        't'           : datetime64[ns] (N,)
        '<sp>_flux'   : float32 (N, 64),  NaN where fill (<= 0)
        '<sp>_energy' : float32 (64,)     energy centres (keV, low->high)
      for sp in 'H+', 'He++', 'He+', 'Na-group', 'O-group'
    """
    raw = pd.read_csv(path, skiprows=4, header=None, sep=r'\s+', engine='python')
    met = raw.iloc[:, 1].to_numpy(dtype='float64')

    try:
        load_messenger_kernels()
        et_arr  = np.array([spice.sct2e(-236, m) for m in met])
        utc_arr = np.array([spice.et2utc(e, 'ISOC', 3) for e in et_arr],
                           dtype='datetime64[ns]')
    except Exception:
        anchor_path = path + '.utc'
        if os.path.exists(anchor_path):
            with open(anchor_path) as _f:
                _utc_str = _f.read().strip()
            _t0 = np.datetime64(_utc_str, 'ns').astype('int64')
        else:
            _t0 = np.datetime64('2012-08-16T00:00:14.973000000', 'ns').astype('int64')
        delta   = ((met - met[0]) * 1e9).astype('int64')
        utc_arr = (_t0 + delta).astype('datetime64[ns]')

    result = {'t': utc_arr}
    for i, sp in enumerate(_FIPS_TAB_SPECIES):
        col0 = 2 + i * 64
        # TAB stores channels high->low; reverse to match _FIPS_ENERGY_KEV (low->high)
        flux = raw.iloc[:, col0:col0 + 64].to_numpy(dtype='float32')[:, ::-1]
        flux[flux <= 0] = np.nan
        result[f'{sp}_flux']   = flux
        result[f'{sp}_energy'] = _FIPS_ENERGY_KEV.copy()
    return result

# ---------------------------------------------------------------------------
# FIPS plots
# ---------------------------------------------------------------------------
def plot_fips_espec_spectrogram(path, species=None, trange=None, orbit=None, save=True):
    """
    Plot FIPS differential-flux spectrograms from a PDS ESPEC TAB file.

    Parameters
    ----------
    path    : path to a FIPS_ESPEC_*_DDR_*.TAB file
    species : list of species to plot; default ['H+']
    trange  : [t0, t1] strings/Timestamps to restrict the window
    orbit   : int orbit number; overrides trange if given
    save    : save a PNG to figures/

    Returns matplotlib Figure.
    """
    if species is None:
        species = ['H+']

    if orbit is not None:
        orb_df = load_bowers_data_pkl(orbit_number=orbit)
        t_obs  = pd.to_datetime(orb_df['time'])
        trange = [t_obs.iloc[0], t_obs.iloc[-1]]

    data  = load_fips_espec_tab(path)
    t_dt  = data['t'].astype('datetime64[ns]')

    if trange is not None:
        t0 = np.datetime64(pd.Timestamp(trange[0]), 'ns')
        t1 = np.datetime64(pd.Timestamp(trange[1]), 'ns')
        mask = (t_dt >= t0) & (t_dt <= t1)
        if mask.sum() < 2:
            raise ValueError(f'trange {trange} contains fewer than 2 FIPS samples.')
        t_dt = t_dt[mask]
        data = {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim == 2 else v)
                for k, v in data.items()}
        data['t'] = t_dt

    t_ns    = t_dt.astype('int64')
    t_edges = _fips_time_edges(t_ns)
    cmap    = plt.cm.nipy_spectral.copy()

    fig, axes = plt.subplots(len(species), 1,
                             figsize=(14, 3 * len(species)),
                             sharex=True,
                             gridspec_kw={'hspace': 0.06})
    if len(species) == 1:
        axes = [axes]

    for ax, sp in zip(axes, species):
        flux    = data[f'{sp}_flux']
        energy  = data[f'{sp}_energy']
        e_edges = _fips_bin_edges(energy)
        T, E    = np.meshgrid(t_edges, e_edges)
        pcm = ax.pcolormesh(T, E, flux.T,
                            cmap=cmap,
                            norm=plt.matplotlib.colors.LogNorm(vmin=1e6, vmax=1e9),
                            shading='flat')
        ax.set_yscale('log')
        ax.set_ylabel('Energy (keV)', fontsize=9)
        ax.set_ylim(e_edges[0], e_edges[-1])
        cb = fig.colorbar(pcm, ax=ax, pad=0.005, fraction=0.015)
        cb.set_label(r'Flux (cm$^{-2}$ s$^{-1}$ keV$^{-1}$ sr$^{-1}$)', fontsize=6)
        ax.text(0.005, 0.96, sp, transform=ax.transAxes, fontsize=10,
                va='top', fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc='k', alpha=0.45))
        ax.grid(True, alpha=0.15, color='white', lw=0.4)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    t0_str = str(t_dt[0])[:10]
    t1_str = str(t_dt[-1])[:10]
    axes[-1].set_xlabel(
        f'UTC {t0_str}' if t0_str == t1_str else f'UTC {t0_str} - {t1_str}',
        fontsize=9)
    date_tag = os.path.basename(path).split('_')[2]
    title    = f'MESSENGER FIPS ESPEC - {date_tag}'
    if orbit is not None:
        title += f'  (orbit {orbit})'
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if save:
        os.makedirs('figures', exist_ok=True)
        sp_tag  = '_'.join(s.replace('+', 'p').replace('-', '') for s in species)
        orb_tag = f'_orb{orbit}' if orbit is not None else ''
        out     = os.path.join('figures', f'fips_espec_{date_tag}{orb_tag}_{sp_tag}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')
    return fig

def plot_fips_for_orbit(orb, species=None, save=True):
    """Plot FIPS spectrograms for the date of *orb*, downloading if needed."""
    orb_df = load_bowers_data_pkl(orbit_number=orb)
    t_obs  = pd.to_datetime(orb_df['time'])
    t_mid  = t_obs.iloc[len(t_obs) // 2]  # midpoint avoids date-boundary failures
    path   = _fips_espec_path_for_date(t_mid)
    return plot_fips_espec_spectrogram(path, species=species, orbit=orb, save=save)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
# Quick look at a time range (mag + H+ spectrogram):
#fig = plot_quick_look('2015-04-14/19:22:06', '2015-04-14/19:26:04')
#fig = plot_quick_look(orbit = 2493, only_cs=True, show_loading=True, show_kt17=True, save_path="figures/quicklook.png")

# Load MAG data and compute FAC components:
#   df = load_bowers_data_pkl(orbit_number=3451)
#   _, Bxm, Bym, Bzm = get_kt17_along_track(df=df)
#   B_perp, B_phi, B_par = transform_to_fac(
#       df['magx'], df['magy'], df['magz'], Bxm, Bym, Bzm,
#       df['ephx'], df['ephy'], df['ephz'])

# FIPS for a specific orbit:
#plot_fips_for_orbit(3941)



# ---------------------------------------------------------------------------
# Analyze loading B_phi e.g. first in orbit 2493
# 'human_loading_labels_partitioned.parquet' stores the pre-built event data, delete it to rebuild if new events are added
# and 'unloading_fac_cache.npz' is for the computed delta B_phi
# ---------------------------------------------------------------------------

_EVENTS_CACHE = os.path.join(_SCRIPT_DIR, 'human_loading_labels_partitioned.parquet')
if os.path.exists(_EVENTS_CACHE):
    events = pd.read_parquet(_EVENTS_CACHE)
    print(f'Loaded cached event table ({len(events)} events) from {_EVENTS_CACHE}')
else:
    events = load_human_loading_labels()
    events.to_parquet(_EVENTS_CACHE, index=False)
    print(f'Saved event table to {_EVENTS_CACHE}')

_UNLOADING_CACHE = os.path.join(_SCRIPT_DIR, 'unloading_fac_cache.npz')

if os.path.exists(_UNLOADING_CACHE):
    _cache = np.load(_UNLOADING_CACHE, allow_pickle=True)
    unloading_data = _cache['unloading_data'].item()
    print(f'Loaded cached FAC data ({len(unloading_data)} events) from {_UNLOADING_CACHE}')
else:
    unloading_data = {}
    _orbit_counters = {}

    for _, event_data in events.iterrows():

        orbit     = event_data['orbit']
        start     = event_data['start']
        partition = event_data['partition']
        stop      = event_data['stop']

        df_event    = load_bowers_data_pkl(trange=[start, stop])
        t_event     = pd.to_datetime(df_event['time'])
        i_partition = int((t_event - partition).abs().argmin())

        df_loading  = df_event.iloc[:i_partition]
        Bx0 = df_loading['magx'].mean()
        By0 = df_loading['magy'].mean()
        Bz0 = df_loading['magz'].mean()

        df_unloading = df_event.iloc[i_partition:]
        n = len(df_unloading)
        B_perp_unload, B_phi_unload, B_par_unload = transform_to_fac(
            df_unloading['magx'], df_unloading['magy'], df_unloading['magz'],
            np.full(n, Bx0), np.full(n, By0), np.full(n, Bz0),
            df_unloading['ephx'], df_unloading['ephy'], df_unloading['ephz'])
        
        DeltaB_perp_unload = B_perp_unload - B_perp_unload[0]
        DeltaB_phi_unload = B_phi_unload - B_phi_unload[0]
        DeltaB_par_unload = B_par_unload - B_par_unload[0]

        letter = chr(ord('a') + _orbit_counters.get(orbit, 0))
        _orbit_counters[orbit] = _orbit_counters.get(orbit, 0) + 1
        key = f'{int(orbit)}{letter}'

        unloading_data[key] = {
            'orbit':  orbit,
            'time':   pd.to_datetime(df_unloading['time']).to_numpy(),
            'x':      df_unloading['ephx'].to_numpy(),
            'y':      df_unloading['ephy'].to_numpy(),
            'z':      df_unloading['ephz'].to_numpy(),
            'B_phi':  B_phi_unload,
            'B_perp': B_perp_unload,
            'B_par':  B_par_unload,
            'DeltaB_phi':  DeltaB_phi_unload,
            'DeltaB_perp': DeltaB_perp_unload,
            'DeltaB_par':  DeltaB_par_unload,
        }

    np.savez(_UNLOADING_CACHE, unloading_data=unloading_data)
    print(f'Saved FAC cache ({len(unloading_data)} events) to {_UNLOADING_CACHE}')

# ---------------------------------------------------------------------------
# Plot all unloading B_phi as colored line segments in lat/lon
# ---------------------------------------------------------------------------
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

vmax=5

norm = plt.Normalize(-vmax, vmax)

fig_map = plt.figure(figsize=(10, 9))
gs = GridSpec(2, 2, figure=fig_map, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
ax_top  = fig_map.add_subplot(gs[0, :])   # lat/lon — full width
ax_ypos = fig_map.add_subplot(gs[1, 0])   # XZ, Y > 0
ax_yneg = fig_map.add_subplot(gs[1, 1])   # XZ, Y < 0

def _add_segments(ax, pts, c):
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    cv   = 0.5 * (c[:-1] + c[1:])
    lc   = LineCollection(segs, cmap='RdBu_r', norm=norm, linewidth=2, alpha=0.8)
    lc.set_array(cv)
    ax.add_collection(lc)

for key, ev in unloading_data.items():
    x, y, z  = ev['x'], ev['y'], ev['z']
    bphi     = ev['DeltaB_phi']

    # Top panel: lat / lon
    r   = np.sqrt(x**2 + y**2 + z**2)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lon = np.degrees(np.arctan2(y, x)) % 360
    _add_segments(ax_top, np.column_stack([lon, lat]), bphi)

    # Bottom panels: XZ plane, split by sign of Y
    for mask, ax in [(y > 0, ax_ypos), (y < 0, ax_yneg)]:
        if mask.sum() < 2:
            continue
        _add_segments(ax, np.column_stack([x[mask], z[mask]]), bphi[mask])

ax_top.set_xlim(100, 260)
ax_top.set_ylim(-60, 60)
ax_top.set_xlabel('Longitude (deg, MSM)')
ax_top.set_ylabel('Latitude (deg, MSM)')
ax_top.set_title(r'Unloading $\Delta B_\phi$ — lat/lon')
ax_top.grid(True, alpha=0.3)

for ax, title in [(ax_ypos, 'XZ plane  (Y > 0)'), (ax_yneg, 'XZ plane  (Y < 0)')]:
    mercury = plt.Circle((0, -0.2), 1.0, color='grey', zorder=0, alpha=0.5)
    ax.add_patch(mercury)
    ax.set_xlabel('X (R$_M$, MSM)')
    ax.set_ylabel('Z (R$_M$, MSM)')
    ax.set_title(title)
    ax.axhline(0, color='k', lw=0.4, alpha=0.4)
    ax.axvline(0, color='k', lw=0.4, alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.autoscale()
    ax.set_aspect(1)

ax_ypos.set_xlim(right=0)
ax_yneg.set_xlim(right=0)
ax_ypos.invert_xaxis()

sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])
fig_map.colorbar(sm, ax=[ax_top, ax_ypos, ax_yneg], label=r'$\Delta B_\phi$ (nT)',
                 shrink=0.6, pad=0.02)

fig_map.savefig(os.path.join(_SCRIPT_DIR, 'figures', 'unloading_Bphi_map.png'),
                dpi=150, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# Interpolated contourf map — mean B_phi per event in lat/lon
# ---------------------------------------------------------------------------
from scipy.interpolate import griddata

# Collect one (lon, lat, mean_bphi) point per event
pts_lon, pts_lat, pts_val = [], [], []
for ev in unloading_data.values():
    x, y, z = ev['x'], ev['y'], ev['z']
    r   = np.sqrt(x**2 + y**2 + z**2)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lon = np.degrees(np.arctan2(y, x)) % 360
    pts_lon.append(np.mean(lon))
    pts_lat.append(np.mean(lat))
    pts_val.append(np.mean(ev['DeltaB_phi']))

pts_lon = np.array(pts_lon)
pts_lat = np.array(pts_lat)
pts_val = np.array(pts_val)

# Regular grid covering the same lon/lat window
grid_lon, grid_lat = np.meshgrid(np.linspace(100, 260, 200),
                                  np.linspace(-60,  60, 150))
# Linear inside convex hull; fill remaining NaNs with nearest-neighbour
grid_val = griddata((pts_lon, pts_lat), pts_val,
                    (grid_lon, grid_lat), method='linear')
nan_mask = np.isnan(grid_val)
if nan_mask.any():
    grid_val[nan_mask] = griddata((pts_lon, pts_lat), pts_val,
                                  (grid_lon[nan_mask], grid_lat[nan_mask]),
                                  method='nearest')

fig_interp, ax_interp = plt.subplots(figsize=(10, 5))
cf = ax_interp.contourf(grid_lon, grid_lat, grid_val,
                        levels=np.linspace(-vmax, vmax, 21),
                        cmap='RdBu_r', extend='both')
ax_interp.scatter(pts_lon, pts_lat, c=pts_val, cmap='RdBu_r',
                  vmin=-vmax, vmax=vmax, s=20, edgecolors='k', lw=0.4, zorder=3)
ax_interp.set_xlim(100, 260)
ax_interp.set_ylim(-60, 60)
ax_interp.set_xlabel('Longitude (deg, MSM)')
ax_interp.set_ylabel('Latitude (deg, MSM)')
ax_interp.set_title(r'Interpolated mean $\Delta B_\phi$ (nT)')
ax_interp.grid(True, alpha=0.3)
fig_interp.colorbar(cf, ax=ax_interp, label=r'$\Delta B_\phi$ (nT)')
fig_interp.tight_layout()
fig_interp.savefig(os.path.join(_SCRIPT_DIR, 'figures', 'unloading_Bphi_contourf.png'),
                   dpi=150, bbox_inches='tight')
plt.show()

'''

# Test
for _, event_data in events.iterrows():

    orbit     = event_data['orbit']
    start     = event_data['start']
    partition = event_data['partition']
    stop      = event_data['stop']

    df_event    = load_bowers_data_pkl(trange=[start, stop])
    t_event     = pd.to_datetime(df_event['time'])
    i_partition = int((t_event - partition).abs().argmin())

    df_loading  = df_event.iloc[:i_partition]
    Bx0 = df_event['magx'].mean()
    By0 = df_event['magy'].mean()
    Bz0 = df_event['magz'].mean()

    df_unloading = df_event.iloc[i_partition:]
    n = len(df_unloading)
    B_perp_unload, B_phi_unload, B_par_unload = transform_to_fac(
        df_unloading['magx'], df_unloading['magy'], df_unloading['magz'],
        np.full(n, Bx0), np.full(n, By0), np.full(n, Bz0),
        df_unloading['ephx'], df_unloading['ephy'], df_unloading['ephz'])
    
    deltaB_perp_unload = B_perp_unload - B_perp_unload[0]
    deltaB_phi_unload = B_phi_unload - B_phi_unload[0]
    deltaB_par_unload = B_par_unload - B_par_unload[0]

    plt.plot(df_event['time'],df_event['magx'])
    plt.plot(df_event['time'],df_event['magy'])
    plt.plot(df_event['time'],df_event['magz'])
    plt.axvline(x = partition)
    plt.axhline(y=By0,color='tab:orange')
    #plt.show()

    #plt.plot(df_unloading ['time'], B_perp_unload)
    plt.plot(df_unloading ['time'], deltaB_phi_unload)
    #plt.plot(df_unloading ['time'], B_par_unload)
    plt.show()

    break

'''