#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MESSENGER magnetometer and FIPS particle data — core utilities.

Functions
---------
load_bowers_data_pkl        Load MAG data (time, Bx/By/Bz, ephemeris, region)
get_kt17_along_track        Evaluate KT173 model field at MESSENGER positions
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
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

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

#print(load_bowers_data_pkl(trange=['2015-04-05/01:50:00', '2015-04-05/01:52:00']))
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
        (orb_df['ephz'] > -1.0) &
        (orb_df['ephz'] < 0.8) &
        ((orb_df['ephx']**2 + orb_df['ephy']**2 + orb_df['ephz']**2) < 4**2)
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
# Trajectory export
# ---------------------------------------------------------------------------
def extract_traj(orbit, save_path=None, events_json=None):
    """
    Save the nightside current-sheet segment of an orbit to a CSV file.

    Columns
    -------
    t_datetime  : ISO-8601 timestamp string
    t_epoch     : seconds since the segment start (float, starts at 0)
    x, y, z     : position in R_M, MSM
    bx, by, bz  : magnetic field in nT, MSM

    The output filename embeds the labelled loading-event start time (read
    from events_json) so downstream scripts can locate the substorm onset
    without re-parsing the labels file.

    Parameters
    ----------
    orbit       : int orbit number
    save_path   : output CSV path (default: <script_dir>/trajectories/orbit_<orbit>_start_<ISO>_traj.csv,
                  or .../orbit_<orbit>_traj.csv if no labelled event is found)
    events_json : path to a loading-events JSON (default: <script_dir>/equatorial_events.json);
                  uses the first event's 'start' for the orbit, if present

    Returns the DataFrame that was written.
    """
    orb_df = load_bowers_data_pkl(orbit_number=orbit)
    seg = filter_orbit_segment(orb_df)
    if seg.empty:
        raise ValueError(f'No nightside current-sheet segment found for orbit {orbit}.')

    t = pd.to_datetime(seg['time'])
    t_epoch = (t - t.iloc[0]).dt.total_seconds().to_numpy()

    out = pd.DataFrame({
        't_datetime': t.dt.strftime('%Y-%m-%dT%H:%M:%S.%f'),
        't_epoch':    t_epoch,
        'x':          seg['ephx'].to_numpy(),
        'y':          seg['ephy'].to_numpy(),
        'z':          seg['ephz'].to_numpy(),
        'bx':         seg['magx'].to_numpy(),
        'by':         seg['magy'].to_numpy(),
        'bz':         seg['magz'].to_numpy(),
    })

    if events_json is None:
        events_json = os.path.join(_SCRIPT_DIR, 'equatorial_events.json')
    event_start = None
    if os.path.exists(events_json):
        with open(events_json) as f:
            events = json.load(f).get(str(orbit), {}).get('loading_events', [])
        if events:
            event_start = pd.Timestamp(events[0]['start'])

    if save_path is None:
        out_dir = os.path.join(_SCRIPT_DIR, 'trajectories')
        os.makedirs(out_dir, exist_ok=True)
        if event_start is not None:
            tag = event_start.strftime('%Y%m%dT%H%M%S')
            save_path = os.path.join(out_dir, f'orbit_{orbit}_start_{tag}_traj.csv')
        else:
            save_path = os.path.join(out_dir, f'orbit_{orbit}_traj.csv')

    out.to_csv(save_path, index=False)
    print(f'Saved: {save_path}  ({len(out)} points)')
    return out

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
                    show_kt17_residual=None,
                    save_path=None, df=None, ylim_mag=None,
                    show_inset=True, df_full=None, _show=True):
    """
    Overview plot for an arbitrary time window or orbit number.

    Rows (top to bottom):
      - Bx / By / Bz / |B| (nT, optionally smoothed) [+ dashed KT17 model if show_kt17=True]
      - ΔBx / ΔBy / ΔBz (obs − KT17) with zero line  [if show_kt17_residual=True]
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
    show_kt17     : if True, overlay the dashed KT17 model field in the top mag panel
    show_kt17_residual : if True, add a residual panel (obs − KT17) below the mag panel;
                    defaults to the value of show_kt17 (old behaviour) when left as None
    species       : FIPS species to show, e.g. ('H+',) or ('H+', 'He++').
                    Pass () for mag-only.
    smooth_sec    : boxcar smoothing window in seconds (0 or None for raw)
    df            : pre-loaded Bowers DataFrame; skips all data loading if provided

    Returns matplotlib Figure.
    """
    from matplotlib.colors import LogNorm

    if df is not None:
        t_pre = pd.to_datetime(df['time'])
        t0, t1 = t_pre.iloc[0], t_pre.iloc[-1]
    elif orbit is not None:
        df = load_bowers_data_pkl(orbit_number=orbit)
        if only_cs:
            df = filter_orbit_segment(df)
            if df.empty:
                raise ValueError(f'No current-sheet crossing found for orbit {orbit}.')
        t_pre = pd.to_datetime(df['time'])
        t0, t1 = t_pre.iloc[0], t_pre.iloc[-1]
    elif t0 is None or t1 is None:
        raise ValueError('Provide either orbit=, t0/t1, or df=.')
    else:
        t0 = pd.Timestamp(t0)
        t1 = pd.Timestamp(t1)
        df = load_bowers_data_pkl(trange=[t0, t1])

    t0 = pd.Timestamp(t0)
    t1 = pd.Timestamp(t1)
    t  = pd.to_datetime(df['time'])

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
    if show_kt17_residual is None:
        show_kt17_residual = show_kt17
    dBx = dBy = dBz = dBmag = None
    Bxm_plot = Bym_plot = Bzm_plot = Bmagm_plot = None
    if show_kt17 or show_kt17_residual:
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
            show_kt17_residual = False

    _region_colors = {
        1: '#4477AA', 2: '#66CCEE', 3: '#CCBB44', 4: '#EE6677', 5: '#AA3377',
    }
    _region_labels = {1: 'MS', 2: 'MSH', 3: 'SW', 4: 'BS', 5: 'MP'}
    has_region = 'Type_num' in df.columns

    n_fips   = len(species)
    nrows    = 1 + int(show_kt17_residual) + int(has_region) + n_fips
    h_ratios = [3] + ([2] if show_kt17_residual else []) + ([0.12] if has_region else []) + [1] * n_fips
    fig, axes = plt.subplots(nrows, 1, sharex=True,
                             figsize=(figsize[0], figsize[1] + int(show_kt17_residual) * 2 + n_fips * 1.5),
                             gridspec_kw={'hspace': 0.05, 'height_ratios': h_ratios})
    axes = list(np.atleast_1d(axes))
    fig._data_axes = axes   # expose to callers (excludes inset / button axes)

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
    ax_mag.legend(loc='lower right', fontsize=8)
    ax_mag.grid(True, alpha=0.3)
    if ylim_mag is not None:
        ax_mag.set_ylim(ylim_mag)
    else:
        B_all = np.concatenate([Bx, By, Bz, Bmag])
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

    if show_kt17_residual and dBx is not None:
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
        ax_bar   = axes[1 + int(show_kt17_residual)]
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

            fips_axes = axes[1 + int(show_kt17_residual) + int(has_region):]
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
            ax_fips0 = axes[1 + int(show_kt17_residual) + int(has_region)]
            ax_fips0.text(0.5, 0.5, f'FIPS unavailable: {e}',
                          ha='center', va='center',
                          transform=ax_fips0.transAxes, fontsize=7)

    smooth_str  = f'  |  {smooth_sec}s smooth' if win > 1 else '  |  raw'
    orbit_str   = f'Orbit {orbit}  —  ' if orbit is not None else ''
    fig.suptitle(
        f'{orbit_str}{t0.strftime("%Y-%m-%d %H:%M")} - {t1.strftime("%H:%M")} UTC{smooth_str}',
        fontsize=10)

    # Ephemeris tick labels: UTC / X / Y / Z (R_M) on the bottom axis
    from matplotlib.dates import num2date
    ax_bot = axes[-1]
    fig.canvas.draw()
    xlim = ax_bot.get_xlim()
    tick_locs = [tk for tk in ax_bot.get_xticks() if xlim[0] <= tk <= xlim[1]]
    if tick_locs and {'ephx', 'ephy', 'ephz'}.issubset(df.columns):
        t_arr = t.to_numpy().astype('datetime64[ns]')
        ex    = df['ephx'].to_numpy(dtype=float)
        ey    = df['ephy'].to_numpy(dtype=float)
        ez    = df['ephz'].to_numpy(dtype=float)
        labels = []
        for tk in tick_locs:
            tk_ts = pd.Timestamp(num2date(tk).replace(tzinfo=None))
            tk_dt = np.datetime64(tk_ts, 'ns')
            idx   = int(np.clip(np.searchsorted(t_arr, tk_dt), 0, len(t_arr) - 1))
            labels.append(
                f"{tk_ts.strftime('%H:%M')}\n"
                f"X={ex[idx]:.2f}\n"
                f"Y={ey[idx]:.2f}\n"
                f"Z={ez[idx]:.2f}"
            )
        ax_bot.set_xticks(tick_locs)
        ax_bot.set_xticklabels(labels, fontsize=7, ha='center')
    ax_bot.set_xlabel(r'UTC  /  $X\ Y\ Z\ (R_M)$', fontsize=8)

    # ── YZ orbit inset (top-right of ax_mag) ─────────────────────────────────
    if show_inset and {'ephy', 'ephz'}.issubset(df.columns):
        if df_full is not None:
            _ctx = df_full
        elif orbit is not None:
            _ctx = load_bowers_data_pkl(orbit_number=orbit)
        else:
            _ctx = df
        Xf = _ctx['ephx'].to_numpy() if 'ephx' in _ctx.columns else np.zeros(len(_ctx))
        Yf = _ctx['ephy'].to_numpy()
        Zf = _ctx['ephz'].to_numpy()
        Ys = df['ephy'].to_numpy()
        Zs = df['ephz'].to_numpy()

        ax_in = ax_mag.inset_axes([0.88, 0.82, 0.21, 0.36])
        day   = Xf > 0
        night = ~day
        ax_in.plot(np.ma.masked_where(night, Yf), np.ma.masked_where(night, Zf),
                   color='steelblue', lw=0.6, zorder=0)
        ax_in.add_patch(plt.Circle((0, -0.2), 1.0, color='white',       zorder=1))
        ax_in.add_patch(plt.Circle((0, -0.2), 1.0, color='saddlebrown', alpha=0.35, zorder=2))
        ax_in.plot(np.ma.masked_where(day, Yf), np.ma.masked_where(day, Zf),
                   color='steelblue', lw=0.6, zorder=3)
        ax_in.plot(Ys, Zs, color='gold', lw=1.2, zorder=4)
        ax_in.scatter(Yf[0],  Zf[0],  marker='o', s=8, color='lime',   zorder=5)
        ax_in.scatter(Yf[-1], Zf[-1], marker='s', s=8, color='tomato', zorder=5)
        ax_in.set_xlim(1.5, -1.5)
        ax_in.set_ylim(-2, 1)
        ax_in.set_aspect('equal')
        ax_in.tick_params(labelsize=4, length=2, pad=1)
        ax_in.set_xlabel('Y (R$_M$)', fontsize=4, labelpad=1)
        ax_in.set_ylabel('Z (R$_M$)', fontsize=4, labelpad=1)
        ax_in.grid(True, alpha=0.2, lw=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if _show:
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
# Jiutong's wavelet code
# ---------------------------------------------------------------------------

def wavelet_coef_psd(time: np.ndarray, signal: np.ndarray, scales: np.ndarray, bandwidth: float = 6.0, downsample: int = 1, downsample_signal: bool = True):
    """
    Compute complex Morlet wavelet coefficients and power spectral density (PSD).
    Complex Morlet wavelet can be written as:

    Parameters:
    ----------
    time : np.ndarray
        Time vector of shape (Nt,). Can be in seconds or `np.datetime64`.
    signal : np.ndarray
        Input signal of shape (Nt,).
    scales : np.ndarray
        Wavelet scales corresponding to frequencies of shape (Nf,).
    bandwidth : float, optional
        bandwidthandwidth parameter for the Morlet wavelet (default is 6.0).
    downsample : int, optional
        Downsampling factor for the output (default is 1, meaning no downsampling).
    downsample_signal : bool, optional
        If True, downsample the input signal before computing wavelet coefficients (default is False).
        If False, the signal is not downsampled, but the output coefficients and PSD are downsampled.

    Returns:
    -------
    time : np.ndarray
        Time vector after optional downsampling.
    frequency : np.ndarray
        Frequencies corresponding to the wavelet scales.
    coef : np.ndarray
        Complex wavelet coefficients of shape (Nf, Nt), where Nf is the number of frequencies.
    psd : np.ndarray
        Power spectral density of shape (Nf, Nt).
    signal : np.ndarray
        Downsampled (moving average) signal if downsampling is applied.

    Notes:
    -----
    - The function uses `scipy.signal.cwt` with the Morlet wavelet by default.
    - `scipy.signal.cwt` is deprecated in SciPy 1.12 and will be removed in SciPy 1.15. Alternatives like PyWavelets or ssqueezepy can be used.
    - The power spectral density (PSD) is computed as the squared magnitude of the wavelet coefficients, scaled by `2 * dt`.
    - Downsampling is applied to the time, coefficients, PSD, and signal if `downsample > 1`.
    """

    if downsample_signal:
        time = time[::downsample]
        signal = signal[::downsample]
        downsample = 1

    if isinstance(time[0], np.datetime64):
        elapsed_time = np.array(time).astype('datetime64[ns]').astype('float') / 1e9
    else:
        elapsed_time = np.array(time)

    dt = elapsed_time[1] - elapsed_time[0]

    # === Option 1: scipy.signal implementation ===
    # bandwidthut, scipy.signal.cwt is deprecated in SciPy 1.12 and will be removed in SciPy 1.15. 
    # They recommend using PyWavelets instead.
    # https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.signal.cwt.html
    # However, as you can see in the bottom code block, pywt.cwt is actually problematic for the Morlet wavelet.

    # widths = bandwidth * scales / (2 * np.pi)
    # coef = scipy.signal.cwt(
    #     signal,
    #     scipy.signal.morlet2,
    #     widths = widths,
    #     w = bandwidth,
    #     dtype = np.complex128
    # )
    # frequency = 1 / dt / scales

    # === Option 2: pywt implementation ===
    # Unlike scipy.signal.cwt, pywt.cwt is not L2-normalized.
    # You need to multiply the coefficients by a factor to make them L2-normalized.
    # Another two problems with pywt.cwt: The precision of the wavelet may be unenough while you can still not adjust the precision yourself.
    # Check https://github.com/PyWavelets/pywt/issues/531
    # This defect has been proposed at 2019 but still not fixed yet.

    # central_frequency = 1.0
    # wavelet = 'cmor%.1f-%.1f' % (bandwidth, central_frequency)
    # coef, frequency = pywt.cwt(signal, scales, wavelet, dt, method = 'fft')
    # coef *= np.sqrt(np.sqrt(bandwidth) * np.sqrt(2 * np.pi))  # amplitude normalization for Morlet

    # === Option 3: ssqueezepy implementation (default used here) ===
    # This one is accurate and fast. They claim that this package is the fastest implementation of the wavelet transform in Python.
    # bandwidthut, this implementation is not elegantly designed and the input parameters are not well documented.
    # Also, it requires package numba, which may raise some compatibility issues.

    import ssqueezepy
    coef, scales = ssqueezepy.cwt(signal, ('morlet', {'mu': bandwidth}), scales = bandwidth / (2 * np.pi) * scales.astype(np.float32), fs = 1 / dt, l1_norm = False)
    frequency = bandwidth / (2 * np.pi) / dt / scales

    psd = (np.abs(coef) ** 2) * (2 * dt)


    return time[::downsample], frequency, coef[:, ::downsample], psd[:, ::downsample], signal[::downsample]

def wfft_coef_psd(time: np.ndarray, signal: np.ndarray, step: int = 1, window: int = 120):
    """
    Compute short-time Fourier transform (STFT) coefficients and power spectral density (PSD) using a sliding Hanning window.

    Parameters:
    ----------
    time : np.ndarray
        Time vector of shape (Nt,). Can be in seconds or `np.datetime64`.
    signal : np.ndarray
        Input signal of shape (Nt,).
    step : int, optional
        Step size for sliding the window (default is 1).
    window : int, optional
        Window length in samples (default is 120).

    Returns:
    -------
    wtime : np.ndarray
        Center time for each window after sliding.
    freq : np.ndarray
        Frequency vector corresponding to the FFT.
    coef : np.ndarray
        Complex FFT coefficients of shape (Nf, Nt), where Nf is the number of frequencies.
    psd : np.ndarray
        Power spectral density of shape (Nf, Nt).
    wsignal : np.ndarray
        Window-averaged signal of shape (Nt,).

    Notes:
    -----
    - The Hanning window is applied to each segment, and normalization is performed based on Parseval's theorem.
    - The PSD is computed as the squared magnitude of the FFT coefficients, scaled by `2 * dt / window`.
    """

    if isinstance(time[0], np.datetime64):
        elapsed_time = np.array(time).astype('datetime64[ns]').astype('float') / 1e9
    else:
        elapsed_time = np.array(time)

    dt = elapsed_time[1] - elapsed_time[0]

    # Apply sliding window view
    wtime = np.lib.stride_tricks.sliding_window_view(elapsed_time, window)[::step][:, 0] + dt * window / 2
    freq = np.fft.fftfreq(window, dt)[:window // 2]
    wsignal = np.lib.stride_tricks.sliding_window_view(signal, window)[::step]

    # Apply Hanning window and normalize based on Parseval's theorem
    wsignal = wsignal * np.sqrt(8 / 3) * np.hanning(window)

    coef = np.fft.fft(wsignal, axis=-1)[:, :window // 2].T
    psd = (np.abs(coef) ** 2) * 2 * dt / window

    if isinstance(time[0], np.datetime64):
        wtime = (np.array(wtime) * 1e9).astype('datetime64[ns]')

    wsignal = np.mean(wsignal, axis=-1)

    return wtime, freq, coef, psd, wsignal

def svd_wave_analysis(coef: np.ndarray, freq_window: int = 5, time_window: int = 5):
    """
    Perform SVD-based wave polarization analysis to compute planarity, ellipticity, and coherence.

    Parameters:
    ----------
    coef : np.ndarray
        Complex coefficient tensor of shape (Nf, Nt, 3), where Nf is the number of frequencies, Nt is the number of time points, and 3 represents the 3 components.
    freq_window : int, optional
        Frequency-domain smoothing window size (default is 5).
    time_window : int, optional
        Time-domain smoothing window size (default is 5).

    Returns:
    -------
    planarity : np.ndarray
        Planarity of the wave of shape (Nf, Nt), defined as `1 - sqrt(s3 / s1)`, where s1 and s3 are the largest and smallest singular values.
    ellipticity_along_k: np.ndarray
        Ellipticity along the wave vector direction of shape (Nf, Nt), defined as the ratio of the second to the first singular value.
    coherence : np.ndarray
        Coherence between the first and second principal components (along vh1 and vh2) of shape (Nf, Nt), computed from the smoothed wavefield spectrum.
    degree_of_polarization : np.ndarray
        3D Degree of polarization of shape (Nf, Nt), defined as `sqrt[3 / 2 * tr(J^2) / tr^2(J) - 1 / 2]`, computed using the wavefield spectrum.
    vh : np.ndarray
        Right singular vectors of shape (Nf, Nt, 3, 3), representing the polarization basis.

    Notes:
    -----
    - The input coefficients are smoothed in both frequency and time domains before performing SVD.
    - The coherence is computed from the wavefield spectrum in the transformed basis.
    """

    spec = np.einsum('ijk,ijl->ijkl', coef, coef.conj())
    spec = bn.move_mean(spec, window=freq_window, min_count=1, axis=0)
    spec = bn.move_mean(spec, window=time_window, min_count=1, axis=1)

    spec_63 = np.concatenate([spec.real, spec.imag], axis=-2)
    u, s, vh = np.linalg.svd(spec_63, full_matrices=False)

    planarity = 1 - np.sqrt(s[:, :, 2] / s[:, :, 0])
    ellipticity_along_k = s[:, :, 1] / s[:, :, 0]

    # Rotate the coefficients to the wave frame, in which the third component is the least significant
    coef_wf = np.einsum('ijk,ijlk->ijl', coef, vh)
    spec_wf = np.einsum('ijk,ijl->ijkl', coef_wf, coef_wf.conj())
    spec_wf = bn.move_mean(spec_wf, window=freq_window, min_count=1, axis=0)
    spec_wf = bn.move_mean(spec_wf, window=time_window, min_count=1, axis=1)

    # There are two ways to compute the degree of polarization:
    # To see the difference in theory, please refer to the paper by Taubenschuss and Santonlik (2019).
    # Equation (28) in Taubenschuss and Santonlik 2019: 
    # degree_of_polarization = np.sqrt(3 / 2 * np.abs(np.trace(np.matmul(spec, spec), axis1 = 2, axis2 = 3) / (np.trace(spec, axis1 = 2, axis2 = 3) ** 2)) - 1 / 2)

    # Equation (74) in Taubenschuss and Santonlik 2019:
    # Be careful about np.linalg.eigh, which returns the eigenvalues in ascending order
    # While np.linalg.svd returns the singular values in descending order  
    w, v = np.linalg.eigh(spec)
    degree_of_polarization = (w[:, :, 2] - w[:, :, 1]) / np.sum(w, axis = -1)

    coherence = np.abs(spec_wf[:, :, 0, 1]) / np.sqrt(np.abs(spec_wf[:, :, 0, 0] * spec_wf[:, :, 1, 1]))

    return planarity, ellipticity_along_k, coherence, degree_of_polarization, vh

def fac_wave_analysis(coef: np.ndarray, magf: np.ndarray):
    """
    Project wave coefficients into the field-aligned coordinate (FAC) system.

    Parameters:
    ----------
    coef : np.ndarray
        Wavelet or FFT coefficient tensor of shape (Nf, Nt, 3), where Nf is the number of frequencies, Nt is the number of time points, and 3 represents the 3 components.
    field : np.ndarray
        Reference magnetic field vector of shape (Nt, 3).

    Returns:
    -------
    compressibility : np.ndarray
        Ratio of parallel power to the total power, indicating the compressibility of the wave.
    ellipticity_along_b : np.ndarray
        Ellipticity along the magnetic field direction, defined as the ratio of left-hand to right-hand polarized power.

    Notes:
    -----
    - The FAC system is defined using the magnetic field vector as the parallel direction, and two perpendicular directions are computed using cross products.
    - The left-hand and right-hand polarized components are computed in the perpendicular plane.
    """
    dir_para = (magf.T / np.linalg.norm(magf, axis = 1)).T
    # Find the reference direction that is furthest from the magnetic field direction
    dir_ref = np.eye(3)[np.argmin(np.abs(dir_para), axis = 1)]

    dir_perp_1 = np.cross(dir_para, dir_ref)
    dir_perp_1 = (dir_perp_1.T / np.linalg.norm(dir_perp_1, axis = 1)).T

    dir_perp_2 = np.cross(dir_para, dir_perp_1)
    dir_perp_2 = (dir_perp_2.T / np.linalg.norm(dir_perp_2, axis = 1)).T

    coef_para = np.einsum('ijk,jk->ij', coef, dir_para)
    coef_perp1 = np.einsum('ijk,jk->ij', coef, dir_perp_1)
    coef_perp2 = np.einsum('ijk,jk->ij', coef, dir_perp_2)

    coef_lh = (coef_perp1 - 1j * coef_perp2) / np.sqrt(2)
    coef_rh = (coef_perp1 + 1j * coef_perp2) / np.sqrt(2)

    ellipticity_along_b = (np.abs(coef_rh) - np.abs(coef_lh)) / (np.abs(coef_rh) + np.abs(coef_lh))
    compressibility = np.abs(coef_para) ** 2 / (np.abs(coef_para) ** 2 + np.abs(coef_lh) ** 2 + np.abs(coef_rh) ** 2)

    return compressibility, ellipticity_along_b

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
# Quick look at a time range (mag + H+ spectrogram):
#fig = plot_quick_look('2015-02-01/07:20:00', '2015-02-01/07:40:00', show_loading=False,show_kt17=True, save_path="figures/quicklook.png")
#fig = plot_quick_look(orbit = 3782, only_cs=True, show_loading=True, show_kt17=True, save_path="figures/quicklook.png")
#fig = plot_quick_look('2013-05-31/09:04:10', '2013-05-31/09:06:40', show_loading=False,show_kt17=False, save_path="figures/quicklook.png")
#fig = plot_quick_look(orbit = 1709, only_cs=True, show_loading=False, show_kt17=True, save_path="figures/quicklook.png")

#fig = plot_quick_look('2013-05-28/17:04:11', '2013-05-28/17:09:11', only_cs=True, show_loading=False, show_kt17=False, save_path="figures/quicklook.png",figsize=(20,6))

# Load MAG data and compute FAC components:
#   df = load_bowers_data_pkl(orbit_number=3451)
#   _, Bxm, Bym, Bzm = get_kt17_along_track(df=df)
#   B_perp, B_phi, B_par = transform_to_fac(
#       df['magx'], df['magy'], df['magz'], Bxm, Bym, Bzm,
#       df['ephx'], df['ephy'], df['ephz'])
#
# FIPS for a specific orbit:
#plot_fips_for_orbit(3941)


# ---------------------------------------------------------------------------
# Analyze loading B_phi e.g. first in orbit 2493
# 'human_loading_labels_partitioned.parquet' stores the pre-built event data, delete it to rebuild if new events are added
# and 'unloading_fac_cache.npz' is for the computed delta B_phi
# ---------------------------------------------------------------------------


_EVENTS_CACHE    = os.path.join(_SCRIPT_DIR, 'human_loading_labels_partitioned.parquet')
_UNLOADING_CACHE = os.path.join(_SCRIPT_DIR, 'unloading_fac_cache.npz')

def _get_unloading_data():
    """Load the unloading FAC cache, building it from scratch if needed."""
    if os.path.exists(_UNLOADING_CACHE):
        _cache = np.load(_UNLOADING_CACHE, allow_pickle=True)
        data = _cache['unloading_data'].item()
        print(f'Loaded cached FAC data ({len(data)} events) from {_UNLOADING_CACHE}')
        return data

    if os.path.exists(_EVENTS_CACHE):
        events = pd.read_parquet(_EVENTS_CACHE)
        print(f'Loaded cached event table ({len(events)} events) from {_EVENTS_CACHE}')
    else:
        events = load_human_loading_labels()
        events.to_parquet(_EVENTS_CACHE, index=False)
        print(f'Saved event table to {_EVENTS_CACHE}')

    data = {}
    orbit_counters = {}
    for _, event_data in events.iterrows():
        orbit     = event_data['orbit']
        start     = event_data['start']
        partition = event_data['partition']
        stop      = event_data['stop']

        df_event    = load_bowers_data_pkl(trange=[start, stop])
        t_event     = pd.to_datetime(df_event['time'])
        i_partition = int((t_event - partition).abs().argmin())

        df_loading   = df_event.iloc[:i_partition]
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
        DeltaB_phi_unload  = B_phi_unload  - B_phi_unload[0]
        DeltaB_par_unload  = B_par_unload  - B_par_unload[0]

        letter = chr(ord('a') + orbit_counters.get(orbit, 0))
        orbit_counters[orbit] = orbit_counters.get(orbit, 0) + 1
        key = f'{int(orbit)}{letter}'

        data[key] = {
            'orbit':  orbit,
            'time':   pd.to_datetime(df_unloading['time']).to_numpy(),
            'x':      df_unloading['ephx'].to_numpy(),
            'y':      df_unloading['ephy'].to_numpy(),
            'z':      df_unloading['ephz'].to_numpy(),
            'Bx':     df_unloading['magx'].to_numpy(),
            'By':     df_unloading['magy'].to_numpy(),
            'Bz':     df_unloading['magz'].to_numpy(),
            'B_phi':  B_phi_unload,
            'B_perp': B_perp_unload,
            'B_par':  B_par_unload,
            'DeltaB_phi':  DeltaB_phi_unload,
            'DeltaB_perp': DeltaB_perp_unload,
            'DeltaB_par':  DeltaB_par_unload,
        }

    np.savez(_UNLOADING_CACHE, unloading_data=data)
    print(f'Saved FAC cache ({len(data)} events) to {_UNLOADING_CACHE}')
    return data

# ---------------------------------------------------------------------------
# Plot all unloading data as colored line segments in lat/lon + XZ planes
# ---------------------------------------------------------------------------

_COMPONENT_META = {
    'B_phi':  ('DeltaB_phi',  r'$\Delta B_\phi$',        'Bphi'),
    'B_perp': ('DeltaB_perp', r'$\Delta B_\perp$',       'Bperp'),
    'B_par':  ('DeltaB_par',  r'$\Delta B_\parallel$',   'Bpar'),
}

def plot_unloading_maps(unloading_data=None, component='B_phi', vmax=10,
                        n_lon_bins=16, n_lat_bins=10, label_events=False,
                        lshell_split=None, rsun=0.387, dist_index=5.0):
    """Three-figure summary of unloading FAC data in lat/lon and XZ planes.

    Parameters
    ----------
    unloading_data : dict
        Output of the caching block; keys like '3941a', values contain
        x/y/z ephemeris arrays and DeltaB_phi/DeltaB_perp/DeltaB_par arrays.
    component : {'B_phi', 'B_perp', 'B_par'}
        Which FAC component to colour the trajectories and scatter points by.
    vmax : float
        Symmetric colour-scale limit (nT).
    n_lon_bins : int
        Number of longitude bins for the 2-D histogram.
    n_lat_bins : int
        Number of latitude bins for the 2-D histogram.
    label_events: bool
        Show event numbers for each trajectory segment, for debug.
    """
    if unloading_data is None:
        unloading_data = _get_unloading_data()
    data_key, lbl, fname_tag = _COMPONENT_META[component]
    norm = plt.Normalize(-vmax, vmax)

    def _add_segments(ax, pts, c):
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        cv   = 0.5 * (c[:-1] + c[1:])
        lc   = LineCollection(segs, cmap='RdBu_r', norm=norm, linewidth=2, alpha=0.8)
        lc.set_array(cv)
        ax.add_collection(lc)

    # ---- Figure 1: three-panel trajectory map --------------------------------
    fig_map = plt.figure(figsize=(10, 9))
    gs = GridSpec(2, 2, figure=fig_map, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
    ax_top  = fig_map.add_subplot(gs[0, :])
    ax_ypos = fig_map.add_subplot(gs[1, 0])
    ax_yneg = fig_map.add_subplot(gs[1, 1])

    for key, ev in unloading_data.items():
        x, y, z = ev['x'], ev['y'], ev['z']
        bval    = ev[data_key]
        r   = np.sqrt(x**2 + y**2 + z**2)
        lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
        lon = np.degrees(np.arctan2(y, x)) % 360
        _add_segments(ax_top, np.column_stack([lon, lat]), bval)
        mid = len(lon) // 2
        if label_events:
            ax_top.text(lon[mid], lat[mid], key, fontsize=5, ha='center', va='bottom',
                    color='k', alpha=0.7, clip_on=True)
        for mask, ax in [(y > 0, ax_ypos), (y < 0, ax_yneg)]:
            if mask.sum() < 2:
                continue
            _add_segments(ax, np.column_stack([x[mask], z[mask]]), bval[mask])

    ax_top.set_xlim(100, 260)
    ax_top.set_ylim(-60, 60)
    ax_top.set_xlabel('Longitude (deg, MSM)')
    ax_top.set_ylabel('Latitude (deg, MSM)')
    ax_top.set_title(f'Unloading {lbl} — lat/lon')
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

    if lshell_split is not None:
        lam = np.linspace(-np.pi / 2, np.pi / 2, 500)
        xfl = -lshell_split * np.cos(lam) ** 3
        zfl =  lshell_split * np.cos(lam) ** 2 * np.sin(lam)
        ok_d = np.sqrt(xfl**2 + (zfl + 0.2)**2) >= 0.8
        for ax in (ax_ypos, ax_yneg):
            ax.plot(np.where(ok_d, xfl, np.nan), np.where(ok_d, zfl, np.nan),
                    color='k', ls='-', lw=1.5, alpha=0.6,
                    label=f'L = {lshell_split} R$_M$')
            ax.legend(fontsize=7, loc='upper right')

    # dipole and KT17 field lines at fixed L values
    lam = np.linspace(-np.pi / 2, np.pi / 2, 500)
    l_values = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(l_values)))
    for i, L_val in enumerate(l_values):
        col = colors[i]

        # analytic dipole (nightside, y=0)
        xfl = -L_val * np.cos(lam) ** 3
        zfl =  L_val * np.cos(lam) ** 2 * np.sin(lam)
        ok_d = np.sqrt(xfl**2 + (zfl + 0.2)**2) >= 0.8

        # KT17 trace
        T  = KT17.TraceField(-L_val, 0.0, 0.0,
                             Rsun=rsun, DistIndex=dist_index,
                             EndSurface=2, TraceDir=0)
        tr = T.GetTrace(0)
        xk, zk = np.array(tr['x']), np.array(tr['z'])
        ok_k = np.sqrt(xk**2 + np.array(tr['y'])**2 + (zk + 0.2)**2) >= 0.8

        for ax in (ax_ypos, ax_yneg):
            ax.plot(np.where(ok_d, xfl, np.nan), np.where(ok_d, zfl, np.nan),
                    color=col, ls='--', lw=0.9, alpha=0.7,
                    label=f'Dipole L={L_val}' if ax is ax_ypos else '_')
            ax.plot(np.where(ok_k, xk, np.nan), np.where(ok_k, zk, np.nan),
                    color=col, ls='-', lw=0.9, alpha=0.7,
                    label=f'KT17 L={L_val}' if ax is ax_ypos else '_')

    # legend with proxy lines for dipole/KT17 style
    from matplotlib.lines import Line2D
    proxy = [Line2D([0], [0], color='grey', ls='--', lw=1.2, label='Dipole'),
             Line2D([0], [0], color='grey', ls='-',  lw=1.2, label='KT17')]
    for ax in (ax_ypos, ax_yneg):
        ax.legend(handles=proxy, fontsize=7, loc='upper right')

    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    fig_map.colorbar(sm, ax=[ax_top, ax_ypos, ax_yneg], label=f'{lbl} (nT)',
                     shrink=0.6, pad=0.02)
    fig_map.savefig(os.path.join(_SCRIPT_DIR, 'figures', f'unloading_{fname_tag}_map.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()

    # ---- Figure 2: scatter of per-event mean value ---------------------------
    pts_lon, pts_lat, pts_val, pts_key, pts_L, pts_xyz = [], [], [], [], [], []
    for key, ev in unloading_data.items():
        x, y, z = ev['x'], ev['y'], ev['z']
        r   = np.sqrt(x**2 + y**2 + z**2)
        lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
        lon = np.degrees(np.arctan2(y, x)) % 360
        L   = np.mean(r**3 / (x**2 + y**2))   # dipole L-shell in R_M (MSM)
        pts_lon.append(np.mean(lon))
        pts_lat.append(np.mean(lat))
        pts_val.append(np.mean(ev[data_key]))
        pts_key.append(key)
        pts_L.append(L)
        pts_xyz.append((np.mean(x), np.mean(y), np.mean(z)))

    pts_lon = np.array(pts_lon)
    pts_lat = np.array(pts_lat)
    pts_val = np.array(pts_val)
    pts_L   = np.array(pts_L)
    pts_xyz = np.array(pts_xyz)   # (N, 3)

    def _scatter_panel(ax, mask, subtitle):
        sc = ax.scatter(pts_lon[mask], pts_lat[mask], c=pts_val[mask], cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax, s=40, edgecolors='k', lw=0.4, zorder=3)
        if label_events:
            for lon, lat, key, xyz in zip(pts_lon[mask], pts_lat[mask],
                                          np.array(pts_key)[mask],
                                          pts_xyz[mask]):
                lbl_txt = f'{key}\n({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})'
                ax.text(lon, lat, lbl_txt, fontsize=4, ha='left', va='bottom',
                        alpha=0.7, clip_on=True)
        ax.set_xlim(100, 260)
        ax.set_ylim(-60, 60)
        ax.set_xlabel('Longitude (deg, MSM)')
        ax.set_ylabel('Latitude (deg, MSM)')
        ax.set_title(subtitle)
        ax.grid(True, alpha=0.3)
        return sc

    all_mask = np.ones(len(pts_lon), dtype=bool)
    if lshell_split is None:
        fig_sc = plt.figure(figsize=(12, 9))
        gs_sc  = GridSpec(2, 2, figure=fig_sc, height_ratios=[1.2, 1],
                          hspace=0.45, wspace=0.3)
        ax_sc   = fig_sc.add_subplot(gs_sc[0, :])
        ax_ypos = fig_sc.add_subplot(gs_sc[1, 0])
        ax_yneg = fig_sc.add_subplot(gs_sc[1, 1])
        sc = _scatter_panel(ax_sc, all_mask, f'Mean {lbl} per unloading event (nT)')
        top_axes = [ax_sc]
    else:
        inside  = pts_L <= lshell_split
        outside = pts_L >  lshell_split
        fig_sc = plt.figure(figsize=(12, 13))
        gs_sc  = GridSpec(3, 2, figure=fig_sc, height_ratios=[1, 1, 1.2],
                          hspace=0.45, wspace=0.3)
        ax_in   = fig_sc.add_subplot(gs_sc[0, :])
        ax_out  = fig_sc.add_subplot(gs_sc[1, :])
        ax_ypos = fig_sc.add_subplot(gs_sc[2, 0])
        ax_yneg = fig_sc.add_subplot(gs_sc[2, 1])
        sc = _scatter_panel(ax_in,  inside,
                            f'L ≤ {lshell_split} R$_M$  —  mean {lbl} (nT)  '
                            f'[{inside.sum()} events]')
        _scatter_panel(ax_out, outside,
                       f'L > {lshell_split} R$_M$  —  mean {lbl} (nT)  '
                       f'[{outside.sum()} events]')
        top_axes = [ax_in, ax_out]

    # XZ scatter panels
    pts_x = pts_xyz[:, 0]
    pts_y = pts_xyz[:, 1]
    pts_z = pts_xyz[:, 2]
    for ax, ymask, title in [(ax_ypos, pts_y > 0, 'XZ plane  (Y > 0)'),
                              (ax_yneg, pts_y < 0, 'XZ plane  (Y < 0)')]:
        sc_xz = ax.scatter(pts_x[ymask], pts_z[ymask], c=pts_val[ymask],
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           s=40, edgecolors='k', lw=0.4, zorder=3)
        ax.add_patch(plt.Circle((0, -0.2), 1.0, color='grey', zorder=0, alpha=0.5))
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

    # dipole and KT17 field lines on XZ panels
    lam = np.linspace(-np.pi / 2, np.pi / 2, 500)
    l_values_sc = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    colors_sc = plt.cm.cool(np.linspace(0.1, 0.9, len(l_values_sc)))
    for col, L_val in zip(colors_sc, l_values_sc):
        xfl = -L_val * np.cos(lam) ** 3
        zfl =  L_val * np.cos(lam) ** 2 * np.sin(lam)
        ok_d = np.sqrt(xfl**2 + (zfl + 0.2)**2) >= 0.8
        T  = KT17.TraceField(-L_val, 0.0, 0.0,
                             Rsun=rsun, DistIndex=dist_index,
                             EndSurface=2, TraceDir=0)
        tr = T.GetTrace(0)
        xk, zk = np.array(tr['x']), np.array(tr['z'])
        ok_k = np.sqrt(xk**2 + np.array(tr['y'])**2 + (zk + 0.2)**2) >= 0.8
        for ax in (ax_ypos, ax_yneg):
            ax.plot(np.where(ok_d, xfl, np.nan), np.where(ok_d, zfl, np.nan),
                    color=col, ls='--', lw=0.9, alpha=0.7)
            ax.plot(np.where(ok_k, xk, np.nan), np.where(ok_k, zk, np.nan),
                    color=col, ls='-',  lw=0.9, alpha=0.7)

    if lshell_split is not None:
        xfl_ls = -lshell_split * np.cos(lam) ** 3
        zfl_ls =  lshell_split * np.cos(lam) ** 2 * np.sin(lam)
        ok_ls  = np.sqrt(xfl_ls**2 + (zfl_ls + 0.2)**2) >= 0.8
        for ax in (ax_ypos, ax_yneg):
            ax.plot(np.where(ok_ls, xfl_ls, np.nan), np.where(ok_ls, zfl_ls, np.nan),
                    color='k', ls='-', lw=1.5, alpha=0.6,
                    label=f'L = {lshell_split} R$_M$')

    from matplotlib.lines import Line2D
    proxy_sc = [Line2D([0], [0], color='grey', ls='--', lw=1.2, label='Dipole'),
                Line2D([0], [0], color='grey', ls='-',  lw=1.2, label='KT17')]
    for ax in (ax_ypos, ax_yneg):
        ax.legend(handles=proxy_sc, fontsize=7, loc='upper right')

    sm_sc = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(-vmax, vmax))
    sm_sc.set_array([])
    fig_sc.colorbar(sm_sc, ax=top_axes + [ax_ypos, ax_yneg],
                    label=f'{lbl} (nT)', shrink=0.6, pad=0.02)

    fig_sc.savefig(os.path.join(_SCRIPT_DIR, 'figures', f'unloading_{fname_tag}_scatter.png'),
                   dpi=150, bbox_inches='tight')
    plt.show()

    # ---- Figure 3: 2-D binned mean ------------------------------------------
    lon_edges = np.linspace(100, 260, n_lon_bins + 1)
    lat_edges = np.linspace(-60,  60, n_lat_bins + 1)
    LON, LAT  = np.meshgrid(lon_edges, lat_edges)

    def _bin_panel(ax, mask, subtitle):
        b_sum   = np.zeros((n_lat_bins, n_lon_bins))
        b_count = np.zeros((n_lat_bins, n_lon_bins), dtype=int)
        for lon, lat, val in zip(pts_lon[mask], pts_lat[mask], pts_val[mask]):
            i = np.searchsorted(lon_edges, lon, side='right') - 1
            j = np.searchsorted(lat_edges, lat, side='right') - 1
            if 0 <= i < n_lon_bins and 0 <= j < n_lat_bins:
                b_sum[j, i]   += val
                b_count[j, i] += 1
        b_mean = np.where(b_count > 0, b_sum / b_count, np.nan)
        pm = ax.pcolormesh(LON, LAT, b_mean, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='flat')
        for i in range(n_lon_bins):
            for j in range(n_lat_bins):
                if b_count[j, i] > 0:
                    cx = 0.5 * (lon_edges[i] + lon_edges[i + 1])
                    cy = 0.5 * (lat_edges[j] + lat_edges[j + 1])
                    ax.text(cx, cy, str(b_count[j, i]),
                            ha='center', va='center', fontsize=7, color='k')
        ax.set_xlim(100, 260)
        ax.set_ylim(-60, 60)
        ax.set_xlabel('Longitude (deg, MSM)')
        ax.set_ylabel('Latitude (deg, MSM)')
        ax.set_title(subtitle)
        ax.grid(True, alpha=0.2)
        return pm

    all_mask = np.ones(len(pts_lon), dtype=bool)
    if lshell_split is None:
        fig_bin, ax_bin = plt.subplots(figsize=(10, 5))
        pm = _bin_panel(ax_bin, all_mask,
                        f'Binned mean {lbl} (nT)  —  '
                        f'{n_lon_bins} × {n_lat_bins} bins  (count labelled)')
        fig_bin.colorbar(pm, ax=ax_bin, label=f'{lbl} (nT)')
    else:
        inside  = pts_L <= lshell_split
        outside = pts_L >  lshell_split
        fig_bin, axs_bin = plt.subplots(2, 1, figsize=(10, 9),
                                        sharex=True, sharey=True, squeeze=False)
        ax_in, ax_out = axs_bin[:, 0]
        pm = _bin_panel(ax_in,  inside,
                        f'L ≤ {lshell_split} R$_M$  —  binned mean {lbl} (nT)  '
                        f'[{inside.sum()} events]')
        _bin_panel(ax_out, outside,
                   f'L > {lshell_split} R$_M$  —  binned mean {lbl} (nT)  '
                   f'[{outside.sum()} events]')
        fig_bin.colorbar(pm, ax=axs_bin.ravel().tolist(), label=f'{lbl} (nT)')

    fig_bin.savefig(os.path.join(_SCRIPT_DIR, 'figures', f'unloading_{fname_tag}_binned.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()

    # ---- Figure 4: scatter of mean Bz / Bz_dipole per event -----------------
    # Dipole Bz in MSM (z-aligned, centred at origin): Bz_dip = M*(3z^2-r^2)/r^5
    # Mercury dipole moment M = 190 nT·R_M^3 gives -190 nT at the magnetic equator.
    _M_MERCURY = 200.0   # nT·R_M^3
    _BZ_DIP_MIN = 0.1    # nT — exclude near-equatorial points to avoid ÷0

    pts_bz_ratio = []
    for key, ev in unloading_data.items():
        x, y, z = ev['x'], ev['y'], ev['z']
        r = np.sqrt(x**2 + y**2 + (z-0.2)**2)
        bx_dip = 3*_M_MERCURY * x*z/r**5
        by_dip = 3*_M_MERCURY * y*z/r**5
        bz_dip = _M_MERCURY * (3*(z-0.2)**2 - r**2) / r**5
        valid = np.abs(bz_dip) >= _BZ_DIP_MIN
        if valid.sum() > 0:
            # Bz_obs/Bz_dip
            #ratio = np.mean(ev['Bz'][valid] / bz_dip[valid])

            b_obs = np.sqrt(ev['Bx'][valid]**2 + ev['By'][valid]**2 + ev['Bz'][valid]**2)
            b_dip = np.sqrt(bx_dip[valid]**2 + by_dip[valid]**2 + bz_dip[valid]**2)
            #ratio = float(np.mean(b_obs / b_dip))

            # mag_elev_obs/mag_elev/dip
            mag_elev_obs = np.arctan(np.abs(ev['Bz'][valid])/(np.sqrt(ev['Bx'][valid]**2+ev['By'][valid]**2)))
            mag_elev_dip = np.arctan(np.abs(bz_dip[valid])/(np.sqrt(bx_dip[valid]**2+by_dip[valid]**2)))
            ratio = np.mean(mag_elev_obs/mag_elev_dip)

        else:
            ratio = np.nan
        pts_bz_ratio.append(np.abs(ratio))

    pts_bz_ratio = np.array(pts_bz_ratio)
    finite_mask = np.isfinite(pts_bz_ratio)

    from matplotlib.colors import LogNorm
    norm_ratio = LogNorm(vmin=0.1, vmax=10)

    fig_bz, ax_bz = plt.subplots(figsize=(10, 5))
    sc_bz = ax_bz.scatter(pts_lon[finite_mask], pts_lat[finite_mask],
                          c=pts_bz_ratio[finite_mask], cmap='RdBu_r',
                          norm=norm_ratio, s=40, edgecolors='k', lw=0.4, zorder=3)
    if label_events:
        for lon, lat, key in zip(pts_lon[finite_mask], pts_lat[finite_mask],
                                 np.array(pts_key)[finite_mask]):
            ax_bz.text(lon, lat, key, fontsize=4, ha='left', va='bottom',
                       alpha=0.7, clip_on=True)
    ax_bz.set_xlim(100, 260)
    ax_bz.set_ylim(-60, 60)
    ax_bz.set_xlabel('Longitude (deg, MSM)')
    ax_bz.set_ylabel('Latitude (deg, MSM)')
    #ax_bz.set_title(r'Mean $B_z\,/\,B_z^{\rm dip}$ per unloading event')
    ax_bz.set_title(r'Max $\Theta_{obs}\,/\Theta_{dip}$ per unloading event')
    ax_bz.grid(True, alpha=0.3)
    #fig_bz.colorbar(sc_bz, ax=ax_bz, label=r'$B_z\,/\,B_z^{\rm dip}$')
    fig_bz.colorbar(sc_bz, ax=ax_bz, label=r'$\Theta_{obs}\,/\Theta_{dip}$')
    fig_bz.savefig(os.path.join(_SCRIPT_DIR, 'figures', 'unloading_bz_ratio_scatter.png'),
                   dpi=150, bbox_inches='tight')
    plt.show()

#plot_unloading_maps(component='B_phi', vmax=10, label_events=True, lshell_split=None)

# ---------------------------------------------------------------------------
# Southward-orbit browser  (foundation for the loading/unloading labelling toolkit)
# ---------------------------------------------------------------------------

def _is_southward(seg_df):
    """True if Z_MSM decreases overall across the nightside crossing segment."""
    z = seg_df['ephz'].to_numpy()
    return float(z[-1]) < float(z[0])

def browse_southward_orbits(n0, n1, species=None, show_kt17=True):
    """Plot each orbit in [n0, n1] whose nightside crossing moves southward.

    Loads each orbit once, restricts to the nightside current-sheet segment,
    and skips any orbit where Z_MSM is not decreasing.  The pre-loaded segment
    is passed directly to plot_quick_look to avoid a second data load.

    Parameters
    ----------
    n0, n1    : int  inclusive orbit range
    species   : list[str] or None  FIPS species to overlay, e.g. ['H+']
    show_kt17 : bool  show KT17 model overlay and ΔB residual panel (slow)
    """
    sp      = tuple(species) if species else ()
    n_shown = 0

    for orb in range(n0, n1 + 1):
        try:
            orb_df = load_bowers_data_pkl(orbit_number=orb)
        except Exception as exc:
            print(f'Orbit {orb}: skipped ({exc})')
            continue

        seg = filter_orbit_segment(orb_df)
        if seg.empty:
            continue
        if not _is_southward(seg):
            continue

        seg = seg[seg['ephz'] > 0]
        if seg.empty:
            continue

        n_shown += 1
        t0_str = pd.to_datetime(seg['time']).iloc[0].strftime('%Y-%m-%d %H:%M')
        print(f'Orbit {orb}  —  {t0_str}  (southward)')
        plot_quick_look(df=seg, orbit=orb, df_full=orb_df, species=sp,
                        show_kt17=show_kt17, show_loading=False, ylim_mag=[-200, 200])
        plt.show()

    print(f'\nDone — showed {n_shown} southward orbit(s) in range {n0}–{n1}.')


def label_southward_orbits(n0, n1, json_path=None, species=None, ylim_mag=None, direction='south'):
    """Interactive loading/unloading event labeller for nightside crossings.

    "Southward"/"northward" here describe the direction Z_MSM moves through
    the crossing (dZ/dt), not hemisphere — southward means the spacecraft
    moves from +Z to -Z across the current sheet, northward is the reverse.

    For each qualifying orbit three clicks mark one event:
        1st click  →  START      (green solid line)
        2nd click  →  PARTITION  (orange dashed line)
        3rd click  →  STOP       (red dashed line + green span)
    Repeat for multiple events per orbit.

    Buttons
    -------
    Save & Next  save labelled events and advance
    Undo         remove last click or last completed event
    No events    mark orbit reviewed with no events and advance
    Skip         advance without saving

    Labels are written to JSON after every Save/No-events action, so progress
    survives an abort.  Format matches human_loading_labels.json:
        { "<orbit>": {"reviewed": true,
                      "loading_events": [{"start": ..., "partition": ...,
                                          "stop": ...}, ...]} }

    Parameters
    ----------
    n0, n1      : int  inclusive orbit range
    json_path   : str  output file (default: <script_dir>/human_loading_labels_new.json)
    species     : list[str] or None  FIPS species to show
    show_kt17   : bool  overlay KT17 residuals (slow)
    ylim_mag    : (ymin, ymax) or None
    direction   : 'south' or 'north' — which crossing direction (dZ/dt sign) to label
    """
    import json as _json
    import matplotlib.dates as mdates
    from matplotlib.widgets import Button

    if direction not in ('south', 'north'):
        raise ValueError("direction must be 'south' or 'north'")

    if json_path is None:
        json_path = os.path.join(_SCRIPT_DIR, 'human_loading_labels_new.json')

    labels = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            labels = _json.load(f)

    sp = tuple(species) if species else ()

    for orb in range(n0, n1 + 1):
        try:
            orb_df = load_bowers_data_pkl(orbit_number=orb)
        except Exception as exc:
            print(f'Orbit {orb}: skipped ({exc})')
            continue

        seg = filter_orbit_segment(orb_df)
        if seg.empty:
            continue
        seg = seg[seg['ephz'] > -1] if direction == 'south' else seg[seg['ephz'] < 1]
        if seg.empty:
            continue
        if _is_southward(seg) != (direction == 'south'):
            continue

        fig = plot_quick_look(df=seg, orbit=orb, df_full=orb_df, species=sp,
                              show_kt17=True, show_loading=False,
                              ylim_mag=ylim_mag, _show=False)

        # make room at the bottom for 4-row ephemeris tick labels + buttons
        fig.subplots_adjust(bottom=0.20)

        click_axes = fig._data_axes   # time-series axes, excludes inset/buttons

        # ── state ──────────────────────────────────────────────────────────
        state = {
            'events': [],    # completed: {'start', 'partition', 'stop'}
            'clicks': [],    # pending timestamps for the current in-progress event
            'artists': [],   # all span/vline artists drawn by _redraw
        }

        # pre-fill from existing labels
        entry = labels.get(str(orb), {})
        for ev in entry.get('loading_events', []):
            state['events'].append({
                'start':     pd.Timestamp(ev['start']),
                'partition': pd.Timestamp(ev.get('partition', ev['start'])),
                'stop':      pd.Timestamp(ev['stop']),
            })

        # ── drawing helpers ─────────────────────────────────────────────────
        _CLICK_COLORS = ['limegreen', 'orange', 'red']
        _CLICK_LABELS = ['START', 'PARTITION', 'STOP']

        def _redraw():
            for a in state['artists']:
                try: a.remove()
                except Exception: pass
            state['artists'].clear()
            for ev in state['events']:
                for ax in click_axes:
                    state['artists'].append(
                        ax.axvspan(ev['start'], ev['stop'],
                                   color='green', alpha=0.15, zorder=2))
                    state['artists'] += [
                        ax.axvline(ev['start'],     color='limegreen', lw=1.5, ls='-',  zorder=3),
                        ax.axvline(ev['partition'],  color='orange',    lw=1.5, ls='--', zorder=3),
                        ax.axvline(ev['stop'],       color='red',       lw=1.5, ls='--', zorder=3),
                    ]
            for i, ts in enumerate(state['clicks']):
                for ax in click_axes:
                    state['artists'].append(
                        ax.axvline(ts, color=_CLICK_COLORS[i], lw=1.5, ls=':', zorder=3))
            fig.canvas.draw_idle()

        def _update_title():
            n_done   = len(state['events'])
            n_clicks = len(state['clicks'])
            if n_clicks == 0:
                msg = (f'Click {_CLICK_LABELS[0]} of event {n_done + 1}'
                       f'  —  or use buttons below')
            else:
                msg = f'Click {_CLICK_LABELS[n_clicks]} of event {n_done + 1}'
            fig.suptitle(f'Orbit {orb}  —  {msg}', fontsize=10, color='darkgreen')
            fig.canvas.draw_idle()

        _redraw()
        _update_title()

        # ── buttons ─────────────────────────────────────────────────────────
        ax_save = fig.add_axes([0.28, 0.02, 0.14, 0.04])
        ax_undo = fig.add_axes([0.43, 0.02, 0.08, 0.04])
        ax_none = fig.add_axes([0.52, 0.02, 0.12, 0.04])
        ax_skip = fig.add_axes([0.65, 0.02, 0.07, 0.04])

        btn_save = Button(ax_save, 'Save & Next ✓', color='#d4f0d4', hovercolor='#90e090')
        btn_undo = Button(ax_undo, 'Undo',           color='#fffacd', hovercolor='#f0e060')
        btn_none = Button(ax_none, 'No events',      color='#f0d4d4', hovercolor='#e08080')
        btn_skip = Button(ax_skip, 'Skip',           color='#e8e8e8', hovercolor='#c0c0c0')

        def _write_json():
            with open(json_path, 'w') as f:
                _json.dump(labels, f, indent=2, sort_keys=True)

        def _save(_ev):
            labels[str(orb)] = {
                'reviewed': True,
                'loading_events': [
                    {'start':     ev['start'].isoformat(),
                     'partition': ev['partition'].isoformat(),
                     'stop':      ev['stop'].isoformat()}
                    for ev in state['events']
                ],
            }
            _write_json()
            print(f'  Orbit {orb}: saved {len(state["events"])} event(s)')
            plt.close(fig)

        def _undo(_ev):
            if state['clicks']:
                state['clicks'].pop()
            elif state['events']:
                state['events'].pop()
            _redraw()
            _update_title()

        def _no_events(_ev):
            labels[str(orb)] = {'reviewed': True, 'loading_events': []}
            _write_json()
            print(f'  Orbit {orb}: no events')
            plt.close(fig)

        def _skip(_ev):
            print(f'  Orbit {orb}: skipped (not saved)')
            plt.close(fig)

        def on_click(event):
            if event.inaxes not in click_axes or event.xdata is None:
                return
            ts = pd.Timestamp(mdates.num2date(event.xdata).replace(tzinfo=None))
            state['clicks'].append(ts)
            if len(state['clicks']) == 3:
                clicks = sorted(state['clicks'])
                state['events'].append(
                    {'start': clicks[0], 'partition': clicks[1], 'stop': clicks[2]})
                state['clicks'].clear()
            _redraw()
            _update_title()

        btn_save.on_clicked(_save)
        btn_undo.on_clicked(_undo)
        btn_none.on_clicked(_no_events)
        btn_skip.on_clicked(_skip)

        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show(block=True)
        fig.canvas.mpl_disconnect(cid)


def plot_labelled_events(json_path=None, save_dir=None, species=None,
                         ylim_mag=None, dpi=150, skip_no_events=True, pad_sec=None,
                         pad_from_partition=False, show_kt17=True, show_kt17_residual=None,
                         figsize=(14, 5)):
    """Plot and save every reviewed orbit from the labelling JSON.

    Uses the same layout as label_southward_orbits (FIPS optional).
    One figure is saved per orbit that has at least one labelled event
    (or all reviewed orbits when skip_no_events=False).

    Parameters
    ----------
    json_path       : str or None  — defaults to human_loading_labels_new.json
    save_dir        : str or None  — defaults to <script_dir>/figures/labelled/
    species         : list[str] or None  — FIPS species overlay
    ylim_mag        : (ymin, ymax) or None
    dpi             : int  — output resolution
    skip_no_events  : bool — skip orbits marked reviewed but with no events
    pad_sec         : float or None — if given, crops the plotted window instead of
                      showing the full current-sheet crossing segment. By default the
                      window is [earliest event start - pad_sec, latest event stop + pad_sec];
                      if pad_from_partition=True it is [earliest partition - pad_sec,
                      latest partition + pad_sec] instead.
    pad_from_partition : bool — see pad_sec above
    show_kt17       : bool — overlay the dashed KT17 model field in the top mag panel
    show_kt17_residual : bool or None — add the obs-minus-KT17 residual panel; defaults
                      to show_kt17 (old behaviour) when left as None
    figsize         : (width, height) passed through to plot_quick_look
    """
    import json as _json

    if json_path is None:
        json_path = os.path.join(_SCRIPT_DIR, 'human_loading_labels_new.json')
    if save_dir is None:
        save_dir = os.path.join(_SCRIPT_DIR, 'figures', 'labelled')
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path) as f:
        labels = _json.load(f)

    sp = tuple(species) if species else ()

    n_saved = 0
    for orb_str, entry in sorted(labels.items(), key=lambda kv: int(kv[0])):
        if not entry.get('reviewed', False):
            continue
        events = entry.get('loading_events', [])
        if skip_no_events and not events:
            continue

        orb = int(orb_str)
        try:
            orb_df = load_bowers_data_pkl(orbit_number=orb)
        except Exception as exc:
            print(f'Orbit {orb}: skipped ({exc})')
            continue

        seg = filter_orbit_segment(orb_df)
        if seg.empty:
            continue
        #seg = seg[seg['ephz'] > -0.5]
        if seg.empty:
            continue

        if pad_sec is not None and events:
            if pad_from_partition:
                t_mid_lo = min(pd.Timestamp(ev.get('partition', ev['start'])) for ev in events)
                t_mid_hi = max(pd.Timestamp(ev.get('partition', ev['stop']))  for ev in events)
            else:
                t_mid_lo = min(pd.Timestamp(ev['start']) for ev in events)
                t_mid_hi = max(pd.Timestamp(ev['stop'])  for ev in events)
            t_lo = t_mid_lo - pd.Timedelta(seconds=pad_sec)
            t_hi = t_mid_hi + pd.Timedelta(seconds=pad_sec)
            seg = seg[(pd.to_datetime(seg['time']) >= t_lo) & (pd.to_datetime(seg['time']) <= t_hi)]
            if seg.empty:
                continue

        fig = plot_quick_look(df=seg, orbit=orb, df_full=orb_df, species=sp, figsize=figsize,
                              show_kt17=show_kt17, show_kt17_residual=show_kt17_residual,
                              show_loading=False, ylim_mag=ylim_mag, _show=False)
        fig.subplots_adjust(bottom=0.20)

        click_axes = fig._data_axes
        _COLORS = {'start': 'limegreen', 'partition': 'orange', 'stop': 'red'}
        for ev in events:
            t_start = pd.Timestamp(ev['start'])
            t_part  = pd.Timestamp(ev.get('partition', ev['start']))
            t_stop  = pd.Timestamp(ev['stop'])
            for ax in click_axes:
                ax.axvspan(t_start, t_stop, color='green', alpha=0.15, zorder=2)
                ax.axvline(t_start, color=_COLORS['start'],     lw=1.5, ls='-',  zorder=3)
                ax.axvline(t_part,  color=_COLORS['partition'], lw=1.5, ls='--', zorder=3)
                ax.axvline(t_stop,  color=_COLORS['stop'],      lw=1.5, ls='--', zorder=3)

        n_ev = len(events)
        date_str = pd.Timestamp(seg['time'].iloc[0]).strftime('%Y-%m-%d')
        title = f'Orbit {orb}  —  {date_str}  —  {n_ev} labelled event(s)'
        if events:
            cats = ', '.join(ev.get('category', '?') for ev in events)
            title += f'  —  type: {cats}'
        fig.suptitle(title, fontsize=10)

        out_path = os.path.join(save_dir, f'orbit_{orb:05d}.png')
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out_path}')
        n_saved += 1

    print(f'\nDone — {n_saved} figure(s) saved to {save_dir}')


def sort_loading_type(json_path=None, species=None, ylim_mag=None, show_kt17=True,
                      show_kt17_residual=None, partition_pad=None, only_undesignated=False,
                      only_show_type=None, figsize=(14, 5)):
    """
    Interactive keyboard categorizer for labelled loading events.

    Steps through every labelled substorm in json_path (same visual style as
    plot_labelled_events), one event at a time, and waits for a keypress:
        a  →  category 'a'
        l  →  category 'l'
        n  →  category 'n'
    The category is written to the event's 'category' field and saved to
    json_path immediately, then the next event is shown. Closing the figure
    without pressing a/l/n leaves that event's category unchanged.

    Parameters
    ----------
    json_path          : str or None — labelling JSON to read/write
                         (default: human_loading_labels_new.json)
    species            : list[str] or None — FIPS species overlay
    ylim_mag           : (ymin, ymax) or None
    show_kt17          : bool — overlay the dashed KT17 model field in the top mag panel
    show_kt17_residual : bool or None — add the obs-minus-KT17 residual panel; defaults
                         to show_kt17 (old behaviour) when left as None
    partition_pad      : float or None — if given, window shown is [partition - partition_pad,
                         partition + partition_pad] (seconds); if None (default), shows the
                         entire labelled loading/unloading span [start, stop]
    only_undesignated  : bool — if True, only show events with no category yet;
                         if False, show every event (title shows its existing
                         category, if any)
    only_show_type     : str or None — if given ('a', 'l', or 'n'), only show events
                         already categorized with that type; takes precedence over
                         only_undesignated
    figsize            : (width, height) passed through to plot_quick_look
    """
    import json as _json

    if json_path is None:
        json_path = os.path.join(_SCRIPT_DIR, 'human_loading_labels_new.json')

    with open(json_path) as f:
        labels = _json.load(f)

    def _write_json():
        with open(json_path, 'w') as f:
            _json.dump(labels, f, indent=2, sort_keys=True)

    sp = tuple(species) if species else ()
    _COLORS = {'start': 'limegreen', 'partition': 'orange', 'stop': 'red'}
    _CATEGORY_KEYS = ('a', 'l', 'n')

    queue = []
    for orb_str, entry in sorted(labels.items(), key=lambda kv: int(kv[0])):
        if not isinstance(entry, dict):
            continue
        for idx, ev in enumerate(entry.get('loading_events', [])):
            if only_show_type is not None:
                if ev.get('category') != only_show_type:
                    continue
            elif only_undesignated and ev.get('category'):
                continue
            queue.append((orb_str, idx))

    print(f'{len(queue)} event(s) to review.  Press a / l / n to categorize each.')

    orb_cache = {}
    for orb_str, idx in queue:
        orb = int(orb_str)
        ev  = labels[orb_str]['loading_events'][idx]

        if orb not in orb_cache:
            try:
                orb_cache[orb] = load_bowers_data_pkl(orbit_number=orb)
            except Exception as exc:
                print(f'Orbit {orb}: skipped ({exc})')
                continue
        orb_df = orb_cache[orb]

        seg = filter_orbit_segment(orb_df)
        if seg.empty:
            continue

        t_start = pd.Timestamp(ev['start'])
        t_part  = pd.Timestamp(ev.get('partition', ev['start']))
        t_stop  = pd.Timestamp(ev['stop'])

        if partition_pad is not None:
            t_lo = t_part - pd.Timedelta(seconds=partition_pad)
            t_hi = t_part + pd.Timedelta(seconds=partition_pad)
        else:
            t_lo = t_start
            t_hi = t_stop

        plot_seg = seg
        cropped = seg[(pd.to_datetime(seg['time']) >= t_lo) & (pd.to_datetime(seg['time']) <= t_hi)]
        if not cropped.empty:
            plot_seg = cropped

        fig = plot_quick_look(df=plot_seg, orbit=orb, df_full=orb_df, species=sp, figsize=figsize,
                              show_kt17=show_kt17, show_kt17_residual=show_kt17_residual,
                              show_loading=False, ylim_mag=ylim_mag, _show=False)
        fig.subplots_adjust(bottom=0.10)

        for ax in fig._data_axes:
            ax.axvspan(t_start, t_stop, color='green', alpha=0.15, zorder=2)
            ax.axvline(t_start, color=_COLORS['start'],     lw=1.5, ls='-',  zorder=3)
            ax.axvline(t_part,  color=_COLORS['partition'], lw=1.5, ls='--', zorder=3)
            ax.axvline(t_stop,  color=_COLORS['stop'],      lw=1.5, ls='--', zorder=3)

        n_events = len(labels[orb_str]['loading_events'])
        cur_cat  = ev.get('category')
        cat_str  = f'current: {cur_cat}' if cur_cat else 'unlabelled'
        date_str = t_start.strftime('%Y-%m-%d')
        fig.suptitle(
            f'Orbit {orb}  —  {date_str}  —  event {idx + 1}/{n_events}  —  {cat_str}\n'
            f'Press a / l / n to categorize', fontsize=10)

        def on_key(event, orb_str=orb_str, ev=ev, fig=fig):
            if event.key in _CATEGORY_KEYS:
                ev['category'] = event.key
                _write_json()
                print(f'  Orbit {orb_str}: categorized as "{event.key}"')
                plt.close(fig)

        cid = fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=True)
        fig.canvas.mpl_disconnect(cid)

    print('Done reviewing all queued events.')


def measure_bz_recovery(orbits=None, json_path=None, category='a', save_dir=None,
                        species=None, ylim_mag=None, show_kt17=True, show_kt17_residual=False,
                        pad_sec=None, pad_from_partition=False, search_pad_sec=120,
                        peak_window_sec=10, smooth_sec=1, figsize=(14, 5), dpi=150):
    """
    Measure how long Bz stays negative around an event's partition (unloading
    onset) time before recovering to positive, for labelled events in the
    given orbits.

    For each qualifying event:
      - finds the most negative (boxcar-smoothed) Bz sample within
        peak_window_sec of the partition time — the negative-Bz peak
      - finds the first sample after that peak where Bz has recovered to >= 0
      - delay_sec = time from the peak to the recovery, written to the
        event's 'bz_recovery_sec' field in json_path ('bz_recovery_time' and
        'bz_negative_onset' timestamps are stored alongside it)
      - saves a figure (same layout/lines as plot_labelled_events) with
        purple vertical lines marking the Bz peak (dotted) and Bz>0
        recovery (dash-dot), so the detection can be checked visually

    Parameters
    ----------
    orbits          : list[int] or None — orbit numbers to process; if None (default),
                      every orbit in json_path with at least one event matching
                      `category` is used
    json_path       : str or None — labelling JSON to read/write
                      (default: human_loading_labels_new.json)
    category        : str or None — only process events with this category
                      (default 'a'); None processes every event regardless of category
    save_dir        : str or None — defaults to <script_dir>/figures/bz_recovery/
    species         : list[str] or None — FIPS species overlay
    ylim_mag        : (ymin, ymax) or None
    show_kt17       : bool — overlay the dashed KT17 model field in the top mag panel
    show_kt17_residual : bool — add the obs-minus-KT17 residual panel (default False,
                      unlike plot_labelled_events/sort_loading_type)
    pad_sec         : float or None — crop the plotted window, as in plot_labelled_events
    pad_from_partition : bool — if True (with pad_sec), window is centred on the
                      partition time instead of [start, stop]
    search_pad_sec  : float — how far past the event stop to keep searching for
                      the recovery crossing (default 120s)
    peak_window_sec : float — the negative-Bz peak is taken as the most negative
                      sample within this many seconds before/after the partition
                      time (default 10s)
    smooth_sec      : boxcar smoothing window (s) used for the crossing detection —
                      should match what you're inspecting visually (default 1)
    figsize         : (width, height)
    dpi             : output resolution

    Returns a list of dicts: {'orbit', 'event_index', 'delay_sec'} (delay_sec
    is None for events where no clean negative-to-positive crossing was found).
    """
    import json as _json

    if json_path is None:
        json_path = os.path.join(_SCRIPT_DIR, 'human_loading_labels_new.json')
    if save_dir is None:
        save_dir = os.path.join(_SCRIPT_DIR, 'figures', 'bz_recovery')
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path) as f:
        labels = _json.load(f)

    def _write_json():
        with open(json_path, 'w') as f:
            _json.dump(labels, f, indent=2, sort_keys=True)

    if orbits is None:
        orbits = sorted(
            int(orb_str) for orb_str, entry in labels.items()
            if any(category is None or ev.get('category') == category
                   for ev in entry.get('loading_events', []))
        )

    sp = tuple(species) if species else ()
    _COLORS = {'start': 'limegreen', 'partition': 'orange', 'stop': 'red'}

    results = []
    for orb in orbits:
        orb_str = str(orb)
        entry = labels.get(orb_str, {})
        events = entry.get('loading_events', [])

        try:
            orb_df = load_bowers_data_pkl(orbit_number=orb)
        except Exception as exc:
            print(f'Orbit {orb}: skipped ({exc})')
            continue

        seg = filter_orbit_segment(orb_df)
        if seg.empty:
            continue

        for idx, ev in enumerate(events):
            if category is not None and ev.get('category') != category:
                continue

            t_start = pd.Timestamp(ev['start'])
            t_part  = pd.Timestamp(ev.get('partition', ev['start']))
            t_stop  = pd.Timestamp(ev['stop'])
            search_hi = t_stop + pd.Timedelta(seconds=search_pad_sec)

            if pad_from_partition and pad_sec is not None:
                t_lo, t_hi = t_part - pd.Timedelta(seconds=pad_sec), t_part + pd.Timedelta(seconds=pad_sec)
            elif pad_sec is not None:
                t_lo, t_hi = t_start - pd.Timedelta(seconds=pad_sec), t_stop + pd.Timedelta(seconds=pad_sec)
            else:
                t_lo, t_hi = t_start, t_stop
            t_hi = max(t_hi, search_hi)   # always show enough to see the recovery crossing

            t_seg = pd.to_datetime(seg['time'])
            plot_seg = seg[(t_seg >= t_lo) & (t_seg <= t_hi)]
            if plot_seg.empty:
                print(f'Orbit {orb} event {idx}: no data in window, skipping')
                continue

            t_arr = pd.to_datetime(plot_seg['time']).to_numpy()
            dt_s  = float(np.median(np.diff(t_arr).astype('timedelta64[ns]').astype(float) / 1e9)) if len(t_arr) > 1 else 1.0
            win   = int(round(smooth_sec / dt_s)) if smooth_sec else 1
            magz  = plot_seg['magz'].to_numpy().astype(float)
            if win > 1:
                kernel = np.ones(win) / win
                pad = win // 2
                Bz = np.convolve(np.pad(magz, pad, mode='reflect'), kernel, mode='valid')[:len(magz)]
            else:
                Bz = magz

            peak_lo = np.datetime64(t_part - pd.Timedelta(seconds=peak_window_sec))
            peak_hi = np.datetime64(t_part + pd.Timedelta(seconds=peak_window_sec))
            peak_mask = (t_arr >= peak_lo) & (t_arr <= peak_hi)
            peak_idxs = np.where(peak_mask)[0]

            delay_sec = None
            t_neg = t_pos = None
            if peak_idxs.size == 0:
                print(f'Orbit {orb} event {idx}: no data within {peak_window_sec}s of partition, skipping')
            else:
                idx_neg = peak_idxs[np.argmin(Bz[peak_idxs])]
                if Bz[idx_neg] >= 0:
                    print(f'Orbit {orb} event {idx}: Bz never went negative near partition, skipping')
                else:
                    rec_mask = (t_arr >= t_arr[idx_neg]) & (t_arr <= np.datetime64(search_hi)) & (Bz >= 0)
                    rec_candidates = np.where(rec_mask)[0]
                    rec_candidates = rec_candidates[rec_candidates > idx_neg]
                    if rec_candidates.size == 0:
                        print(f'Orbit {orb} event {idx}: Bz did not recover to positive within the search window')
                    else:
                        idx_pos = rec_candidates[0]
                        t_neg = pd.Timestamp(t_arr[idx_neg])
                        t_pos = pd.Timestamp(t_arr[idx_pos])
                        delay_sec = (t_pos - t_neg).total_seconds()

                        ev['bz_recovery_sec']    = delay_sec
                        ev['bz_recovery_time']   = t_pos.isoformat()
                        ev['bz_negative_onset']  = t_neg.isoformat()
                        _write_json()
                        print(f'Orbit {orb} event {idx}: Bz negative -> positive in {delay_sec:.1f} s')

            results.append({'orbit': orb, 'event_index': idx, 'delay_sec': delay_sec})

            fig = plot_quick_look(df=plot_seg, orbit=orb, df_full=orb_df, species=sp, figsize=figsize,
                                  show_kt17=show_kt17, show_kt17_residual=show_kt17_residual,
                                  show_loading=False, ylim_mag=ylim_mag, _show=False)
            for ax in fig._data_axes:
                ax.axvspan(t_start, t_stop, color='green', alpha=0.15, zorder=2)
                ax.axvline(t_start, color=_COLORS['start'],     lw=1.5, ls='-',  zorder=3)
                ax.axvline(t_part,  color=_COLORS['partition'], lw=1.5, ls='--', zorder=3)
                ax.axvline(t_stop,  color=_COLORS['stop'],      lw=1.5, ls='--', zorder=3)
                if t_neg is not None:
                    ax.axvline(t_neg, color='purple', lw=1.5, ls=':',  zorder=4, label='Bz peak')
                if t_pos is not None:
                    ax.axvline(t_pos, color='purple', lw=1.5, ls='-.', zorder=4, label='Bz > 0')

            date_str = t_start.strftime('%Y-%m-%d')
            cat_str  = ev.get('category', '?')
            delay_str = f'{delay_sec:.1f} s' if delay_sec is not None else 'not found'
            fig.suptitle(f'Orbit {orb}  —  {date_str}  —  event {idx}  —  type: {cat_str}  —  '
                        f'Bz recovery: {delay_str}', fontsize=10)

            out_path = os.path.join(save_dir, f'orbit_{orb:05d}_ev{idx}.png')
            fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

    print(f'\nDone — processed {len(results)} event(s).')
    return results


_SIM_EPOCH = pd.Timestamp('2000-01-01T00:00:00')

def load_simulation_json(path, t_start_s=None, t_stop_s=None, t_partition_s=None):
    """Load a simulation JSON file and return (DataFrame, label_entry) for plot_fac_events.

    The JSON must contain 'times' (seconds), 'Bx', 'By', 'Bz'.
    Position (x, y, z) is parsed from the filename, e.g. x-1.1_y0_z0.5_...

    Parameters
    ----------
    path          : str   path to simulation JSON file
    t_start_s     : float or None   start time in seconds (default: first sample)
    t_stop_s      : float or None   stop  time in seconds (default: last sample)
    t_partition_s : float or None   partition time in seconds (default: midpoint)

    Returns
    -------
    df    : pd.DataFrame  with columns time, magx, magy, magz, ephx, ephy, ephz
    label : dict          label_entry compatible with plot_fac_events sim_data
    """
    import json as _json, re

    with open(path) as f:
        raw = _json.load(f)

    times = np.array(raw['times'], dtype=float)
    Bx    = np.array(raw['Bx'],    dtype=float)
    By    = np.array(raw['By'],    dtype=float)
    Bz    = np.array(raw['Bz'],    dtype=float)

    # parse x, y, z from filename  e.g. x-1.1_y0_z0.5_t90_to_150
    fname = os.path.basename(path)
    def _parse_coord(axis):
        m = re.search(rf'{axis}(-?[\d.]+)', fname)
        return float(m.group(1)) if m else 0.0
    x_pos = _parse_coord('x')
    y_pos = _parse_coord('y')
    z_pos = _parse_coord('z')

    # apply time window
    t0  = t_start_s     if t_start_s     is not None else times[0]
    t1  = t_stop_s      if t_stop_s      is not None else times[-1]
    t_p = t_partition_s if t_partition_s is not None else 0.5 * (t0 + t1)

    mask   = (times >= t0) & (times <= t1)
    times  = times[mask]
    Bx, By, Bz = Bx[mask], By[mask], Bz[mask]

    timestamps = _SIM_EPOCH + pd.to_timedelta(times, unit='s')

    df = pd.DataFrame({
        'time': timestamps,
        'magx': Bx, 'magy': By, 'magz': Bz,
        'ephx': x_pos, 'ephy': y_pos, 'ephz': z_pos,
    })

    def _to_iso(t_s):
        return (_SIM_EPOCH + pd.to_timedelta(t_s, unit='s')).isoformat()

    label = {
        'start':     _to_iso(t0),
        'partition': _to_iso(t_p),
        'stop':      _to_iso(t1),
        'reviewed':  True,
    }
    return df, label


def plot_fac_events(json_path=None, save_dir=None, dpi=150, skip_no_events=True,
                    b0_ref='start', bpar_smooth_sec=1, smooth_sec=0,
                    n_lon_bins=18, n_lat_bins=12, label_events=False,
                    show_wavelet=False, sim_data=None):
    """Plot each labelled event in field-aligned coordinates (FAC).

    The FAC frame is fixed at a single reference field direction B0.  The three
    components plotted are:

        B_par  — parallel to B0
        B_phi  — azimuthal  (B0 × R, normalised)
        B_norm — meridional / normal (completes right-hand set)

    Multiple events per orbit are saved with letter suffixes:
        orbit_03772a.png, orbit_03772b.png, …

    Parameters
    ----------
    json_path  : str or None  — defaults to human_loading_labels_new.json
    save_dir   : str or None  — defaults to <script_dir>/figures/labelled_fac/
    dpi        : int
    skip_no_events : bool  — skip reviewed orbits with no events
    b0_ref          : 'start' or 'partition'
                      'start'     — B0 taken at the first sample of the event
                      'partition' — B0 taken at the sample nearest the partition time
    bpar_smooth_sec : float  smoothing window (seconds) applied to B_par before
                      fitting it to B_phi and B_norm.  0 = no fit overlay.
    smooth_sec      : float  if > 0, apply an independent boxcar of this width to
                      each FAC component and overplot; second figure shows residuals.
    """
    import json as _json

    if json_path is None:
        json_path = os.path.join(_SCRIPT_DIR, 'human_loading_labels_new.json')
    if save_dir is None:
        save_dir = os.path.join(_SCRIPT_DIR, 'figures', 'labelled_fac')
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path) as f:
        labels = _json.load(f)

    # merge simulation entries into the label dict
    _sim_dfs = {}   # key -> DataFrame for sim entries
    if sim_data:
        for sim_key, (sim_df, sim_label) in sim_data.items():
            labels[sim_key] = {
                'reviewed': True,
                'loading_events': [sim_label],
            }
            _sim_dfs[sim_key] = sim_df

    _letters = 'abcdefghijklmnopqrstuvwxyz'
    n_saved = 0
    event_agg = []   # per-event: mean lon/lat + mean dB_phi over unloading window

    def _sort_key(kv):
        try:
            return (0, int(kv[0]))
        except ValueError:
            return (1, kv[0])

    for orb_str, entry in sorted(labels.items(), key=_sort_key):
        if not entry.get('reviewed', False):
            continue
        events = entry.get('loading_events', [])
        if skip_no_events and not events:
            continue

        if orb_str in _sim_dfs:
            orb_df  = _sim_dfs[orb_str]
            is_sim  = True
        else:
            orb = int(orb_str)
            try:
                orb_df = load_bowers_data_pkl(orbit_number=orb)
            except Exception as exc:
                print(f'Orbit {orb}: skipped ({exc})')
                continue
            is_sim = False

        t_orb = pd.to_datetime(orb_df['time'])

        for ev_idx, ev in enumerate(events):
            t_start = pd.Timestamp(ev['start'])
            t_part  = pd.Timestamp(ev.get('partition', ev['start']))
            t_stop  = pd.Timestamp(ev['stop'])

            mask = (t_orb >= t_start) & (t_orb <= t_stop)
            seg  = orb_df[mask]
            if len(seg) < 3:
                print(f'  Orbit {orb} event {ev_idx}: too few points, skipped')
                continue

            t_seg = pd.to_datetime(seg['time'])
            Bx = seg['magx'].to_numpy(dtype=float)
            By = seg['magy'].to_numpy(dtype=float)
            Bz = seg['magz'].to_numpy(dtype=float)
            Bmag = np.sqrt(Bx**2+By**2+Bz**2)
            rx = seg['ephx'].to_numpy(dtype=float)
            ry = seg['ephy'].to_numpy(dtype=float)
            rz = seg['ephz'].to_numpy(dtype=float)

            # FAC basis fixed at the chosen reference time
            if b0_ref == 'partition':
                t_ns  = t_seg.values.astype('datetime64[ns]')
                t_p   = np.datetime64(t_part, 'ns')
                i0    = int(np.clip(np.searchsorted(t_ns, t_p), 0, len(t_ns) - 1))
            else:
                i0 = 0  # start of event
            b0  = np.array([Bx[i0], By[i0], Bz[i0]])
            b0n = b0 / np.linalg.norm(b0)
            Bx_ref = np.full(len(seg), b0n[0])
            By_ref = np.full(len(seg), b0n[1])
            Bz_ref = np.full(len(seg), b0n[2])

            B_norm, B_phi, B_par = transform_to_fac(
                Bx, By, Bz,
                Bx_ref, By_ref, Bz_ref,
                rx, ry, rz,
            )

            if not is_sim:
                _, Bxkt, Bykt, Bzkt = get_kt17_along_track(df=seg)
                Bmagkt = np.sqrt(Bxkt**2 + Bykt**2 + Bzkt**2)

            # per-component boxcar smooth (smooth_sec mode)
            if smooth_sec > 0 and len(B_par) > 2:
                dt_s = t_seg.diff().dt.total_seconds().median()
                win  = max(1, int(round(smooth_sec / dt_s)))
                def _sm(arr):
                    return (pd.Series(arr)
                            .rolling(win, center=True, min_periods=1)
                            .mean().to_numpy())
                sm_par  = _sm(B_par)
                sm_phi  = _sm(B_phi)
                sm_norm = _sm(B_norm)
            else:
                sm_par = sm_phi = sm_norm = None

            # aggregate mean dB_phi over the unloading window for summary figures
            if sm_phi is not None:
                unload_mask = (t_seg >= t_part).to_numpy()
                if unload_mask.sum() > 0:
                    r_u   = np.sqrt(rx[unload_mask]**2 + ry[unload_mask]**2 + rz[unload_mask]**2)
                    lon_u = np.degrees(np.arctan2(ry[unload_mask], rx[unload_mask])) % 360
                    lat_u = np.degrees(np.arcsin(np.clip(rz[unload_mask] / r_u, -1, 1)))
                    dBphi_u = (B_phi - sm_phi)[unload_mask]
                    suffix = _letters[ev_idx] if len(events) > 1 else ''
                    event_agg.append({
                        'lon': float(np.nanmean(lon_u)),
                        'lat': float(np.nanmean(lat_u)),
                        'val': float(np.nanmean(dBphi_u)),
                        'key': f'{orb}{suffix}',
                    })

            # unit vectors at the reference point for title annotation
            R0      = np.array([rx[i0], ry[i0], rz[i0]])
            phi_v   = np.cross(b0n, R0)
            phi_h   = phi_v / np.linalg.norm(phi_v)
            perp_h  = np.cross(phi_h, b0n)

            def _fmtvec(v):
                return '[{:.2f}, {:.2f}, {:.2f}]'.format(*v)

            fig, (ax_par, ax_phi, ax_norm) = plt.subplots(
                3, 1, figsize=(10, 7), sharex=True)

            # smooth B_par with a boxcar of bpar_smooth_sec seconds
            if bpar_smooth_sec > 0 and len(B_par) > 2:
                dt_s = t_seg.diff().dt.total_seconds().median()
                win  = max(1, int(round(bpar_smooth_sec / dt_s)))
                B_par_sm = (pd.Series(B_par)
                            .rolling(win, center=True, min_periods=1)
                            .mean()
                            .to_numpy())
            else:
                B_par_sm = None

            # index of partition time — anchor point for the fit
            t_ns_arr = t_seg.values.astype('datetime64[ns]')
            i_part   = int(np.clip(
                np.searchsorted(t_ns_arr, np.datetime64(t_part, 'ns')),
                0, len(t_ns_arr) - 1))

            fit_lines = {}   # keyed by 'phi' / 'norm', stores (fit_line, slope, intercept)

            _sm_by_key = {'par': sm_par, 'phi': sm_phi, 'norm': sm_norm}

            for ax, data, color, lbl, key in [
                (ax_par,  B_par,  'blue',   '$B_{\\parallel}$', 'par'),
                (ax_phi,  B_phi,  'orange', '$B_{\\phi}$',      'phi'),
                (ax_norm, B_norm, 'purple', '$B_{\\perp}$',     'norm'),
            ]:
                ax.plot(t_seg, data, color=color, lw=0.9)
                if _sm_by_key[key] is not None:
                    ax.plot(t_seg, _sm_by_key[key], color='k', lw=1.4, ls='--', alpha=0.8,
                            label=f'{smooth_sec}s smooth')
                    ax.legend(fontsize=7, loc='upper right')
                if np.nanmin(data) <= 0 <= np.nanmax(data):
                    ax.axhline(0, color='k', lw=0.4, alpha=0.4)
                ax.axvline(t_part, color='grey', lw=1.2, ls='--', alpha=0.7)
                ax.set_ylabel(f'{lbl} (nT)', fontsize=9)
                ax.grid(True, alpha=0.3)

            if is_sim:
                # show elapsed seconds instead of clock time
                import matplotlib.ticker as _mticker, datetime as _dt
                _epoch_aware = _SIM_EPOCH.to_pydatetime().replace(tzinfo=_dt.timezone.utc)
                ax_norm.xaxis.set_major_formatter(
                    _mticker.FuncFormatter(
                        lambda x, _, _ep=_epoch_aware: f'{(mdates.num2date(x) - _ep).total_seconds():.0f}s'))
                ax_norm.set_xlabel('Time (s)', fontsize=9)
            else:
                ax_norm.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate(rotation=30, ha='right')
            suffix = _letters[ev_idx] if len(events) > 1 else ''
            b0_label  = 'partition' if b0_ref == 'partition' else 'start'
            orb_label = orb_str if is_sim else f'Orbit {orb_str}'
            if is_sim:
                time_label = f't={t_start.second + (t_start - _SIM_EPOCH).total_seconds():.0f}–{t_stop.second + (t_stop - _SIM_EPOCH).total_seconds():.0f}s'
            else:
                time_label = f'{t_start.strftime("%Y-%m-%d %H:%M")} – {t_stop.strftime("%H:%M")} UTC'
            fig.suptitle(
                f'{orb_label}{suffix}  —  FAC (ref: {b0_label})  ({time_label})\n'
                f'$\\hat{{b}}_{{\\parallel}}$={_fmtvec(b0n)}  '
                f'$\\hat{{b}}_{{\\phi}}$={_fmtvec(phi_h)}  '
                f'$\\hat{{b}}_{{\\perp}}$={_fmtvec(perp_h)}',
                fontsize=8)
            fig.tight_layout()

            orb_tag = orb_str if is_sim else f'{orb:05d}'
            fname = f'orbit_{orb_tag}{suffix}.png'
            out_path = os.path.join(save_dir, fname)
            fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {out_path}')
            n_saved += 1

            # ── difference figure ─────────────────────────────────────────
            if sm_par is not None:
                diff_data = [
                    (B_par  - sm_par,  'blue',
                     f'$B_{{\\parallel}}$ $-$ {smooth_sec}s avg'),
                    (B_phi  - sm_phi,  'orange',
                     f'$B_{{\\phi}}$ $-$ {smooth_sec}s avg'),
                    (B_norm - sm_norm, 'purple',
                     f'$B_{{\\perp}}$ $-$ {smooth_sec}s avg'),
                ]
                fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
                for ax2, (resid, color, lbl2) in zip(axes2, diff_data):
                    ax2.plot(t_seg, resid, color=color, lw=0.9)
                    ax2.axhline(0, color='k', lw=0.4, alpha=0.4)
                    ax2.axvline(t_part, color='grey', lw=1.2, ls='--', alpha=0.7)
                    ax2.set_ylim(-6, 6)
                    ax2.set_ylabel(f'{lbl2} (nT)', fontsize=9)
                    ax2.grid(True, alpha=0.3)
                if is_sim:
                    import matplotlib.ticker as _mticker, datetime as _dt
                    _epoch_aware = _SIM_EPOCH.to_pydatetime().replace(tzinfo=_dt.timezone.utc)
                    axes2[-1].xaxis.set_major_formatter(
                        _mticker.FuncFormatter(
                            lambda x, _, _ep=_epoch_aware: f'{(mdates.num2date(x) - _ep).total_seconds():.0f}s'))
                else:
                    axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                fig2.autofmt_xdate(rotation=30, ha='right')
                fig2.suptitle(
                    f'{orb_label}{suffix}  —  FAC residuals ({smooth_sec}s avg removed)  '
                    f'({time_label})',
                    fontsize=9)
                fig2.tight_layout()
                fname2    = f'orbit_{orb_tag}{suffix}_diff.png'
                out_path2 = os.path.join(save_dir, fname2)
                fig2.savefig(out_path2, dpi=dpi, bbox_inches='tight')
                plt.close(fig2)
                print(f'  Saved: {out_path2}')
            elif B_par_sm is not None:
                diff_data = [
                    (B_par - B_par_sm, 'blue',   '$B_{\\parallel} - \\langle B_{\\parallel}\\rangle$'),
                    (B_phi  - fit_lines['phi'][0]  if 'phi'  in fit_lines else B_phi  - B_par_sm,
                     'orange', '$B_{\\phi} - \\mathrm{fit}$'),
                    (B_norm - fit_lines['norm'][0] if 'norm' in fit_lines else B_norm - B_par_sm,
                     'purple', '$B_{\\perp} - \\mathrm{fit}$'),
                ]
                fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
                for ax2, (resid, color, lbl) in zip(axes2, diff_data):
                    ax2.plot(t_seg, resid, color=color, lw=0.9)
                    ax2.axhline(0, color='k', lw=0.4, alpha=0.4)
                    ax2.axvline(t_part, color='grey', lw=1.2, ls='--', alpha=0.7)
                    ax2.set_ylim(-6, 6)
                    ax2.set_ylabel(f'{lbl} (nT)', fontsize=9)
                    ax2.grid(True, alpha=0.3)
                if is_sim:
                    import matplotlib.ticker as _mticker, datetime as _dt
                    _epoch_aware = _SIM_EPOCH.to_pydatetime().replace(tzinfo=_dt.timezone.utc)
                    axes2[-1].xaxis.set_major_formatter(
                        _mticker.FuncFormatter(
                            lambda x, _, _ep=_epoch_aware: f'{(mdates.num2date(x) - _ep).total_seconds():.0f}s'))
                else:
                    axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                fig2.autofmt_xdate(rotation=30, ha='right')
                fig2.suptitle(
                    f'{orb_label}{suffix}  —  FAC residuals  '
                    f'({time_label})',
                    fontsize=9)
                fig2.tight_layout()
                fname2    = f'orbit_{orb_tag}{suffix}_diff.png'
                out_path2 = os.path.join(save_dir, fname2)
                fig2.savefig(out_path2, dpi=dpi, bbox_inches='tight')
                plt.close(fig2)
                print(f'  Saved: {out_path2}')

            # ── wavelet spectrogram figure ────────────────────────────────
            if show_wavelet and len(t_seg) > 20:
                scales    = np.power(2, np.linspace(1, 7, 500))
                bandwidth = 6.0
                ut = t_seg.values   # numpy datetime64 array

                time_w, freq_w, _, psd_bt,  _ = wavelet_coef_psd(ut, Bmag,  scales, bandwidth)
                _,      _,      _, psd_phi, _ = wavelet_coef_psd(ut, B_phi-sm_phi, scales, bandwidth)

                # cone of influence — boundary frequency as a function of time
                t_s     = (time_w - time_w[0]).astype(float) / 1e9        # seconds from start
                t_end_s = (time_w[-1] - time_w).astype(float) / 1e9       # seconds from end
                d_s     = np.maximum(np.minimum(t_s, t_end_s), 1e-10)
                coi_freq_t = np.clip(np.sqrt(2) * bandwidth / (2 * np.pi) / d_s,
                                     freq_w[0], freq_w[-1])

                fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                for ax3, psd, title in [
                    (ax3a, psd_bt,  '|B| (nT²/Hz)'),
                    (ax3b, psd_phi, '$B_\\phi$ (nT²/Hz)'),
                ]:
                    pc = ax3.pcolormesh(time_w, freq_w, np.log10(np.clip(psd, 1e-6, None)),
                                        cmap='nipy_spectral', vmin=-2, vmax=3, shading='auto')
                    ax3.fill_between(time_w, freq_w[0], coi_freq_t,
                                     color='white', alpha=0.35, zorder=3)
                    ax3.axvline(t_part, color='white', lw=1.2, ls='--', alpha=0.8, zorder=4)
                    ax3.set_yscale('log')
                    ax3.set_ylabel('Frequency (Hz)', fontsize=9)
                    ax3.set_title(title, fontsize=9, loc='left')
                    ax3.grid(True, alpha=0.2, color='white')
                    plt.colorbar(pc, ax=ax3, label='log₁₀ PSD', pad=0.01)

                if is_sim:
                    import matplotlib.ticker as _mticker, datetime as _dt
                    _epoch_aware = _SIM_EPOCH.to_pydatetime().replace(tzinfo=_dt.timezone.utc)
                    ax3b.xaxis.set_major_formatter(
                        _mticker.FuncFormatter(
                            lambda x, _, _ep=_epoch_aware: f'{(mdates.num2date(x) - _ep).total_seconds():.0f}s'))
                else:
                    ax3b.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                fig3.autofmt_xdate(rotation=30, ha='right')
                fig3.suptitle(
                    f'{orb_label}{suffix}  —  wavelet spectrogram  '
                    f'({time_label})',
                    fontsize=9)
                fig3.tight_layout()
                fname3    = f'orbit_{orb_tag}{suffix}_wavelet.png'
                out_path3 = os.path.join(save_dir, fname3)
                fig3.savefig(out_path3, dpi=dpi, bbox_inches='tight')
                plt.close(fig3)
                print(f'  Saved: {out_path3}')

    print(f'\nDone — {n_saved} FAC figure(s) saved to {save_dir}')

    if not event_agg:
        return

    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    agg_lons = np.array([e['lon'] for e in event_agg])
    agg_lats = np.array([e['lat'] for e in event_agg])
    agg_vals = np.array([e['val'] for e in event_agg])

    abs_max  = max(abs(agg_vals.min()), abs(agg_vals.max()))
    norm_agg = Normalize(vmin=-abs_max, vmax=abs_max)
    cmap_agg = plt.cm.bwr
    clabel   = r'$\langle\delta B_\phi\rangle$ (nT)'
    phase_tag = f'{smooth_sec}s avg removed'

    # --- scatter in lon/lat ---
    fig_sc, ax_sc = plt.subplots(figsize=(10, 5))
    ax_sc.scatter(agg_lons, agg_lats, c=agg_vals, cmap=cmap_agg, norm=norm_agg,
                  s=60, zorder=3, edgecolors='k', lw=0.5)
    if label_events:
        for e in event_agg:
            ax_sc.text(e['lon'], e['lat'], e['key'],
                       fontsize=6, ha='left', va='bottom', alpha=0.8, clip_on=True)
    ax_sc.set_xlim(90, 270)
    ax_sc.set_ylim(-90, 90)
    ax_sc.set_xlabel('East Longitude (°)')
    ax_sc.set_ylabel('Latitude (°)')
    ax_sc.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_sc.grid(True, alpha=0.2)
    sm_sc = ScalarMappable(cmap=cmap_agg, norm=norm_agg)
    sm_sc.set_array([])
    fig_sc.colorbar(sm_sc, ax=ax_sc, label=clabel)
    fig_sc.suptitle(
        f'Mean $\\delta B_\\phi$ (unloading)  —  {phase_tag}  '
        f'({len(event_agg)} events)', fontsize=11)
    fig_sc.tight_layout()
    fig_sc.savefig(os.path.join(save_dir, f'dBphi_scatter_{smooth_sec}s.png'),
                   dpi=dpi, bbox_inches='tight')
    plt.show()

    # --- binned map in lon/lat ---
    lon_edges = np.linspace(90, 270, n_lon_bins + 1)
    lat_edges = np.linspace(-90, 90, n_lat_bins + 1)
    LON, LAT  = np.meshgrid(lon_edges, lat_edges)

    b_sum   = np.zeros((n_lat_bins, n_lon_bins))
    b_count = np.zeros((n_lat_bins, n_lon_bins), dtype=int)
    for lon_v, lat_v, val_v in zip(agg_lons, agg_lats, agg_vals):
        i = int(np.searchsorted(lon_edges, lon_v, side='right') - 1)
        j = int(np.searchsorted(lat_edges, lat_v, side='right') - 1)
        if 0 <= i < n_lon_bins and 0 <= j < n_lat_bins:
            b_sum[j, i]   += val_v
            b_count[j, i] += 1
    b_mean = np.where(b_count > 0, b_sum / b_count, np.nan)

    fig_bin, ax_bin = plt.subplots(figsize=(10, 5))
    pm = ax_bin.pcolormesh(LON, LAT, b_mean, cmap=cmap_agg, norm=norm_agg, shading='flat')
    for i in range(n_lon_bins):
        for j in range(n_lat_bins):
            if b_count[j, i] > 0:
                cx = 0.5 * (lon_edges[i] + lon_edges[i + 1])
                cy = 0.5 * (lat_edges[j] + lat_edges[j + 1])
                ax_bin.text(cx, cy, str(b_count[j, i]),
                            ha='center', va='center', fontsize=7, color='k')
    ax_bin.set_xlim(90, 270)
    ax_bin.set_ylim(-90, 90)
    ax_bin.set_xlabel('East Longitude (°)')
    ax_bin.set_ylabel('Latitude (°)')
    ax_bin.grid(True, alpha=0.2)
    fig_bin.colorbar(pm, ax=ax_bin, label=clabel)
    fig_bin.suptitle(
        f'Binned mean $\\delta B_\\phi$ (unloading)  —  {phase_tag}  '
        f'({n_lon_bins}×{n_lat_bins} bins, count labelled)', fontsize=11)
    fig_bin.tight_layout()
    fig_bin.savefig(os.path.join(save_dir, f'dBphi_binned_{smooth_sec}s.png'),
                    dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_event_map(event_keys=None, json_path=None, color_by_alt=False, color_by_category=False):
    """Plot YZ and lon/lat trajectory maps for a list of labelled events.

    If event_keys is None or empty, all events in the JSON are shown.

    Parameters
    ----------
    event_keys  : list[str] or None  e.g. ['3784a', '3784b', '3789']
    json_path   : str or None  defaults to human_loading_labels_new.json
    color_by_alt: bool  colour each segment by altitude (R_M above surface)
                        instead of a distinct colour per event
    color_by_category : bool  colour by the event's sort_loading_type category
                        ('a' or 'l') instead of a distinct colour per event;
                        events categorized 'n' are hidden, and uncategorized
                        events are skipped. Takes precedence over color_by_alt.
    """
    import json as _json
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if json_path is None:
        json_path = os.path.join(_SCRIPT_DIR, 'human_loading_labels_new.json')
    with open(json_path) as f:
        labels = _json.load(f)

    _letters = 'abcdefghijklmnopqrstuvwxyz'

    if not event_keys:
        # build key list from every event in the JSON
        event_keys = []
        for orb_str, entry in sorted(labels.items(), key=lambda kv: int(kv[0])):
            evs = entry.get('loading_events', [])
            for i in range(len(evs)):
                suffix = _letters[i] if len(evs) > 1 else ''
                event_keys.append(f'{orb_str}{suffix}')

    def _resolve(key):
        """key -> (orb, event_dict) or (orb, None) if not found."""
        key = str(key).strip()
        if key[-1].isalpha():
            orb, ev_idx = int(key[:-1]), _letters.index(key[-1])
        else:
            orb, ev_idx = int(key), 0
        events = labels.get(str(orb), {}).get('loading_events', [])
        return orb, (events[ev_idx] if ev_idx < len(events) else None)

    _CATEGORY_COLORS = {'a': 'tab:red', 'l': 'tab:blue'}

    if color_by_category:
        filtered_keys = []
        event_colors  = []
        for key in event_keys:
            _, ev = _resolve(key)
            cat = ev.get('category') if ev else None
            if cat not in _CATEGORY_COLORS:   # drops 'n', missing, and unresolved keys
                continue
            filtered_keys.append(key)
            event_colors.append(_CATEGORY_COLORS[cat])
        event_keys = filtered_keys
    else:
        event_cmap   = plt.cm.tab10
        event_colors = [event_cmap(i % 10) for i in range(len(event_keys))]

    fig, (ax_xz, ax_yz, ax_ll) = plt.subplots(1, 3, figsize=(18, 5))

    alt_cmap = plt.cm.plasma
    all_alts = []   # collected to set shared norm after first pass
    color_by_alt = color_by_alt and not color_by_category

    # two-pass when color_by_alt: first collect all altitudes, then plot
    segments_xz, segments_yz, segments_ll, seg_alts = [], [], [], []

    for ev_color, key in zip(event_colors, event_keys):
        key = str(key).strip()
        if key[-1].isalpha():
            orb    = int(key[:-1])
            ev_idx = _letters.index(key[-1])
        else:
            orb    = int(key)
            ev_idx = 0

        entry  = labels.get(str(orb), {})
        events = entry.get('loading_events', [])
        if ev_idx >= len(events):
            print(f'  {key}: event index {ev_idx} not found, skipping')
            continue

        ev      = events[ev_idx]
        t_start = pd.Timestamp(ev['start'])
        t_stop  = pd.Timestamp(ev['stop'])

        try:
            orb_df = load_bowers_data_pkl(orbit_number=orb)
        except Exception as exc:
            print(f'  {key}: {exc}')
            continue

        t_orb = pd.to_datetime(orb_df['time'])
        seg   = orb_df[(t_orb >= t_start) & (t_orb <= t_stop)]
        if seg.empty:
            print(f'  {key}: empty segment')
            continue

        Y = seg['ephy'].to_numpy(dtype=float)
        Z = seg['ephz'].to_numpy(dtype=float)
        X = seg['ephx'].to_numpy(dtype=float)
        r   = np.sqrt(X**2 + Y**2 + Z**2)
        alt = r - 1.0                                           # altitude in R_M
        lat = np.degrees(np.arcsin(np.clip(Z / r, -1, 1)))
        lon = np.degrees(np.arctan2(Y, X)) % 360

        if color_by_alt:
            segments_xz.append((X, Z, alt))
            segments_yz.append((Y, Z, alt))
            segments_ll.append((lon, lat, alt))
            all_alts.append(alt)
        else:
            label = ev.get('category') if color_by_category else key
            ax_xz.plot(X, Z, color=ev_color, lw=1.5, label=label)
            ax_xz.scatter(X[0],  Z[0],  color=ev_color, marker='o', s=20, zorder=5)
            ax_xz.scatter(X[-1], Z[-1], color=ev_color, marker='s', s=20, zorder=5)
            ax_yz.plot(Y, Z, color=ev_color, lw=1.5, label=label)
            ax_yz.scatter(Y[0],  Z[0],  color=ev_color, marker='o', s=20, zorder=5)
            ax_yz.scatter(Y[-1], Z[-1], color=ev_color, marker='s', s=20, zorder=5)
            ax_ll.plot(lon, lat, color=ev_color, lw=1.5, label=label)
            ax_ll.scatter(lon[0],  lat[0],  color=ev_color, marker='o', s=20, zorder=5)
            ax_ll.scatter(lon[-1], lat[-1], color=ev_color, marker='s', s=20, zorder=5)

    if color_by_alt and all_alts:
        all_alts_cat = np.concatenate(all_alts)
        norm = Normalize(vmin=np.nanmin(all_alts_cat), vmax=np.nanmax(all_alts_cat))

        def _add_lc(ax, xs, ys, vals):
            pts  = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc   = LineCollection(segs, cmap=alt_cmap, norm=norm, lw=1.5)
            lc.set_array(0.5 * (vals[:-1] + vals[1:]))
            ax.add_collection(lc)
            return lc

        for X, Z, alt in segments_xz:
            _add_lc(ax_xz, X, Z, alt)
        for Y, Z, alt in segments_yz:
            _add_lc(ax_yz, Y, Z, alt)
        for lon, lat, alt in segments_ll:
            _add_lc(ax_ll, lon, lat, alt)

        sm = ScalarMappable(cmap=alt_cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=[ax_xz, ax_yz, ax_ll], label='Altitude (R$_M$)',
                     fraction=0.02, pad=0.02)

    # XZ panel — Mercury outline (MSM: planet centre at Z = -0.2 R_M)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax_xz.fill(np.cos(theta), -0.2 + np.sin(theta),
               color='saddlebrown', alpha=0.3, zorder=0)
    ax_xz.plot(np.cos(theta), -0.2 + np.sin(theta),
               color='saddlebrown', lw=1, zorder=1)
    ax_xz.set_xlabel('X$_{MSM}$ (R$_M$)')
    ax_xz.set_ylabel('Z$_{MSM}$ (R$_M$)')
    ax_xz.set_aspect('equal')
    ax_xz.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_xz.axvline(0, color='k', lw=0.4, alpha=0.3)
    ax_xz.grid(True, alpha=0.2)
    ax_xz.invert_xaxis()

    # YZ panel — Mercury outline (MSM: planet centre at Z = -0.2 R_M)
    ax_yz.fill(np.cos(theta), -0.2 + np.sin(theta),
               color='saddlebrown', alpha=0.3, zorder=0)
    ax_yz.plot(np.cos(theta), -0.2 + np.sin(theta),
               color='saddlebrown', lw=1, zorder=1)
    ax_yz.set_xlabel('Y$_{MSM}$ (R$_M$)')
    ax_yz.set_ylabel('Z$_{MSM}$ (R$_M$)')
    ax_yz.set_aspect('equal')
    ax_yz.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_yz.axvline(0, color='k', lw=0.4, alpha=0.3)
    ax_yz.grid(True, alpha=0.2)
    ax_yz.set_xlim(1.5, -1.5)

    # lon/lat panel
    ax_ll.set_xlabel('East Longitude (°)')
    ax_ll.set_ylabel('Latitude (°)')
    ax_ll.set_xlim(90, 270)
    ax_ll.set_ylim(-80, 80)
    ax_ll.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_ll.grid(True, alpha=0.2)

    def _dedup_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles))   # keeps one handle per unique label
        ax.legend(seen.values(), seen.keys(), fontsize=8)

    if color_by_category:
        _dedup_legend(ax_xz)
        _dedup_legend(ax_yz)
        _dedup_legend(ax_ll)
    elif len(event_keys) <= 8:
        ax_xz.legend(fontsize=8)
        ax_yz.legend(fontsize=8)
        ax_ll.legend(fontsize=8)

    fig.suptitle('Event trajectory segments', fontsize=11)
    fig.tight_layout()
    _fig_dir = os.path.join(_SCRIPT_DIR, 'figures')
    os.makedirs(_fig_dir, exist_ok=True)
    fig.savefig(os.path.join(_fig_dir, 'event_map.png'), dpi=150, bbox_inches='tight')
    plt.show()

#browse_southward_orbits(1200, 2000, species=['H+'])
#label_southward_orbits(944, 946, species=["H+"],json_path="equatorial_events.json", direction='south')

#plot_labelled_events(species=['H+'],json_path="equatorial_events.json", pad_sec=90, show_kt17=True, pad_from_partition=True, show_kt17_residual=False, figsize = (7,8))

measure_bz_recovery(json_path="equatorial_events.json", species=['H+'])
#sort_loading_type(json_path="equatorial_events.json", species=["H+"], only_undesignated=False, partition_pad=30, show_kt17_residual=False, figsize = (12,8),only_show_type='a')


#_events = json.load(open(os.path.join(_SCRIPT_DIR, 'equatorial_events.json')))
#for _orb in sorted(int(k) for k, v in _events.items() if v.get('loading_events')):
#    extract_traj(_orb, events_json='equatorial_events.json')

#plot_fac_events(b0_ref='start', smooth_sec=30, label_events=False, show_wavelet=False )
#plot_event_map(['3784a', '3772c', '4035', '3772b', '3789', '3783'])
#plot_event_map(json_path="equatorial_events.json", color_by_category=True)

'''
df, label = load_simulation_json(
      'x-1.61_y0.35_z0.37_t110_to_300.json',
      t_start_s=175,
      t_stop_s=300,
      t_partition_s=240)
plot_fac_events(b0_ref='start', sim_data={'x-1.1_y0_z0.5': (df,label)},smooth_sec=30,show_wavelet=True, label_events=False)
'''

# ---------------------------------------------------------------------------
# MVA analysis
# ---------------------------------------------------------------------------

def _normvec(v):
    """Normalise rows of a 2D array to unit vectors."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms > 0, norms, 1.0)

def mva(v, cdir=None):
    # returns results of MVA on the array of 3D vector v. If cdir (3D vector)
    # is provided, the analysis is performed in the plane perpendicular to it
    v_cov = np.cov(v, rowvar=False, bias=True)
    if cdir is not None:
        ccol  = cdir.reshape(3, 1)
        cunit = ccol / np.linalg.norm(ccol)
        d_mat = np.identity(3) - np.dot(cunit, cunit.T)
        v_cov = np.matmul(np.matmul(d_mat, v_cov), d_mat)
    return np.linalg.eigh(v_cov)

def plot_mva_map(event_keys=None, json_path=None, background='linear', phase='full',
                 color_by='dB_min', clim=None, n_lon_bins=16, n_lat_bins=10):
    """Map MVA results for labelled events in XY and lon/lat space.

    Runs the same MVA pipeline as run_orbit_mva for every event and plots
    each trajectory segment coloured by a scalar derived from the MVA result.

    Parameters
    ----------
    event_keys  : list[str] or None  — subset of events; None = all in JSON
    json_path   : str or None        — defaults to human_loading_labels_new.json
    background  : 'kt17' or 'linear' — same as run_orbit_mva
    phase       : 'full' or 'unloading' — same as run_orbit_mva
    color_by    : 'ratio'     — eigenvalue ratio λ_max / λ_min  (MVA quality)
                  'dB_min'   — RMS of dB projected onto min-var direction
    clim        : (vmin, vmax) or None — override the auto colorbar limits
    """
    import json as _json
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, LogNorm

    if json_path is None:
        json_path = os.path.join(_SCRIPT_DIR, 'human_loading_labels_new.json')
    with open(json_path) as f:
        labels = _json.load(f)

    _letters = 'abcdefghijklmnopqrstuvwxyz'

    if not event_keys:
        event_keys = []
        for orb_str, entry in sorted(labels.items(), key=lambda kv: int(kv[0])):
            evs = entry.get('loading_events', [])
            for i in range(len(evs)):
                suffix = _letters[i] if len(evs) > 1 else ''
                event_keys.append(f'{orb_str}{suffix}')

    segments_yz, segments_ll, all_pt_vals = [], [], []
    event_summaries = []  # per-event: mean position, mean scalar, min-var direction in YZ

    for key in event_keys:
        key = str(key).strip()
        if key[-1].isalpha():
            orb_int = int(key[:-1])
            ev_idx  = _letters.index(key[-1])
        else:
            orb_int = int(key)
            ev_idx  = 0

        entry  = labels.get(str(orb_int), {})
        events = entry.get('loading_events', [])
        if ev_idx >= len(events):
            print(f'  {key}: not found, skipping')
            continue

        ev      = events[ev_idx]
        t_start = pd.Timestamp(ev['start'])
        t_part  = pd.Timestamp(ev.get('partition', ev['start']))
        t_stop  = pd.Timestamp(ev['stop'])
        t_win   = t_part if phase == 'unloading' else t_start

        try:
            orb_df = load_bowers_data_pkl(orbit_number=orb_int)
        except Exception as exc:
            print(f'  {key}: {exc}')
            continue

        t_orb = pd.to_datetime(orb_df['time'])
        seg   = orb_df[(t_orb >= t_win) & (t_orb <= t_stop)]
        if len(seg) < 10:
            continue

        B = np.column_stack([seg['magx'].to_numpy(dtype=float),
                              seg['magy'].to_numpy(dtype=float),
                              seg['magz'].to_numpy(dtype=float)])
        if background.lower() == 'linear':
            t_frac = np.linspace(0.0, 1.0, len(seg))
            B_bg   = B[0] + t_frac[:, None] * (B[-1] - B[0])
        else:
            _, Bxkt, Bykt, Bzkt = get_kt17_along_track(df=seg)
            B_bg = np.column_stack([Bxkt, Bykt, Bzkt])
        dB = B - B_bg

        indok  = np.where(~np.isnan(dB[:, 0]))[0]
        if len(indok) < 5:
            continue
        dB_int = dB[indok]
        B_int  = B[indok]
        B_ave  = np.average(B_int, axis=0)
        B_unit = B_ave / np.linalg.norm(B_ave)
        eigval, eigvec = mva(dB_int, cdir=B_unit)
        # enforce sign convention: min-var direction points toward -Y
        if eigvec[1, 1] > 0:
            eigvec[:, 1] = -eigvec[:, 1]

        if color_by == 'ratio':
            scalar = eigval[2] / (abs(eigval[1]) + 1e-9)
            vals_pt = np.full(len(seg), scalar)
        else:  # dB_min — per-point projection onto min-var direction
            vals_pt = dB @ eigvec[:, 1]  # shape (N,) — all seg points

        X  = seg['ephx'].to_numpy(dtype=float)
        Y  = seg['ephy'].to_numpy(dtype=float)
        Z  = seg['ephz'].to_numpy(dtype=float)
        r   = np.sqrt(X**2 + Y**2 + Z**2)
        lat = np.degrees(np.arcsin(np.clip(Z / r, -1, 1)))
        lon = np.degrees(np.arctan2(Y, X)) % 360

        segments_yz.append((Y, Z, vals_pt))
        segments_ll.append((lon, lat, vals_pt))
        finite = vals_pt[np.isfinite(vals_pt)]
        all_pt_vals.extend(finite.tolist())

        mindir = eigvec[:, 1]
        event_summaries.append({
            'Y':   float(np.nanmean(Y)),
            'Z':   float(np.nanmean(Z)),
            'lon': float(np.nanmean(lon)),
            'lat': float(np.nanmean(lat)),
            'val': float(np.nanmean(finite)) if len(finite) else 0.0,
            'dy':  float(mindir[1]),
            'dz':  float(mindir[2]),
        })

    if not segments_yz:
        print('No events to plot.')
        return

    all_arr = np.array(all_pt_vals)
    if color_by == 'ratio':
        vmin = clim[0] if clim is not None else max(all_arr.min(), 0.1)
        vmax = clim[1] if clim is not None else all_arr.max()
        norm   = LogNorm(vmin=vmin, vmax=vmax)
        clabel = r'$\lambda_{max}/\lambda_{min}$'
        cmap   = plt.cm.plasma
    else:
        if clim is not None:
            vmin, vmax = clim
        else:
            abs_max = max(abs(all_arr.min()), abs(all_arr.max()))
            vmin, vmax = -abs_max, abs_max
        norm   = Normalize(vmin=vmin, vmax=vmax)
        clabel = r'$\delta B_{min}$ (nT)'
        cmap   = plt.cm.bwr

    fig, (ax_yz, ax_ll) = plt.subplots(1, 2, figsize=(13, 5))

    def _add_lc(ax, xs, ys, vals):
        pts  = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap=cmap, norm=norm, lw=1.8)
        lc.set_array(0.5 * (vals[:-1] + vals[1:]))
        ax.add_collection(lc)

    for Y, Z, vals in segments_yz:
        _add_lc(ax_yz, Y, Z, vals)
    for lon, lat, vals in segments_ll:
        _add_lc(ax_ll, lon, lat, vals)

    # YZ panel — Mercury outline (MSM: planet centre at Z = -0.2 R_M)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax_yz.fill(np.cos(theta), -0.2 + np.sin(theta),
               color='saddlebrown', alpha=0.3, zorder=0)
    ax_yz.plot(np.cos(theta), -0.2 + np.sin(theta),
               color='saddlebrown', lw=1, zorder=1)
    ax_yz.set_xlabel('Y$_{MSM}$ (R$_M$)')
    ax_yz.set_ylabel('Z$_{MSM}$ (R$_M$)')
    ax_yz.set_aspect('equal')
    ax_yz.set_xlim(1.5, -1.5)   # +Y on the left
    ax_yz.autoscale_view()
    ax_yz.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_yz.axvline(0, color='k', lw=0.4, alpha=0.3)
    ax_yz.grid(True, alpha=0.2)

    # lon/lat panel
    ax_ll.set_xlabel('East Longitude (°)')
    ax_ll.set_ylabel('Latitude (°)')
    ax_ll.set_xlim(100, 260)
    ax_ll.set_ylim(-20,60)
    ax_ll.set_aspect(1)
    ax_ll.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_ll.grid(True, alpha=0.2)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig.tight_layout()
    fig.subplots_adjust(right=0.87)
    cax = fig.add_axes([0.90, 0.12, 0.018, 0.74])
    fig.colorbar(sm, cax=cax, label=clabel)

    phase_lbl = phase.capitalize()
    bg_lbl    = 'KT17' if background.lower() == 'kt17' else 'Linear'
    fig.suptitle(f'MVA map  —  {phase_lbl} / {bg_lbl} background  '
                 f'({len(segments_yz)} events)', fontsize=11)
    _fig_dir = os.path.join(_SCRIPT_DIR, 'figures')
    os.makedirs(_fig_dir, exist_ok=True)
    fig.savefig(os.path.join(_fig_dir, f'mva_map_{phase}_{background}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    # --- Second figure: per-event scatter + min-var direction ---
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    Ys   = np.array([e['Y']   for e in event_summaries])
    Zs   = np.array([e['Z']   for e in event_summaries])
    vals = np.array([e['val'] for e in event_summaries])
    dys  = np.array([e['dy']  for e in event_summaries])
    dzs  = np.array([e['dz']  for e in event_summaries])

    # quiver: sign already enforced to point toward -Y; length = YZ-plane fraction
    arrow_len = 0.35  # full length when min-var dir lies entirely in YZ plane
    ax2.quiver(Ys, Zs, dys * arrow_len, dzs * arrow_len,
               angles='xy', scale_units='xy', scale=1,
               pivot='mid', color='k', width=0.004,
               headwidth=4, headlength=5, zorder=3)

    ax2.scatter(Ys, Zs, c=vals, cmap=cmap, norm=norm,
                s=60, zorder=4, edgecolors='k', lw=0.5)

    theta = np.linspace(0, 2 * np.pi, 300)
    ax2.fill(np.cos(theta), -0.2 + np.sin(theta), color='saddlebrown', alpha=0.3, zorder=0)
    ax2.plot(np.cos(theta), -0.2 + np.sin(theta), color='saddlebrown', lw=1, zorder=1)
    ax2.set_xlabel('Y$_{MSM}$ (R$_M$)')
    ax2.set_ylabel('Z$_{MSM}$ (R$_M$)')
    ax2.set_aspect('equal')
    ax2.set_xlim(1.5, -1.5)
    ax2.set_ylim(-0.5,1.25)
    ax2.autoscale_view()
    ax2.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax2.axvline(0, color='k', lw=0.4, alpha=0.3)
    ax2.grid(True, alpha=0.2)

    sm2 = ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    fig2.colorbar(sm2, ax=ax2, label=clabel)
    fig2.suptitle(f'MVA summary  —  {phase_lbl} / {bg_lbl}\n'
                  f'dot = mean $\\delta B_{{min}}$, arrow = min-var direction (YZ projection)',
                  fontsize=10)
    fig2.tight_layout()
    fig2.savefig(os.path.join(_fig_dir, f'mva_map_summary_{phase}_{background}.png'),
                 dpi=150, bbox_inches='tight')
    plt.show()

    # --- Figure 3: 2-D binned mean in lon/lat ------------------------------------
    ev_lons = np.array([e['lon'] for e in event_summaries])
    ev_lats = np.array([e['lat'] for e in event_summaries])
    ev_vals = np.array([e['val'] for e in event_summaries])

    lon_edges = np.linspace(90, 270, n_lon_bins + 1)
    lat_edges = np.linspace(-90, 90, n_lat_bins + 1)
    LON, LAT  = np.meshgrid(lon_edges, lat_edges)

    b_sum   = np.zeros((n_lat_bins, n_lon_bins))
    b_count = np.zeros((n_lat_bins, n_lon_bins), dtype=int)
    for lon_v, lat_v, val_v in zip(ev_lons, ev_lats, ev_vals):
        i = int(np.searchsorted(lon_edges, lon_v, side='right') - 1)
        j = int(np.searchsorted(lat_edges, lat_v, side='right') - 1)
        if 0 <= i < n_lon_bins and 0 <= j < n_lat_bins:
            b_sum[j, i]   += val_v
            b_count[j, i] += 1
    b_mean = np.where(b_count > 0, b_sum / b_count, np.nan)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    pm = ax3.pcolormesh(LON, LAT, b_mean, cmap=cmap, norm=norm, shading='flat')
    for i in range(n_lon_bins):
        for j in range(n_lat_bins):
            if b_count[j, i] > 0:
                cx = 0.5 * (lon_edges[i] + lon_edges[i + 1])
                cy = 0.5 * (lat_edges[j] + lat_edges[j + 1])
                ax3.text(cx, cy, str(b_count[j, i]),
                         ha='center', va='center', fontsize=7, color='k')
    ax3.set_xlim(90, 270)
    ax3.set_ylim(-90, 90)
    ax3.set_xlabel('Longitude (deg, MSM)')
    ax3.set_ylabel('Latitude (deg, MSM)')
    ax3.grid(True, alpha=0.2)
    fig3.colorbar(pm, ax=ax3, label=clabel)
    fig3.suptitle(f'MVA binned mean  —  {phase_lbl} / {bg_lbl}  '
                  f'({n_lon_bins}×{n_lat_bins} bins, count labelled)', fontsize=11)
    fig3.tight_layout()
    fig3.savefig(os.path.join(_fig_dir, f'mva_map_binned_{phase}_{background}.png'),
                 dpi=150, bbox_inches='tight')
    plt.show()

def run_orbit_mva(orb, json_path=None, save_dir=None, use_filter=False, dpi=150,
                  background='linear', phase='full'):
    """Run constrained MVA on labelled event(s) for orbit `orb`.

    `orb` can be an orbit number (runs all events for that orbit) or a
    specific event key string such as '3566a' (runs only that event).

    Parameters
    ----------
    orb        : int or str  orbit number, or key like '3566a'
    json_path  : str   label file  (default: human_loading_labels_new.json)
    save_dir   : str   output dir  (default: figures/mva/)
    use_filter : bool  apply bandpass filter to dB before FAC estimate
    dpi        : int
    background : 'kt17' or 'linear'
                 'kt17'   — subtract KT17 model field (default)
                 'linear' — subtract a straight-line interpolation of Bx/By/Bz
                            between the first and last samples of the analysis window
    phase      : 'full' or 'unloading'
                 'full'      — use the entire event (start → stop)
                 'unloading' — use only the unloading phase (partition → stop)
    """
    import json as _json

    _letters = 'abcdefghijklmnopqrstuvwxyz'

    # parse key: '3566a' → orb=3566, only_ev=0;  3566 → orb=3566, only_ev=None
    key = str(orb).strip()
    if key[-1].isalpha():
        orb_int  = int(key[:-1])
        only_ev  = _letters.index(key[-1])
    else:
        orb_int  = int(key)
        only_ev  = None

    if json_path is None:
        json_path = os.path.join(_SCRIPT_DIR, 'human_loading_labels_new.json')
    if save_dir is None:
        save_dir = os.path.join(_SCRIPT_DIR, 'figures', 'mva')
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path) as f:
        labels = _json.load(f)

    events = labels.get(str(orb_int), {}).get('loading_events', [])
    if not events:
        print(f'Orbit {orb_int}: no labelled events found in {json_path}')
        return
    if only_ev is not None:
        if only_ev >= len(events):
            print(f'{key}: event index {only_ev} out of range ({len(events)} events)')
            return
        events = [events[only_ev]]

    orb_df = load_bowers_data_pkl(orbit_number=orb_int)

    for ev_idx, ev in enumerate(events):
        if only_ev is not None:
            ev_idx = only_ev
        t_start = pd.Timestamp(ev['start'])
        t_part  = pd.Timestamp(ev.get('partition', ev['start']))
        t_stop  = pd.Timestamp(ev['stop'])

        # select analysis window based on phase
        t_win_start = t_part if phase == 'unloading' else t_start

        t_orb = pd.to_datetime(orb_df['time'])
        mask  = (t_orb >= t_win_start) & (t_orb <= t_stop)
        seg   = orb_df[mask]

        if len(seg) < 10:
            print(f'  Orbit {orb_int} event {ev_idx}: too few points ({len(seg)}), skipped')
            continue

        # ── variable translations ─────────────────────────────────────────
        ti = pd.to_datetime(seg['time']).values.astype('datetime64[ns]')
        B  = np.column_stack([seg['magx'].to_numpy(dtype=float),
                               seg['magy'].to_numpy(dtype=float),
                               seg['magz'].to_numpy(dtype=float)])
        if background.lower() == 'linear':
            # interpolate between first and last sample of the analysis window
            t_frac = np.linspace(0.0, 1.0, len(seg))
            B_bg   = B[0] + t_frac[:, None] * (B[-1] - B[0])
        else:
            _, Bxkt, Bykt, Bzkt = get_kt17_along_track(df=seg)
            B_bg = np.column_stack([Bxkt, Bykt, Bzkt])
        dB = B - B_bg

        R  = np.column_stack([seg['ephx'].to_numpy(dtype=float),
                               seg['ephy'].to_numpy(dtype=float),
                               seg['ephz'].to_numpy(dtype=float)])

        dtime_beg = np.datetime64(t_start, 'ns')
        dtime_end = np.datetime64(t_stop,  'ns')

        # spacecraft velocity (finite-difference of position, R_M/s) — used only
        # to orient mindir sign consistently along the spacecraft track
        t_s   = (ti - ti[0]).astype('float64') / 1e9
        dt_s  = np.diff(t_s)
        v_raw = np.diff(R, axis=0) / dt_s[:, None]
        eV3d  = np.vstack([v_raw, v_raw[-1:]])            # pad last → shape (N, 3)

        dB_df = pd.DataFrame(dB, columns=['dBx', 'dBy', 'dBz'],
                              index=pd.DatetimeIndex(ti))

        # ── MVA (unchanged from reference) ────────────────────────────────
        # update MVA interval if span has been selected
        tmva_int = np.array([dtime_beg, dtime_end], dtype='datetime64')
        # select quantities for MVA interval and remove NaN points
        indok = np.where(
            (ti >= tmva_int[0]) & (ti <= tmva_int[1]) & ~np.isnan(dB[:, 0])
        )[0]
        dB_int, B_int = dB[indok, :], B[indok, :]
        B_ave  = np.average(B_int, axis=0)
        B_unit = B_ave / np.linalg.norm(B_ave)
        # apply constrained MVA
        eigval, eigvec = mva(dB_int, cdir=B_unit)
        # select the minvar orientation according to sat. velocity
        eV3d_ave = np.average(eV3d[indok[:-1]], axis=0)
        mindir   = eigvec[:, 1]
        if np.sum(mindir * eV3d_ave) < 0:
            mindir = -eigvec[:, 1]
        maxdir = np.cross(B_unit, mindir)

        print('MVA interval: ', tmva_int)
        print('B_unit= %10.2f' % eigval[0], np.round(B_unit, decimals=4))
        print('mindir= %10.2f' % eigval[1], np.round(mindir, decimals=4))
        print('maxdir= %10.2f' % eigval[2], np.round(maxdir, decimals=4))

        # ── coordinate transform (unchanged from reference) ───────────────
        # transform magnetic perturbation in MVA frame
        geo2mva  = np.stack((B_unit, mindir, maxdir), axis=1)
        dBmva_df = pd.DataFrame(np.matmul(dB_df.values, geo2mva),
                                 columns=['dB B', 'dB min', 'dB max'],
                                 index=pd.DatetimeIndex(ti))

        # ── plot ──────────────────────────────────────────────────────────
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))

        ax1.plot(dBmva_df.index, dBmva_df['dB B'],   color='blue',  lw=0.9, label='dB ∥ B')
        ax1.plot(dBmva_df.index, dBmva_df['dB min'], color='red',   lw=0.9, label='dB min-var')
        ax1.plot(dBmva_df.index, dBmva_df['dB max'], color='green', lw=0.9, label='dB max-var')
        ax1.axhline(0, color='k', lw=0.4, alpha=0.4)
        ax1.set_ylabel('ΔB (nT)')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        lbl = (f'λ_max={eigval[2]:.2f}  λ_min={eigval[1]:.2f}  '
               f'λ_max/λ_min={eigval[2]/(abs(eigval[1])+1e-9):.1f}')
        ax1.set_title(lbl, fontsize=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate(rotation=30, ha='right')

        suffix = _letters[ev_idx] if only_ev is None and len(events) > 1 else (
                 _letters[ev_idx] if only_ev is not None else '')
        def _fv(v):
            return '[{:.2f}, {:.2f}, {:.2f}]'.format(*v)

        fig.suptitle(
            f'Orbit {orb_int}{suffix}  —  MVA  '
            f'({t_start.strftime("%Y-%m-%d %H:%M")} – {t_stop.strftime("%H:%M")} UTC)\n'
            f'B_unit={_fv(B_unit)}  mindir={_fv(mindir)}  maxdir={_fv(maxdir)}',
            fontsize=8)
        fig.tight_layout()

        fname    = f'orbit_{orb_int:05d}{suffix}_mva.png'
        out_path = os.path.join(save_dir, fname)
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out_path}')

#run_orbit_mva('3527',background='KT17',phase='unloading')

#plot_mva_map(background='linear',phase='full',clim=[-15,15],
#             json_path='human_loading_labels.json')

# ---------------------------------------------------------------------------
# Plot trajecotries of two orbits, for seminar
# ---------------------------------------------------------------------------

def plot_orbit_xz(orbits, colors=None, labels=None):
    """Plot orbit trajectories for a list of orbit numbers in XZ and YZ planes.

    Parameters
    ----------
    orbits : list[int]   orbit numbers to plot
    colors : list[str] or None   line colours (cycles through default if None)
    labels : list[str] or None   legend labels (orbit number used if None)
    """
    fig, (ax_xz, ax_yz) = plt.subplots(1, 2, figsize=(12, 6))

    theta = np.linspace(0, 2 * np.pi, 300)

    # load all orbits first so we can do layered drawing in YZ
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    orbit_data = []
    for i, orb in enumerate(orbits):
        color = colors[i] if colors else default_colors[i % len(default_colors)]
        label = labels[i] if labels else f'Orbit {orb}'
        try:
            df = load_bowers_data_pkl(orbit_number=int(orb))
        except Exception as exc:
            print(f'Orbit {orb}: {exc}')
            continue
        date_str = pd.to_datetime(df['time'].iloc[0]).strftime('%Y-%m-%d')
        orbit_data.append((
            df['ephx'].to_numpy(dtype=float),
            df['ephy'].to_numpy(dtype=float),
            df['ephz'].to_numpy(dtype=float),
            color, f'{label}  ({date_str})',
        ))

    # XZ panel — planet, then full orbits on top
    ax_xz.fill(np.cos(theta), -0.2 + np.sin(theta), color='grey', alpha=0.8, zorder=1)
    ax_xz.plot(np.cos(theta), -0.2 + np.sin(theta), color='grey', lw=1, zorder=2)
    for x, y, z, color, label in orbit_data:
        ax_xz.plot(x, z, color=color, lw=1.4, label=label, zorder=3)

    # YZ panel — three layers: dayside behind planet, planet, nightside in front
    for x, y, z, color, label in orbit_data:
        y_day = np.where(x > 0, y, np.nan)
        z_day = np.where(x > 0, z, np.nan)
        ax_yz.plot(y_day, z_day, color=color, lw=1.4, zorder=1)

    ax_yz.fill(np.cos(theta), -0.2 + np.sin(theta), color='grey', alpha=0.8, zorder=2)
    ax_yz.plot(np.cos(theta), -0.2 + np.sin(theta), color='grey', lw=1, zorder=2)

    for x, y, z, color, label in orbit_data:
        y_night = np.where(x <= 0, y, np.nan)
        z_night = np.where(x <= 0, z, np.nan)
        ax_yz.plot(y_night, z_night, color=color, lw=1.4, label=label, zorder=3)

    ax_xz.set_xlabel('X$_{MSM}$ (R$_M$)')
    ax_xz.set_ylabel('Z$_{MSM}$ (R$_M$)')
    ax_xz.set_aspect('equal')
    ax_xz.set_xlim(-0.25,-1.5)
    ax_xz.set_ylim(-1,1.1)

    ax_yz.set_xlabel('Y$_{MSM}$ (R$_M$)')
    ax_yz.set_ylabel('Z$_{MSM}$ (R$_M$)')
    ax_yz.set_xlim(1.5, -1.5)   # +Y on the left, consistent with plot_event_map
    ax_yz.set_aspect('equal')
    ax_yz.set_xlim(1.125,-1.125)
    ax_yz.set_ylim(-1,1.25)

    for ax in (ax_xz, ax_yz):
        ax.axhline(0, color='k', lw=0.4, alpha=0.3)
        ax.axvline(0, color='k', lw=0.4, alpha=0.3)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)

    fig.tight_layout()
    plt.show()

#plot_orbit_xz([3964, 3709])

# ---------------------------------------------------------------------------
# Biot-Savart field perturbation tools
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

def plot_dipole_field_line(lon_deg=180.0, L=2.0, n_pts=500, rsun=0.387, dist_index=50.0,
                           obs_point=None, I0=None,
                           lon_range=None, n_lines=5, dip_fact = 1):
    """Plot dipole and KT17 field lines and optionally compute the Biot-Savart perturbation.

    Pass either a single `lon_deg` or a `lon_range=(lo, hi)` with `n_lines` evenly-spaced
    field lines.  When an observation point and current are given, contributions from all
    field lines are summed.

    Parameters
    ----------
    lon_deg : float
        Equatorial crossing longitude (degrees, +X toward +Y; 180° = nightside).
        Ignored when lon_range is provided.
    L : float
        L-shell equatorial distance (R_M, MSM).
    n_pts : int
        Points along each analytic dipole field line.
    rsun : float
        Mercury–Sun distance in AU for KT17 (default 0.387).
    dist_index : float
        KT17 disturbance index 0–97 (default 50).
    obs_point : array-like (3,), optional
        Observation point [x, y, z] in R_M (MSM).
    I0 : float, optional
        Current per field line in Amperes.
    lon_range : (float, float), optional
        (lon_min, lon_max) in degrees.  Overrides lon_deg; n_lines are traced.
    n_lines : int
        Number of field lines when lon_range is given (default 5).
    dip_fact: float
        Enhancement factor of Bz_dip

    Returns
    -------
    fig, (ax_yz, ax_xz), all_dipole, all_kt17
        all_dipole / all_kt17 : list of xyz arrays, one per field line
    """
    lons = np.linspace(lon_range[0], lon_range[1], n_lines) if lon_range is not None \
           else np.array([lon_deg])
    multi = len(lons) > 1
    lam   = np.linspace(-np.pi / 2, np.pi / 2, n_pts)

    # ---- compute all field lines --------------------------------------------
    all_dipole, all_kt17 = [], []
    field_lines = []   # list of dicts with plot-ready data per longitude

    for lon in lons:
        phi  = np.radians(lon)
        x_eq = L * np.cos(phi)
        y_eq = L * np.sin(phi)

        # analytic dipole
        x_d = L * np.cos(lam) ** 3 * np.cos(phi)
        y_d = L * np.cos(lam) ** 3 * np.sin(phi)
        z_d = L * np.cos(lam) ** 2 * np.sin(lam) * dip_fact
        ok_d = np.sqrt(x_d**2 + y_d**2 + (z_d + 0.2)**2) >= 0.8
        xyz_d = np.column_stack([x_d, y_d, z_d])[ok_d]
        all_dipole.append(xyz_d)

        # KT17 trace
        T  = KT17.TraceField(x_eq, y_eq, 0.0,
                             Rsun=rsun, DistIndex=dist_index,
                             EndSurface=2, TraceDir=0)
        tr = T.GetTrace(0)
        xk, yk, zk = np.array(tr['x']), np.array(tr['y']), np.array(tr['z'])
        ok_k = np.sqrt(xk**2 + yk**2 + (zk + 0.2)**2) >= 0.8
        xyz_k = np.column_stack([xk, yk, zk])[ok_k]
        all_kt17.append(xyz_k)

        def _fp(arr, mask):
            idx = np.where(mask)[0]
            if len(idx) < 2:
                return np.empty((0, 3))
            return np.array([[arr[0, idx[0]], arr[1, idx[0]], arr[2, idx[0]]],
                             [arr[0, idx[-1]], arr[1, idx[-1]], arr[2, idx[-1]]]])

        field_lines.append(dict(
            lon=lon,
            x_dp=np.where(ok_d, x_d, np.nan), y_dp=np.where(ok_d, y_d, np.nan),
            z_dp=np.where(ok_d, z_d, np.nan),
            xk_p=np.where(ok_k, xk, np.nan), yk_p=np.where(ok_k, yk, np.nan),
            zk_p=np.where(ok_k, zk, np.nan),
            fp_d=_fp(np.array([x_d, y_d, z_d]), ok_d),
            fp_k=_fp(np.array([xk, yk, zk]),    ok_k),
            x_eq=x_eq, y_eq=y_eq,
            xyz_d=xyz_d, xyz_k=xyz_k,
        ))

    # ---- Biot-Savart: sum over all field lines  ------------------------------
    B_dip = B_kt17_tot = None
    quiver_scale = None
    if obs_point is not None and I0 is not None:
        obs    = np.asarray(obs_point, dtype=float)
        n      = len(lons)
        I_each = I0 / n          # current per field line so total = I0
        B_dip      = sum(biot_savart_fac(fl['xyz_d'], obs, fl['lon'], I_each)
                         for fl in field_lines)
        B_kt17_tot = sum(biot_savart_fac(fl['xyz_k'], obs, fl['lon'], I_each)
                         for fl in field_lines)
        # azimuthal unit vector at the observation point: φ̂ = (−sin φ, cos φ, 0)
        phi_obs = np.arctan2(obs[1], obs[0])
        phi_hat = np.array([-np.sin(phi_obs), np.cos(phi_obs), 0.0])

        print(f'\nBiot-Savart dB at obs={obs}  (I0={I0:.2e} A total, {I_each:.2e} A per line × {n})')
        for lbl, B in [('Dipole', B_dip), ('KT17  ', B_kt17_tot)]:
            Bphi = float(np.dot(B, phi_hat))
            print(f'  {lbl} : Bx={B[0]:+.3f}  By={B[1]:+.3f}  Bz={B[2]:+.3f}'
                  f'  |B|={np.linalg.norm(B):.3f}  B_phi={Bphi:+.3f} nT')
        quiver_scale = 0.3 / max(np.linalg.norm(B_dip), np.linalg.norm(B_kt17_tot), 1e-30)

    # ---- plot ----------------------------------------------------------------
    lw    = 1.0 if multi else 1.8
    alpha = 0.5 if multi else 1.0

    def _arrows(ax, h_arr, z_arr, clr, lon):
        valid = np.where(~(np.isnan(h_arr) | np.isnan(z_arr)))[0]
        if len(valid) < 4:
            return
        hv, zv = h_arr[valid], z_arr[valid]
        eq_i   = int(np.argmin(np.abs(zv)))
        if lon < 180:
            legs = [(hv[:eq_i+1],       zv[:eq_i+1],       True),
                    (hv[eq_i:][::-1],   zv[eq_i:][::-1],   True)]
        else:
            legs = [(hv[:eq_i+1][::-1], zv[:eq_i+1][::-1], True),
                    (hv[eq_i:],          zv[eq_i:],          True)]
        step = max(1, len(hv) // 20)
        for lh, lz, _ in legs:
            if len(lh) < 3:
                continue
            for i in np.linspace(step, len(lh) - 1 - step, 2, dtype=int):
                ax.annotate('', xy=(lh[i], lz[i]), xytext=(lh[i-step], lz[i-step]),
                            arrowprops=dict(arrowstyle='->', color=clr,
                                           lw=1.2, mutation_scale=10, alpha=alpha))

    fig, (ax_yz, ax_xz) = plt.subplots(1, 2, figsize=(10, 5))

    for fi, fl in enumerate(field_lines):
        lon  = fl['lon']
        first = fi == 0
        for ax, hlbl in [(ax_yz, 'Y'), (ax_xz, 'X')]:
            hi = 1 if hlbl == 'Y' else 0
            hd  = fl['y_dp'] if hlbl == 'Y' else fl['x_dp']
            hk  = fl['yk_p'] if hlbl == 'Y' else fl['xk_p']
            h_eq = fl['y_eq'] if hlbl == 'Y' else fl['x_eq']
            fp_d, fp_k = fl['fp_d'], fl['fp_k']

            ax.plot(hd, fl['z_dp'], color='royalblue', lw=lw, alpha=alpha,
                    label='Dipole' if first else None)
            ax.plot(hk, fl['zk_p'], color='darkorange', lw=lw, alpha=alpha,
                    ls='--', label='KT17' if first else None)
            _arrows(ax, hd, fl['z_dp'], 'royalblue', lon)
            _arrows(ax, hk, fl['zk_p'], 'darkorange', lon)
            ax.scatter([h_eq], [0], color='k', s=15 if multi else 30, zorder=5)
            if len(fp_d):
                ax.scatter(fp_d[:, hi], fp_d[:, 2], color='royalblue',
                           s=25 if multi else 50, zorder=6, marker='o',
                           edgecolors='k', lw=0.4, alpha=alpha)
            if len(fp_k):
                ax.scatter(fp_k[:, hi], fp_k[:, 2], color='darkorange',
                           s=25 if multi else 50, zorder=6, marker='o',
                           edgecolors='k', lw=0.4, alpha=alpha)

    # obs point and total quiver (drawn once, outside field-line loop)
    for ax, hlbl in [(ax_yz, 'Y'), (ax_xz, 'X')]:
        hi = 1 if hlbl == 'Y' else 0
        if obs_point is not None:
            obs = np.asarray(obs_point, dtype=float)
            ax.scatter([obs[hi]], [obs[2]], color='red', s=60, zorder=7,
                       marker='*', edgecolors='k', lw=0.4, label='Obs')
            if quiver_scale is not None:
                ax.quiver(obs[hi], obs[2],
                          B_dip[hi] * quiver_scale, B_dip[2] * quiver_scale,
                          color='royalblue', angles='xy', scale_units='xy', scale=1,
                          width=0.008, alpha=1.0, edgecolor='k', linewidth=0.5, zorder=8)
                ax.quiver(obs[hi], obs[2],
                          B_kt17_tot[hi] * quiver_scale, B_kt17_tot[2] * quiver_scale,
                          color='darkorange', angles='xy', scale_units='xy', scale=1,
                          width=0.008, alpha=1.0, edgecolor='k', linewidth=0.5, zorder=8)
        ax.add_patch(plt.Circle((0, -0.2), 1.0, color='grey', alpha=0.2, zorder=0))
        ax.add_patch(plt.Circle((0, -0.2), 0.8, color='grey', alpha=1.0, zorder=0))
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.axvline(0, color='k', lw=0.5, alpha=0.4)
        ax.set_xlabel(f'{hlbl} (R$_M$, MSM)')
        ax.set_ylabel('Z (R$_M$, MSM)')
        ax.set_title(f'{hlbl}Z plane')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    ax_yz.invert_xaxis()
    ax_xz.invert_xaxis()

    if lon_range is not None:
        title_lon = f'lon = {lon_range[0]}°–{lon_range[1]}°  ({n_lines} lines)'
    else:
        title_lon = f'lon = {lon_deg}°'
    fig.suptitle(f'Dipole vs KT17  —  {title_lon},  L = {L} R$_M$  (DI = {dist_index})',
                 fontsize=10)
    fig.tight_layout()
    plt.show()

    return fig, (ax_yz, ax_xz), all_dipole, all_kt17

def biot_savart_fac(xyz_wire, obs_point, lon_deg, I0=1.0):
    """Magnetic field perturbation at obs_point from a steady field-aligned current.

    The current is symmetric about the magnetic equator:
        lon_deg <  180° → converging  (both legs flow toward the equatorial crossing)
        lon_deg >= 180° → diverging   (both legs flow away from the equatorial crossing)

    The wire is split at its equatorial crossing (minimum |z|) and each leg
    contributes with the appropriate current direction via the Biot-Savart law:
        dB = (μ₀/4π) I₀ (dl × r̂) / r²

    Parameters
    ----------
    xyz_wire  : ndarray (N, 3)   field-line coordinates in R_M (MSM),
                                  ordered south footpoint → equator → north footpoint
    obs_point : array-like (3,)  observation point in R_M (MSM)
    lon_deg   : float            equatorial longitude of the field line (degrees)
    I0        : float            current magnitude in Amperes

    Returns
    -------
    B : ndarray (3,)  magnetic field perturbation [Bx, By, Bz] in nT (MSM)
    """
    R_M_m        = 2.439e6   # metres per Mercury radius
    mu0_over_4pi = 1e-7      # T·m / A
    obs          = np.asarray(obs_point, dtype=float)

    eq_i  = int(np.argmin(np.abs(xyz_wire[:, 2])))
    south = xyz_wire[:eq_i + 1]   # south footpoint → equator
    north = xyz_wire[eq_i:]       # equator → north footpoint

    if lon_deg < 180:
        # converging: south flows toward equator (forward), north flows toward equator (reversed)
        legs = [south, north[::-1]]
    else:
        # diverging: south flows away from equator (reversed), north flows away (forward)
        legs = [south[::-1], north]

    def _bs_leg(wire):
        if len(wire) < 2:
            return np.zeros(3)
        dl    = np.diff(wire, axis=0) * R_M_m
        mid   = 0.5 * (wire[:-1] + wire[1:])
        r_vec = (obs - mid) * R_M_m
        r_mag = np.linalg.norm(r_vec, axis=1)
        valid = r_mag > 1e-10
        r_hat = np.zeros_like(r_vec)
        r_hat[valid] = r_vec[valid] / r_mag[valid, np.newaxis]
        dB        = np.cross(dl, r_hat)
        dB[valid] /= r_mag[valid, np.newaxis] ** 2
        return dB.sum(axis=0)

    B = _bs_leg(legs[0]) + _bs_leg(legs[1])
    return mu0_over_4pi * I0 * B * 1e9  # nT

def validate_biot_savart(n=5000, L=500.0, I0=1e6):
    """Validate biot_savart_fac against analytic field of a finite straight wire.

    Wire runs along z-axis from -L to +L (R_M units), current in +z direction.
    Observation points at (rho, 0, 0).  Analytic solution:
        B_y = (mu0/2pi) * I * L / (rho * sqrt(L^2 + rho^2))   [nT]
    In the limit L >> rho this approaches the infinite-wire result mu0*I/(2*pi*rho).
    """
    R_M_m        = 2.439e6
    mu0_over_4pi = 1e-7

    wire = np.column_stack([
        np.zeros(n), np.zeros(n), np.linspace(-L, L, n)
    ])
    dl  = np.diff(wire, axis=0) * R_M_m      # (n-1, 3)  in metres
    mid = 0.5 * (wire[:-1] + wire[1:])       # (n-1, 3)  in R_M

    print(f'\nBiot-Savart validation — straight wire along z, ±{L:.0f} R_M, I0={I0:.2e} A')
    print(f'{"rho (R_M)":>10}  {"B_num (nT)":>12}  {"B_finite (nT)":>14}'
          f'  {"B_inf (nT)":>12}  {"rel err":>9}  {"dir ok":>7}')

    for rho in [0.5, 1.0, 2.0, 5.0, 10.0]:
        obs   = np.array([rho, 0.0, 0.0])
        r_vec = (obs - mid) * R_M_m           # (n-1, 3)  in metres
        r_mag = np.linalg.norm(r_vec, axis=1)
        valid = r_mag > 1e-10
        r_hat = np.zeros_like(r_vec)
        r_hat[valid] = r_vec[valid] / r_mag[valid, np.newaxis]

        dB = np.cross(dl, r_hat)
        dB[valid] /= r_mag[valid, np.newaxis] ** 2
        B_num = mu0_over_4pi * I0 * dB.sum(axis=0) * 1e9   # nT

        rho_m = rho * R_M_m
        L_m   = L   * R_M_m
        B_fin = 2e-7 * I0 * L_m / (rho_m * np.sqrt(L_m**2 + rho_m**2)) * 1e9
        B_inf = 2e-7 * I0 / rho_m * 1e9

        B_mag = np.linalg.norm(B_num)
        err   = abs(B_mag - B_fin) / B_fin
        # for wire along +z and obs at +x, By should be positive, Bx≈Bz≈0
        dir_ok = B_num[1] > 0 and abs(B_num[0]) < 1e-6 * B_mag and abs(B_num[2]) < 1e-6 * B_mag

        print(f'{rho:>10.1f}  {B_mag:>12.4f}  {B_fin:>14.4f}'
              f'  {B_inf:>12.4f}  {err:>9.2e}  {"yes" if dir_ok else "NO":>7}')

    print()

#validate_biot_savart()

#plot_dipole_field_line(lon_deg=110, L=1.3)

#plot_dipole_field_line(lon_deg=170, L=1.6, obs_point=[-1.3, 0.2,0.25], I0=11e3)   

#plot_dipole_field_line(lon_range=(130,175), n_lines=20, L=1.8, obs_point=[-1.41,0.4,0.54], I0=11e3, dist_index=10, dip_fact = 1.0)   

#plot_dipole_field_line(lon_range=(100,260), n_lines=15, L=1.5, obs_point=[-0.75, 0.5,0.6], I0=11e3, dist_index=10)   

'''

events = load_human_loading_labels()         
events = events[events['orbit'] == 3289] 
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

    # Compute basis vectors (constant because background field is fixed)
    B0  = np.array([Bx0, By0, Bz0])
    R0  = np.array([df_unloading['ephx'].iloc[0],
                    df_unloading['ephy'].iloc[0],
                    df_unloading['ephz'].iloc[0]])
    b_hat    = B0 / np.linalg.norm(B0)
    phi_vec  = np.cross(b_hat, R0)
    phi_hat  = phi_vec / np.linalg.norm(phi_vec)
    perp_hat = np.cross(phi_hat, b_hat)

    def _fmt(v): return f'({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})'

    t_unload = pd.to_datetime(df_unloading['time'])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_unload, B_par_unload,  color='red',  lw=0.9, label='$B_{\\parallel}$')
    ax.plot(t_unload, B_phi_unload,  color='green', lw=0.9, label='$B_{\\phi}$')
    ax.plot(t_unload, B_perp_unload, color='blue',   lw=0.9, label='$B_{\\perp}$')
    ax.axhline(B_par_unload[0],  color='red',  lw=0.6, ls='--', alpha=0.6)
    ax.axhline(B_phi_unload[0],  color='green', lw=0.6, ls='--', alpha=0.6)
    ax.axhline(B_perp_unload[0], color='blue',   lw=0.6, ls='--', alpha=0.6)
    ax.set_xlabel('UTC')
    ax.set_ylabel('B (nT)')
    ax.set_title(
        f'Orbit {int(orbit)} — FAC unloading phase\n'
        r'$\hat{b}$=' + _fmt(b_hat) +
        r'   $\hat{\phi}$=' + _fmt(phi_hat) +
        r'   $\hat{\perp}$=' + _fmt(perp_hat),
        fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Ephemeris tick labels
    from matplotlib.dates import num2date
    fig.canvas.draw()
    xlim = ax.get_xlim()
    tick_locs = [tk for tk in ax.get_xticks() if xlim[0] <= tk <= xlim[1]]
    if tick_locs:
        t_ns = t_unload.astype('int64').to_numpy()
        ex   = df_unloading['ephx'].to_numpy(dtype=float)
        ey   = df_unloading['ephy'].to_numpy(dtype=float)
        ez   = df_unloading['ephz'].to_numpy(dtype=float)
        labels = []
        for tk in tick_locs:
            tk_ts = pd.Timestamp(num2date(tk).replace(tzinfo=None))
            idx   = int(np.argmin(np.abs(t_ns - np.int64(tk_ts.value))))
            labels.append(f"{tk_ts.strftime('%H:%M')}\n"
                          f"X={ex[idx]:.2f}\n"
                          f"Y={ey[idx]:.2f}\n"
                          f"Z={ez[idx]:.2f}")
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(labels, fontsize=7, ha='center')
    ax.set_xlabel(r'UTC  /  $X\ Y\ Z\ (R_M)$', fontsize=8)

    plt.tight_layout()
    plt.show()

    break

'''