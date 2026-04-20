#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:20:25 2026

@author: alexandercushen
"""

#import pyspedas
#import pytplot
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import pickle
import pandas as pd
import spiceypy as spice
import urllib.request
import re
import json
import KT17

KERNEL_DIR  = os.path.expanduser("~/mercury_dipolarizations/messenger_kernels")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DI_CSV     = os.path.join(_SCRIPT_DIR, 'orb_num_start_ut_rhel_di.csv')

# ---------------------------------------------------------------------------
# DistIndex lookup table (loaded once, used by get_kt17_along_track)
# ---------------------------------------------------------------------------
_DI_TABLE: 'pd.DataFrame | None' = None

def _get_di_table() -> 'pd.DataFrame':
    """Load and cache the orbit DistIndex table. Called lazily on first use."""
    global _DI_TABLE
    if _DI_TABLE is not None:
        return _DI_TABLE
    df = pd.read_csv(_DI_CSV, parse_dates=['start_ut'])
    df = df.dropna(subset=['di']).sort_values('start_ut').reset_index(drop=True)
    _DI_TABLE = df
    return _DI_TABLE

def _lookup_dist_index(t: 'pd.Timestamp', default: float = 50.0) -> float:
    """
    Return the DistIndex for the orbit whose start_ut is the latest time <= t.
    Falls back to `default` if t is before all known orbit start times or the
    table contains no valid di values.
    """
    tbl = _get_di_table()
    before = tbl[tbl['start_ut'] <= t]
    if before.empty:
        return default
    return float(before.iloc[-1]['di'])

# spiceypy methods
def load_messenger_kernels():
    """Load all kernels found in KERNEL_DIR."""
    kernel_files = [f for f in os.listdir(KERNEL_DIR) 
                    if f.endswith(('.bsp', '.tls', '.tpc', '.tf', '.tsc'))]
    if not kernel_files:
        raise FileNotFoundError(f"No kernel files found in {KERNEL_DIR}. "
                                f"Please download them manually from:\n"
                                f"https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/")
    for f in sorted(kernel_files):
        path = os.path.join(KERNEL_DIR, f)
        spice.furnsh(path)
        print(f"  Loaded: {f}")

def download_messenger_trajectory(trange, coord='mso', dt_sec=60.0):
    load_messenger_kernels()

    frame_map = {'mso': 'MSGR_MSO', 'msm': 'MSGR_MSM'}
    frame = frame_map.get(coord.lower(), 'MSGR_MSO')

    fmt = "%Y-%m-%d/%H:%M:%S"
    t_start = datetime.strptime(trange[0], fmt)
    t_end   = datetime.strptime(trange[1], fmt)

    n_steps = int((t_end - t_start).total_seconds() / dt_sec) + 1
    times_dt = [t_start + timedelta(seconds=i * dt_sec) for i in range(n_steps)]
    times_et = [spice.str2et(t.strftime("%Y-%m-%dT%H:%M:%S")) for t in times_dt]

    xyz = []
    n_gaps = 0
    for et in times_et:
        try:
            state, _ = spice.spkpos('MESSENGER', et, frame, 'NONE', 'MERCURY')
            xyz.append(state)
        except spice.utils.exceptions.SpiceSPKINSUFFDATA:
            xyz.append([np.nan, np.nan, np.nan])
            n_gaps += 1
    xyz = np.array(xyz)

    times_dt = np.array(times_dt)
    R_M = 2440.0
    r = np.sqrt(np.nansum(xyz**2, axis=1))

    print(f"[MESSENGER] loaded {len(times_dt)} time steps  |  "
          f"t: {times_dt[0]} → {times_dt[-1]}  |  "
          f"coord: {coord}  |  "
          f"r_range: {np.nanmin(r)/R_M:.2f}–{np.nanmax(r)/R_M:.2f} R_M"
          + (f"  |  gaps: {n_gaps} steps" if n_gaps else ""))

    spice.kclear()

    return {'messenger': {
        'time':  times_dt,
        'x':     xyz[:, 0],
        'y':     xyz[:, 1],
        'z':     xyz[:, 2],
        'coord': coord,
    }}

def plot_messenger_trajectory(traj, trange=None, color_by_r=False, showFig = True):
    """
    Plot MESSENGER trajectory in XZ and YZ planes.

    Parameters
    ----------
    traj       : dict returned by download_messenger_trajectory
    trange     : optional tuple/list of two datetime objects or ISO strings
                 to restrict the plotted interval
    color_by_r : if True, colour the track by radial distance (R_M)
    """
    data = traj['messenger']
    t    = data['time']
    x    = data['x']
    y    = data['y']
    z    = data['z']
    coord = data['coord'].upper()
    R_M  = 2440.0  # km

    # --- optional time mask ---
    if trange is not None:
        def _to_dt(v):
            return datetime.strptime(v, "%Y-%m-%d/%H:%M:%S") if isinstance(v, str) else v
        t0, t1 = _to_dt(trange[0]), _to_dt(trange[1])
        mask = (t >= t0) & (t <= t1)
        t, x, y, z = t[mask], x[mask], y[mask], z[mask]

    # insert NaN breaks at time gaps > 2× the median cadence
    t_ns    = t.astype('datetime64[ns]').astype(np.int64)
    dt      = np.diff(t_ns)
    gap_thr = 2 * np.median(dt)
    gaps    = np.where(dt > gap_thr)[0] + 1          # indices where new segment starts
    if len(gaps):
        for offset, g in enumerate(gaps):
            ins = g + offset                          # shift as we insert
            nan_row = np.array([np.nan])
            x = np.insert(x, ins, np.nan)
            y = np.insert(y, ins, np.nan)
            z = np.insert(z, ins, np.nan)

    x_rm = x / R_M
    y_rm = y / R_M
    z_rm = z / R_M
    r_rm = np.sqrt(x_rm**2 + y_rm**2 + z_rm**2)

    norm = plt.Normalize(r_rm.min(), r_rm.max())
    cmap = plt.cm.plasma

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"MESSENGER trajectory  [{pd.Timestamp(t[0]).strftime('%Y-%m-%d %H:%M')} – "
        f"{pd.Timestamp(t[-1]).strftime('%Y-%m-%d %H:%M')} UT]  |  {coord}",
        fontsize=11
    )

    # behind_mask: spacecraft is occulted by Mercury in the projected plane
    # XZ plane (viewed from +Y): behind when y<0 AND sqrt(x²+z²)<=1
    # YZ plane (viewed from +X): behind when x<0 AND sqrt(y²+z²)<=1
    behind_xz = (y_rm > 0) & (np.sqrt(x_rm**2 + z_rm**2) <= 1.0)
    behind_yz = (x_rm > 0) & (np.sqrt(y_rm**2 + z_rm**2) <= 1.0)

    panels = [
        (axes[0], x_rm, z_rm, behind_xz, f"X$_{{{coord}}}$ (R$_M$)", f"Z$_{{{coord}}}$ (R$_M$)", "XZ plane"),
        (axes[1], y_rm, z_rm, behind_yz, f"Y$_{{{coord}}}$ (R$_M$)", f"Z$_{{{coord}}}$ (R$_M$)", "YZ plane"),
    ]

    def _plot_track(ax, hval, vval, behind, color, lw, cmap, norm, r_rm, color_by_r):
        """Draw track solid normally, dotted where behind Mercury's disk."""
        front = ~behind
        changes = np.where(np.diff(front.astype(int)))[0] + 1
        bounds  = np.concatenate(([0], changes, [len(front)]))
        sc = None
        for i in range(len(bounds) - 1):
            sl = slice(bounds[i], min(bounds[i + 1] + 1, len(hval)))  # +1 for join
            is_front = front[bounds[i]]
            seg_lw    = lw        if is_front else lw * 0.5
            seg_alpha = 1.0       if is_front else 0.8
            zord      = 4         if is_front else 2
            if color_by_r:
                sc = ax.scatter(hval[sl], vval[sl], c=r_rm[sl], cmap=cmap,
                                norm=norm, s=4 if is_front else 2,
                                zorder=zord, alpha=seg_alpha)
            else:
                ax.plot(hval[sl], vval[sl], ls='-', color=color,
                        lw=seg_lw, alpha=seg_alpha, zorder=zord)
        return sc

    sc = None
    for ax, hval, vval, behind, xlabel, ylabel, title in panels:
        sc_panel = _plot_track(ax, hval, vval, behind,
                               color='steelblue', lw=1.0,
                               cmap=cmap, norm=norm, r_rm=r_rm,
                               color_by_r=color_by_r)
        if sc_panel is not None:
            sc = sc_panel

        # mark start / end
        ax.scatter(hval[0],  vval[0],  marker='o', s=60, color='lime',   zorder=5, label='Start')
        ax.scatter(hval[-1], vval[-1], marker='s', s=60, color='tomato', zorder=5, label='End')

        # Mercury body
        mercury = plt.Circle((0, 0), 1.0, color='saddlebrown', alpha=0.35, zorder=1)
        ax.add_patch(mercury)
        ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.3)
        ax.axvline(0, color='k', lw=0.4, ls='--', alpha=0.3)

        # square axes: find the wider data span and apply it symmetrically to both axes
        h_lim = max(abs(np.nanmax(hval)), abs(np.nanmin(hval)))
        v_lim = max(abs(np.nanmax(vval)), abs(np.nanmin(vval)))
        lim = max(h_lim, v_lim) * 1.05  # 5% padding
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)

    axes[1].invert_xaxis()  # YZ panel: +Y on left, -Y on right

    if color_by_r and sc is not None:
        cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.04)
        cbar.set_label("r (R$_M$)")

    plt.tight_layout()
    if showFig:
        plt.show()
    return fig

# kt17 model tools
def plot_kt17_streamplot(xlim, zlim, y0=0.0, nx=40, nz=40, **kt17_kwargs):
    """
    Streamplot of the KT17 magnetic field in the XZ plane at fixed y0.

    Parameters
    ----------
    xlim       : (xmin, xmax) in R_M MSM
    zlim       : (zmin, zmax) in R_M MSM
    y0         : y-slice position (R_M MSM), default 0
    nx, nz     : grid resolution
    **kt17_kwargs : passed directly to KT17.TraceField (e.g. Rsm, DistIndex)
    """
    x1d = np.linspace(xlim[0], xlim[1], nx)
    z1d = np.linspace(zlim[0], zlim[1], nz)
    xx, zz = np.meshgrid(x1d, z1d)
    yy = np.full_like(xx, y0)

    T = KT17.ModelField(xx.ravel(), yy.ravel(), zz.ravel(), **kt17_kwargs)

    Bx = T[0].reshape(nz, nx)
    By = T[1].reshape(nz, nx)
    Bz = T[2].reshape(nz, nx)

    r = np.sqrt(xx**2 + yy**2 + zz**2)
    Bx = np.where(r < 1.0, np.nan, Bx)
    Bz = np.where(r < 1.0, np.nan, Bz)

    fig, ax = plt.subplots(figsize=(7, 7))
    sp = ax.streamplot(
        x1d, z1d, Bx, Bz, broken_streamlines=False,
        color=np.log10(np.sqrt(Bx**2 + Bz**2)),
        cmap='plasma', linewidth=1.0, density=1.5, arrowsize=1.2,
    )
    cbar = fig.colorbar(sp.lines, ax=ax)
    cbar.set_label('log$_{10}$|B| (nT)')

    ax.add_patch(plt.Circle((0, 0), 1.0, color='saddlebrown', alpha=0.6, zorder=5))
    ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.3)
    ax.axvline(0, color='k', lw=0.4, ls='--', alpha=0.3)
    ax.set_xlim(xlim[1],xlim[0])
    ax.set_ylim(zlim)
    ax.set_xlabel('X$_{MSM}$ (R$_M$)')
    ax.set_ylabel('Z$_{MSM}$ (R$_M$)')
    ax.set_title(f'KT17 field — XZ plane at Y = {y0:.2f} R$_M$')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
    return fig

def bowers_traj(trange):
    """
    Load Bowers ephemeris for a trange and return a trajectory dict compatible
    with plot_messenger_trajectory.

    Bowers ephx/y/z are in R_M (MSM); the dict stores values in km so the
    plotter's internal /R_M conversion gives back the correct R_M values.
    """
    R_M = 2440.0
    df = load_bowers_data_pkl(trange=trange)
    return {'messenger': {
        'time':  pd.to_datetime(df['time']).to_numpy(),
        'x':     df['ephx'].to_numpy() * R_M,
        'y':     df['ephy'].to_numpy() * R_M,
        'z':     df['ephz'].to_numpy() * R_M,
        'coord': 'msm',
    }}

def get_kt17_along_track(trange=None, df=None, **kt17_kwargs):
    """
    Evaluate the KT17 model field at MESSENGER's observed positions.

    Parameters
    ----------
    trange      : optional [start, end] as 'YYYY-MM-DD/HH:MM:SS' strings.
                  If df is provided, trange is derived from it automatically.
    df          : optional pre-loaded Bowers DataFrame (skips reload if provided)
    **kt17_kwargs : forwarded to KT17.ModelField (e.g. Rsm, DistIndex)

    Returns
    -------
    time : np.ndarray of datetime64
    Bx, By, Bz : np.ndarray (nT), model field in MSM at each position
    """
    from astropy.time import Time
    import astropy.units as u

    if df is None:
        if trange is None:
            raise ValueError("Either trange or df must be provided.")
        df = load_bowers_data_pkl(trange=trange)

    x = df['ephx'].to_numpy()
    y = df['ephy'].to_numpy()
    z = df['ephz'].to_numpy()

    from astropy.coordinates import get_body, solar_system_ephemeris

    fmt = "%Y-%m-%d/%H:%M:%S"
    t_obs = pd.to_datetime(df['time'])
    t0 = Time(t_obs.iloc[0].to_pydatetime())
    with solar_system_ephemeris.set('builtin'):
        mercury = get_body('mercury', t0)
        sun     = get_body('sun',     t0)
    rsun = mercury.separation_3d(sun).to(u.AU).value
    #print(f"  Rsun = {rsun:.4f} AU")

    # Use DistIndex from the lookup table unless the caller overrides it explicitly.
    if 'DistIndex' not in kt17_kwargs:
        kt17_kwargs = dict(kt17_kwargs)   # don't mutate the caller's dict
        DistIndex_val = _lookup_dist_index(t_obs.iloc[0])
        kt17_kwargs['DistIndex'] = DistIndex_val
        #print(f"  DistIndex = {DistIndex_val:.1f}")

    T = KT17.ModelField(x, y, z, Rsun=rsun, **kt17_kwargs)
    Bx = T[0]
    By = T[1]
    Bz = T[2]

    return df['time'].to_numpy(), Bx, By, Bz

def plot_mag_timeseries(trange, show_model=True, save_path=None, fontsize=15, **kt17_kwargs):
    """
    Plot observed Bx, By, Bz from the Bowers dataset over a time range,
    with an optional KT17 model overplot.

    Parameters
    ----------
    trange      : [start, end] as 'YYYY-MM-DD/HH:MM:SS' strings
    show_model  : if True, overplot KT17 model field
    **kt17_kwargs : forwarded to KT17.ModelField when show_model=True
    """
    df = load_bowers_data_pkl(trange=trange)
    t_obs = pd.to_datetime(df['time'])

    if show_model:
        t_mod, Bx_mod, By_mod, Bz_mod = get_kt17_along_track(trange, df=df, **kt17_kwargs)
        t_mod = pd.to_datetime(t_mod)

    components = [
        ('magx', 'B$_x$', Bx_mod if show_model else None),
        ('magy', 'B$_y$', By_mod if show_model else None),
        ('magz', 'B$_z$', Bz_mod if show_model else None),
    ]

    fmt = "%Y-%m-%d/%H:%M:%S"
    t0 = pd.Timestamp(datetime.strptime(trange[0], fmt))
    t1 = pd.Timestamp(datetime.strptime(trange[1], fmt))

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"MESSENGER MAG  [{trange[0]} → {trange[1]}]", fontsize=11)
    fig.subplots_adjust(hspace=0)

    for ax, (col, label, mod) in zip(axes, components):
        ax.plot(t_obs, df[col], color='steelblue', lw=0.7, label='Observed')
        if show_model and mod is not None:
            ax.plot(t_mod, mod, color='tomato', lw=1.0, ls='--', label='KT17')
        ax.set_ylabel(f'{label} (nT)', fontsize=fontsize)
        ax.set_xlim(t0, t1)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.4)
        ax.grid(True, alpha=0.2)
        if show_model:
            ax.legend(fontsize=8, loc='upper right')

    # --- ephemeris tick labels ---
    fig.canvas.draw()
    tick_locs = axes[-1].get_xticks()
    tick_times = [mdates.num2date(t).replace(tzinfo=None) for t in tick_locs]

    t_arr  = t_obs.values.astype('datetime64[ns]')
    x_arr  = df['ephx'].to_numpy()
    y_arr  = df['ephy'].to_numpy()
    z_arr  = df['ephz'].to_numpy()

    labels = []
    for tt in tick_times:
        idx = np.searchsorted(t_arr, np.datetime64(tt, 'ns'))
        idx = int(np.clip(idx, 0, len(t_arr) - 1))
        labels.append(
            f"{tt.strftime('%H:%M:%S')}\n{x_arr[idx]:.3f}\n{y_arr[idx]:.3f}\n{z_arr[idx]:.3f}"
        )
    axes[-1].set_xticks(tick_locs)
    axes[-1].set_xticklabels(labels, fontsize=fontsize)

    # row labels on the right
    row_labels = ['UT', 'X (R$_M$)', 'Y (R$_M$)', 'Z (R$_M$)']
    for i, rl in enumerate(row_labels):
        axes[-1].annotate(
            rl, xy=(1.01, -0.06 * i), xycoords=('axes fraction', 'axes fraction'),
            fontsize=fontsize * 0.6, va='top', ha='left', annotation_clip=False
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return fig

def transform_to_fac(bx_meas, by_meas, bz_meas, bx_mod, by_mod, bz_mod, rx, ry, rz):
    """
    Transforms magnetic field components into a Field-Aligned Coordinate (FAC) system.
    Accepts arrays of N time steps.
    Returns: B_perp, B_phi, B_par  (each shape N)
    """
    # Stack into (N, 3) arrays
    B_meas = np.column_stack([bx_meas, by_meas, bz_meas])
    B_mod  = np.column_stack([bx_mod,  by_mod,  bz_mod])
    R      = np.column_stack([rx,      ry,      rz])

    # 1. Parallel unit vector: along model field
    b_norm = np.linalg.norm(B_mod, axis=1, keepdims=True)
    b_hat  = B_mod / b_norm

    # 2. Azimuthal unit vector: perp to both field and position vector
    phi_vec = np.cross(b_hat, R)
    phi_hat = phi_vec / np.linalg.norm(phi_vec, axis=1, keepdims=True)

    # 3. Radial-meridional unit vector: completes right-handed basis
    perp_hat = np.cross(phi_hat, b_hat)

    # 4. Project measured field onto each basis vector
    b_par  = np.sum(B_meas * b_hat,    axis=1)
    b_phi  = np.sum(B_meas * phi_hat,  axis=1)
    b_perp = np.sum(B_meas * perp_hat, axis=1)

    return b_perp, b_phi, b_par

def set_ephemeris_ticklabels(ax, df, fontsize=15, coords='xyz'):
    """Replace the x-axis tick labels on *ax* with multi-row ephemeris labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df : pandas.DataFrame
        Must contain columns 'time', 'ephx', 'ephy', 'ephz'.
    fontsize : float
    coords : {'xyz', 'latlon'}
        'xyz'    — show UT / X / Y / Z  (R_M MSM, default)
        'latlon' — show UT / E.Lon / Lat / Alt where Alt = r - 1 R_M

    Returns
    -------
    ax : matplotlib.axes.Axes  (same object, modified in place)
    """
    t_obs = pd.to_datetime(df['time'])

    tick_locs  = ax.get_xticks()
    tick_times = [mdates.num2date(t).replace(tzinfo=None) for t in tick_locs]

    t_arr = t_obs.values.astype('datetime64[ns]')
    x_arr = df['ephx'].to_numpy()
    y_arr = df['ephy'].to_numpy()
    z_arr = df['ephz'].to_numpy()

    if coords == 'latlon':
        r_arr   = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
        lat_arr = np.degrees(np.arcsin(np.clip(z_arr / r_arr, -1, 1)))
        lon_arr = np.degrees(np.arctan2(y_arr, x_arr)) % 360
        alt_arr = r_arr - 1.0          # altitude above surface in R_M

        labels = []
        for tt in tick_times:
            idx = int(np.clip(np.searchsorted(t_arr, np.datetime64(tt, 'ns')), 0, len(t_arr) - 1))
            labels.append(
                f"{tt.strftime('%H:%M:%S')}\n{lon_arr[idx]:.1f}°\n"
                f"{lat_arr[idx]:+.1f}°\n{alt_arr[idx]:.3f}"
            )
        row_labels = ['UT', 'E.Lon', 'Lat', 'Alt (R$_M$)']
    else:
        labels = []
        for tt in tick_times:
            idx = int(np.clip(np.searchsorted(t_arr, np.datetime64(tt, 'ns')), 0, len(t_arr) - 1))
            labels.append(
                f"{tt.strftime('%H:%M:%S')}\n{x_arr[idx]:.3f}\n{y_arr[idx]:.3f}\n{z_arr[idx]:.3f}"
            )
        row_labels = ['UT', 'X (R$_M$)', 'Y (R$_M$)', 'Z (R$_M$)']

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(labels, fontsize=fontsize)

    for i, rl in enumerate(row_labels):
        ax.annotate(
            rl, xy=(1.01, -0.06 * i), xycoords=('axes fraction', 'axes fraction'),
            fontsize=fontsize * 0.9, va='top', ha='left', annotation_clip=False
        )

    return ax

def plot_field_aligned_timeseries(trange=None, save_path=None, fontsize=15,
                                  highlight=False, show_loading_times=False,
                                  df=None, ext_dr_ivs=None, ext_load_ivs=None,
                                  **kt17_kwargs):
    """
    Plot observed magnetic field in terms of field-aligned values, as given by the
    KT model field.

    Parameters
    ----------
    trange        : [start, end] as 'YYYY-MM-DD/HH:MM:SS' strings.
                    Optional if df is provided (derived from df if omitted).
    df            : pre-loaded DataFrame (skips load_bowers_data_pkl call)
    ext_dr_ivs    : list of [t_start, t_end] Timestamp pairs to shade as DR intervals
    ext_load_ivs  : list of [t_start, t_end] Timestamp pairs to shade as loading intervals
    **kt17_kwargs : forwarded to get_kt17_along_track
    """
    if df is None:
        df = load_bowers_data_pkl(trange=trange)
    t_obs = pd.to_datetime(df['time'])
    X      = df['ephx'].to_numpy()
    Y      = df['ephy'].to_numpy()
    Z      = df['ephz'].to_numpy()
    Bx_obs = df['magx'].to_numpy()
    By_obs = df['magy'].to_numpy()
    Bz_obs = df['magz'].to_numpy()
    Bmag_obs = np.sqrt(Bx_obs**2+By_obs**2+Bz_obs**2)

    t_mod, Bx_mod, By_mod, Bz_mod = get_kt17_along_track(trange, df=df, **kt17_kwargs)
    Bmag_mod = np.sqrt(Bx_mod**2+By_mod**2+Bz_mod**2)
    t_mod = pd.to_datetime(t_mod)

    # Compute field-aligned strength
    B_perp, B_phi, B_para = transform_to_fac(Bx_obs, By_obs, Bz_obs, 
                                             Bx_mod, By_mod, Bz_mod, X, Y, Z)
    
    # Normalize to delta
    B_para = B_para - Bmag_mod

    if trange is not None:
        fmt = "%Y-%m-%d/%H:%M:%S"
        t0 = pd.Timestamp(datetime.strptime(trange[0], fmt))
        t1 = pd.Timestamp(datetime.strptime(trange[1], fmt))
    else:
        t0 = t_obs.iloc[0]
        t1 = t_obs.iloc[-1]

    '''
    components = [
        (B_perp, r'$\Delta B_\perp$'),
        (B_phi, r'$\Delta B_\phi$'),
        (B_para, r'$\Delta B_\parallel$'),
    ]
    '''

    fig, axes = plt.subplots(3, 1, figsize=(18, 15), sharex=True,  
                             gridspec_kw={'height_ratios': [1, 3, 3]}) 
    title_str = f"{trange[0]} → {trange[1]}" if trange is not None else f"{t0.strftime('%Y-%m-%d %H:%M:%S')} → {t1.strftime('%H:%M:%S')}"
    fig.suptitle(f"MESSENGER MAG  [{title_str}]", fontsize=fontsize)
    fig.subplots_adjust(hspace=0)

    axes[0].plot(t_obs, Bmag_obs, color='black', lw=0.7)
    axes[0].plot(t_obs, Bmag_mod, color='black', lw=0.7, linestyle = "dashed")
    axes[0].set_ylim(0,500)

    axes[1].plot(t_obs, Bx_obs, color='red', lw=0.7, label=r'$B_x$')
    axes[1].plot(t_obs, Bx_mod, color='red', lw=0.7, linestyle = "dashed")
    axes[1].plot(t_obs, By_obs, color='green', lw=0.7, label=r'$B_y$')
    axes[1].plot(t_obs, By_mod, color='green', lw=0.7, linestyle = "dashed")
    axes[1].plot(t_obs, Bz_obs, color='blue', lw=0.7, label=r'$B_z$')
    axes[1].plot(t_obs, Bz_mod, color='blue', lw=0.7, linestyle = "dashed")
    axes[1].legend(loc='lower right', fontsize=fontsize)
    axes[1].set_ylim(-200,500)

    axes[2].plot(t_obs, B_perp, color='red', lw=0.7, label=r'$\Delta B_\perp$')
    axes[2].plot(t_obs, B_phi, color='green', lw=0.7, label=r'$\Delta B_\perp$')
    axes[2].plot(t_obs, B_para, color='blue', lw=0.7, label=r'$\Delta B_\parallel$')
    axes[2].axhline(y=0,color='black')
    axes[2].legend(loc='lower right', fontsize=fontsize)
    axes[2].set_ylim(-100,100)

    # Shade externally-supplied loading / DR intervals (e.g. from _apply_dr_filter)
    if ext_load_ivs:
        for lt in ext_load_ivs:
            for ax in axes:
                ax.axvspan(lt[0], lt[1], color='limegreen', alpha=0.20, zorder=0)
    if ext_dr_ivs:
        for iv in ext_dr_ivs:
            for ax in axes:
                ax.axvspan(iv[0], iv[1], color='gold', alpha=0.30, zorder=0)
            axes[1].axvline(iv[0], color='darkorange', lw=1.0, ls='--', alpha=0.8)
            axes[1].axvline(iv[1], color='darkorange', lw=1.0, ls='--', alpha=0.8)

    # Work out highlighted times of interest
    if highlight:
        dt_sec = (t_obs.iloc[1] - t_obs.iloc[0]).total_seconds() if len(t_obs) > 1 else 1.0
        min_samples = max(1, int(60.0 / dt_sec))
        #mask = (B_para > 20) & (np.abs(B_phi)<30)

        # Shade intervals where B_para > 20% Bmag_mod for at least 60 s continuously
        mask = (B_para > (Bmag_mod*0.2)) & (np.abs(B_phi)<Bmag_mod*0.2)
        padded = np.concatenate([[False], mask, [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) >= min_samples:
                t_start = t_obs.iloc[s]
                t_end   = t_obs.iloc[min(e, len(t_obs) - 1)]
                for ax in axes:
                    ax.axvspan(t_start, t_end, color='gold', alpha=0.3, zorder=0)

        # Shade intervals where B_perp > 20% Bmag_mod for at least 20 s continuously
        min_samples = max(1, int(20.0 / dt_sec))
        mask = (B_perp > (Bmag_mod*0.2))
        padded = np.concatenate([[False], mask, [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) >= min_samples:
                t_start = t_obs.iloc[s]
                t_end   = t_obs.iloc[min(e, len(t_obs) - 1)]
                for ax in axes:
                    ax.axvspan(t_start, t_end, color='green', alpha=0.3, zorder=0)

    if show_loading_times:

        loading_times = find_substorm_loading(df, **kt17_kwargs)

        if len(loading_times)>0:
            for loading_time in loading_times:
                for ax in axes:
                    ax.axvspan(loading_time[0], loading_time[1], color='green', alpha=0.3, zorder=0)

        DR_times = find_DRs_following_substorm(df, loading_times, **kt17_kwargs)

        if len(DR_times)>0:
            for DR_time in DR_times:
                for ax in axes:
                    ax.axvspan(DR_time[0], DR_time[1], color='gold', alpha=0.3, zorder=0)

    for ax in axes:
        ax.set_ylabel(f'B (nT)', fontsize=fontsize)
        ax.set_xlim(t0, t1)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.4)
        ax.grid(True, alpha=0.2)

    # --- ephemeris tick labels ---
    fig.canvas.draw()
    set_ephemeris_ticklabels(axes[-1], df, fontsize=fontsize)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return fig

# Bowers file method
def load_bowers_data_pkl(trange=None, orbit_number=None, filename=None):
    """
    Load MESSENGER_Full_Data_Ab_MSM filtered to an optional time range or orbit number.

    Columns: time, ephx, ephy, ephz (R_M MSM), magx, magy, magz, magamp (nT),
             Transition, Type_num, orbit_number.

    On first call the pickle is converted to Parquet for fast future loads.

    Parameters
    ----------
    trange       : optional [start, end] as 'YYYY-MM-DD/HH:MM:SS' strings or datetimes
    orbit_number : optional int or list of ints — load all rows for that orbit(s)
    """
    import pyarrow.parquet as pq

    if filename is None:
        filename = os.path.expanduser('~/mercury_dipolarizations/MESSENGER_Full_Data_Ab_MSM.pkl')

    parquet_path = os.path.splitext(filename)[0] + '.parquet'

    if not os.path.exists(parquet_path):
        print("First run: converting pickle → Parquet (one-time, may be slow)...")
        with open(filename, 'rb') as f:
            df_full = pickle.load(f)
        df_full['time'] = pd.to_datetime(df_full['time'])
        df_full = df_full.sort_values('time').reset_index(drop=True)
        df_full.to_parquet(parquet_path, index=False, row_group_size=50_000)
        print(f"Saved: {parquet_path}  ({os.path.getsize(parquet_path)/1e6:.1f} MB)")
        del df_full

    fmt = "%Y-%m-%d/%H:%M:%S"
    def _to_ts(v):
        return pd.Timestamp(datetime.strptime(v, fmt)) if isinstance(v, str) else pd.Timestamp(v)

    if orbit_number is not None:
        orbs = [orbit_number] if np.isscalar(orbit_number) else list(orbit_number)
        df = pq.read_table(parquet_path, filters=[('orbit_number', 'in', orbs)]).to_pandas()
        #print(f"Loaded {len(df):,} rows  (orbit(s) {orbs})")
    elif trange is not None:
        t0, t1 = _to_ts(trange[0]), _to_ts(trange[1])
        df = pq.read_table(parquet_path, filters=[('time', '>=', t0), ('time', '<=', t1)]).to_pandas()
        #print(f"Loaded {len(df):,} rows  ({trange[0]} → {trange[1]})")
    else:
        df = pq.read_table(parquet_path).to_pandas()
        #print(f"Loaded {len(df):,} rows")

    return df

def find_substorm_loading(df, **kt17_kwargs):
    '''For a given df, look for times of substorm loading, and return a list of starts and stops'''

    loading_times = []

    # Unpack data
    t_obs = pd.to_datetime(df['time'])
    X      = df['ephx'].to_numpy()
    Y      = df['ephy'].to_numpy()
    Z      = df['ephz'].to_numpy()
    Bx_obs = df['magx'].to_numpy()
    By_obs = df['magy'].to_numpy()
    Bz_obs = df['magz'].to_numpy()

    # Work out timestep
    dt_sec = (t_obs.iloc[1] - t_obs.iloc[0]).total_seconds() if len(t_obs) > 1 else 1.0

    # Use pre-computed model field if present, otherwise compute it
    if 'Bx_mod' in df.columns and 'By_mod' in df.columns and 'Bz_mod' in df.columns:
        Bx_mod = df['Bx_mod'].to_numpy()
        By_mod = df['By_mod'].to_numpy()
        Bz_mod = df['Bz_mod'].to_numpy()
    else:
        _, Bx_mod, By_mod, Bz_mod = get_kt17_along_track(df=df, **kt17_kwargs)

    # Compute field-aligned strength
    B_perp, B_phi, B_para = transform_to_fac(Bx_obs, By_obs, Bz_obs, 
                                             Bx_mod, By_mod, Bz_mod, X, Y, Z)
    
    Bmag_mod = np.sqrt(Bx_mod**2+By_mod**2+Bz_mod**2)

    # Normalize to delta
    B_para = B_para - Bmag_mod

    # Compute time derivative of Bx variation relative to model
    Bx1 = Bx_obs - Bx_mod
    smooth_window = 30  # moving average window (seconds)
    smooth_samples = max(1, int(smooth_window / dt_sec))
    kernel = np.ones(smooth_samples) / smooth_samples
    Bx1_smooth = np.convolve(Bx1, kernel, mode='same')
    dBx1_dt = np.gradient(Bx1_smooth, dt_sec)

    # Find loading intervals
    t0    = 20 # min duration of loading (seconds)
    grace = 5 # max gap to bridge between adjacent intervals (seconds)
    mask = (Bx_obs < Bx_mod) & (Bx_obs<0) & (Bz_obs<Bz_mod) & (B_para>0)

    min_samples   = max(1, int(t0    / dt_sec))
    grace_samples = max(1, int(grace / dt_sec))
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    # Merge intervals separated by a gap <= grace_samples
    merged_starts, merged_ends = [], []
    for s, e in zip(starts, ends):
        if merged_ends and (s - merged_ends[-1]) <= grace_samples:
            merged_ends[-1] = e   # extend previous interval
        else:
            merged_starts.append(s)
            merged_ends.append(e)

    for s, e in zip(merged_starts, merged_ends):
        # Require the loading time is long enough, and starts with a decreasing Bx1
        if (e - s) >= min_samples and dBx1_dt[s] < 0:
            t_start = t_obs.iloc[s]
            t_end   = t_obs.iloc[min(e, len(t_obs) - 1)]
            loading_times.append([t_start, t_end])

    return loading_times
            
def find_DRs_following_substorm(df, loading_times, **kt17_kwargs):
    '''Look for DRs following substorm loading, and return times'''

    DR_times = []

    # Unpack data
    t_obs = pd.to_datetime(df['time'])
    X      = df['ephx'].to_numpy()
    Y      = df['ephy'].to_numpy()
    Z      = df['ephz'].to_numpy()
    Bx_obs = df['magx'].to_numpy()
    By_obs = df['magy'].to_numpy()
    Bz_obs = df['magz'].to_numpy()

    # Work out timestep
    dt_sec = (t_obs.iloc[1] - t_obs.iloc[0]).total_seconds() if len(t_obs) > 1 else 1.0

    # Use pre-computed model field if present, otherwise compute it
    if 'Bx_mod' in df.columns and 'By_mod' in df.columns and 'Bz_mod' in df.columns:
        Bx_mod = df['Bx_mod'].to_numpy()
        By_mod = df['By_mod'].to_numpy()
        Bz_mod = df['Bz_mod'].to_numpy()
    else:
        _, Bx_mod, By_mod, Bz_mod = get_kt17_along_track(df=df, **kt17_kwargs)

    Bmag_mod = np.sqrt(Bx_mod**2 + By_mod**2 + Bz_mod**2)

    # Compute field-aligned strength
    B_perp, B_phi, B_para = transform_to_fac(Bx_obs, By_obs, Bz_obs,
                                             Bx_mod, By_mod, Bz_mod, X, Y, Z)
    B_para = B_para - Bmag_mod

    
    # Find DR intervals 
    tmin = 60  # min duration of DR (seconds)
    tmax = 7*60 # max duration of DR (seconds)
    mask = (B_para > 0.05*Bmag_mod)

    min_samples = max(1, int(tmin / dt_sec))
    max_samples = max(1, int(tmax / dt_sec))
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    tdelay = pd.Timedelta(seconds=5 * 60)

    for s, e in zip(starts, ends):
        if (e - s) < min_samples or (e - s) > max_samples or s == 0:
            continue

        # Require mean(|B_para|) > mean(|B_perp|) and mean(|B_phi|) over the interval
        if np.mean(np.abs(B_para[s:e])) <= np.mean(np.abs(B_perp[s:e])):
            continue
        if np.mean(np.abs(B_para[s:e])) <= np.mean(np.abs(B_phi[s:e])):
            continue

        t_start = t_obs.iloc[s]
        t_end   = t_obs.iloc[min(e, len(t_obs) - 1)]

        # 1. t_start must be within tdelay of t_stop of at least one loading event
        near_loading = any(
            abs(t_start - lt[1]) <= tdelay
            for lt in loading_times
        )
        if not near_loading:
            continue

        # 2. must not overlap with any loading event
        overlaps_loading = any(
            t_start < lt[1] and t_end > lt[0]
            for lt in loading_times
        )
        if overlaps_loading:
            continue

        # 3. must not duplicate an already-added DR interval
        already_present = any(
            dr[0] == t_start and dr[1] == t_end
            for dr in DR_times
        )
        if already_present:
            continue

        DR_times.append([t_start, t_end])

    return DR_times

def filter_orbit_segment(orb_df):
    """
    Apply geometric selection criteria to a single orbit's DataFrame and return
    the trimmed segment that passes all checks.

    Returns a slice of orb_df covering the first qualifying continuous segment,
    or an empty DataFrame if the orbit is rejected.

    Criteria
    --------
    Spatial  : ephx < 0.1, -1.2 < ephz < 0.8, r < 1.8 R_M
    Azimuth  : fully within 90°–270° (nightside)
    Motion   : mean dZ > 0 (northward)
    """
    empty = orb_df.iloc[0:0]  # empty with same columns

    criteria = (
        (orb_df['ephx'] < 0.0) &
        (orb_df['ephz'] > -1.5) &
        (orb_df['ephz'] < 0.7) &
        ((orb_df['ephx']**2 + orb_df['ephy']**2 + orb_df['ephz']**2) < 3**2)
    ).to_numpy()

    # find start of first True run
    starts = np.where(np.diff(criteria.astype(int)) == 1)[0] + 1
    if criteria[0]:
        starts = np.concatenate([[0], starts])

    if len(starts) == 0:
        return empty

    seg_start = starts[0]

    # find end of that run
    ends = np.where(np.diff(criteria.astype(int)) == -1)[0] + 1
    ends = ends[ends > seg_start]
    seg_end = ends[0] - 1 if len(ends) > 0 else len(criteria) - 1

    seg_x = orb_df['ephx'].to_numpy()[seg_start:seg_end + 1]
    seg_y = orb_df['ephy'].to_numpy()[seg_start:seg_end + 1]

    # azimuthal check: must be fully within 90°–270° (nightside)
    phi = np.degrees(np.arctan2(seg_y, seg_x)) % 360
    if not np.all((phi >= 90) & (phi <= 270)):
        return empty

    # must be moving northward on average
    dz = np.diff(orb_df['ephz'].to_numpy())[seg_start:seg_end + 1]
    if not np.mean(dz) > 0:
        return empty

    return orb_df.iloc[seg_start:seg_end + 1]


    """Load previously saved filter parameters, falling back to FILTER_PARAMS defaults."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'filter_params.json')
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        params = dict(FILTER_PARAMS, **saved)
        print(f"Loaded filter params from {path}")
        return params
    return FILTER_PARAMS.copy()

def batch_plot_orbits(orbit_start, orbit_end, fig_dir='figures',
                      plot_func=None, highlight = False, 
                      show_loading_times = False, **kt17_kwargs):
    """
    Plot and save timeseries figures for each orbit in [orbit_start, orbit_end].

    Parameters
    ----------
    orbit_start, orbit_end : int
        Inclusive range of orbit numbers to process.
    fig_dir : str
        Directory to save figures into (created if needed).
    plot_func : callable, optional
        Plotting function to call per orbit. Must accept (trange, save_path=, **kwargs).
        Defaults to plot_mag_timeseries.
    **kt17_kwargs : forwarded to the plot function
    """
    if plot_func is None:
        plot_func = plot_field_aligned_timeseries
    import pyarrow.parquet as pq

    os.makedirs(fig_dir, exist_ok=True)

    parquet_path = os.path.expanduser('~/mercury_dipolarizations/MESSENGER_Full_Data_Ab_MSM.parquet')
    df_meta = pq.read_table(
        parquet_path,
        filters=[('orbit_number', '>=', orbit_start), ('orbit_number', '<=', orbit_end)],
        columns=['time', 'orbit_number', 'ephx', 'ephy', 'ephz']
    ).to_pandas()
    df_meta['time'] = pd.to_datetime(df_meta['time'])

    orbits = sorted(df_meta['orbit_number'].dropna().unique())
    print(f"Processing {len(orbits)} orbits ({orbit_start}–{orbit_end})...")
    selected_tranges = []

    for orb in orbits:
        orb_mask = df_meta['orbit_number'] == orb
        orb_df   = df_meta.loc[orb_mask]

        seg_df = filter_orbit_segment(orb_df)
        if seg_df.empty:
            print(f"  Orbit {int(orb):4d}  — rejected by criteria, skipping")
            continue

        times = seg_df['time'].to_numpy()
        t0 = pd.Timestamp(times[0])
        t1 = pd.Timestamp(times[-1])

        trange = [t0.strftime('%Y-%m-%d/%H:%M:%S'), t1.strftime('%Y-%m-%d/%H:%M:%S')]
        date_str = t0.strftime('%Y%m%d')
        fname = os.path.join(fig_dir, f'fac_{date_str}_orbit{int(orb):04d}.png')

        selected_tranges.append(trange)
        print(f"  Orbit {int(orb):4d}  {trange[0]} → {trange[1]}  →  {fname}")
        try:
            print("Plotting orbit#"+str(int(orb)))
            plot_func(trange, save_path=fname, highlight = highlight, 
                      show_loading_times = show_loading_times, **kt17_kwargs)
        except Exception as e:
            print(f"    FAILED: {e}")

    # --- overview trajectory plot of all selected segments ---
    if selected_tranges:
        print(f"\nPlotting overview trajectory for {len(selected_tranges)} selected segments...")
        dfs = [load_bowers_data_pkl(trange=tr) for tr in selected_tranges]
        df_all = pd.concat(dfs, ignore_index=True)
        R_M = 2440.0
        combined_traj = {'messenger': {
            'time':  pd.to_datetime(df_all['time']).to_numpy(),
            'x':     df_all['ephx'].to_numpy() * R_M,
            'y':     df_all['ephy'].to_numpy() * R_M,
            'z':     df_all['ephz'].to_numpy() * R_M,
            'coord': 'msm',
        }}
        overview_path = os.path.join(fig_dir, f'overview_orbits_{orbit_start}_{orbit_end}.png')
        fig = plot_messenger_trajectory(combined_traj, showFig = False)
        fig.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {overview_path}")

def _human_labels_for_orbit(orb, json_path=None):
    """
    Read human_dr_labels.json and return a human_labels dict suitable for
    passing to _plot_orbit_into_subfig, or None if the orbit has no entry.

        {'load_ivs': [(t0, t1), ...], 'dr_ivs': [(t0, t1), ...]}
    """
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), 'human_dr_labels.json')
    with open(json_path) as f:
        labels = json.load(f)
    entry = labels.get(str(orb))
    if not isinstance(entry, dict) or not entry.get('dr'):
        return None
    load_ivs, dr_ivs = [], []
    if 'loading_start' in entry and 'loading_stop' in entry:
        load_ivs.append((pd.Timestamp(entry['loading_start']),
                         pd.Timestamp(entry['loading_stop'])))
    if 'events' in entry:
        for ev in entry['events']:
            if 'dr_start' in ev and 'dr_stop' in ev:
                dr_ivs.append((pd.Timestamp(ev['dr_start']),
                               pd.Timestamp(ev['dr_stop'])))
    elif 'dr_start' in entry and 'dr_stop' in entry:
        dr_ivs.append((pd.Timestamp(entry['dr_start']),
                       pd.Timestamp(entry['dr_stop'])))
    return {'load_ivs': load_ivs, 'dr_ivs': dr_ivs}


def _plot_orbit_into_subfig(subfig, orb, label, fontsize=8,
                            ephemeris_labels=False, ephemeris_coords='xyz',
                            human_labels=None, bx_zero_line=False):
    '''Load one orbit, run the v3 filter, and populate a two-panel subfigure.

    Parameters
    ----------
    ephemeris_labels : bool
        If True, replace the x-axis time ticks with ephemeris values.
    ephemeris_coords : {'xyz', 'latlon'}
    bx_zero_line : bool
        If True, draw a vertical black dashed line on both panels at each
        time where the 60 s rolling average of observed Bx crosses zero.
        Coordinate system for the ephemeris labels (passed to
        set_ephemeris_ticklabels).  'xyz' shows X/Y/Z in R_M; 'latlon' shows
        east longitude, latitude, and altitude (r - 1 R_M).
    human_labels : dict or None
        If provided, use these pre-parsed interval lists for the coloured fill
        instead of running the automated filter.  Expected keys:
            'load_ivs' — list of (t_start, t_stop) Timestamps for loading
            'dr_ivs'   — list of (t_start, t_stop) Timestamps for DRs
        Typically built by passing the relevant entry from human_dr_labels.json.
        The automated filter is still run to compute the model field and FAC
        components; only the shading is overridden.
    '''

    full_orb_df = load_bowers_data_pkl(orbit_number=orb)
    orb_df      = filter_orbit_segment(full_orb_df)

    if orb_df.empty:
        subfig.suptitle(f'{label}: orbit {orb} (no data)', fontsize=fontsize)
        return []

    t_obs  = pd.to_datetime(orb_df['time'])
    X      = orb_df['ephx'].to_numpy()
    Y      = orb_df['ephy'].to_numpy()
    Z      = orb_df['ephz'].to_numpy()

    # full-orbit trajectory for the inset (before spatial filter)
    X_full = full_orb_df['ephx'].to_numpy()
    Y_full = full_orb_df['ephy'].to_numpy()
    Z_full = full_orb_df['ephz'].to_numpy()
    Bx_obs = orb_df['magx'].to_numpy()
    By_obs = orb_df['magy'].to_numpy()
    Bz_obs = orb_df['magz'].to_numpy()

    # ── v3 filter — always run to get model field + FAC components ──
    load_ivs_auto, dr_ivs_auto, dbg = _apply_dr_filter(orb_df)
    Bx_mod = dbg['Bxm'];  By_mod = dbg['Bym'];  Bz_mod = dbg['Bzm']
    B_perp = dbg['B_perp'];  B_phi = dbg['B_phi'];  B_para = dbg['B_para']

    # choose which intervals to shade
    if human_labels is not None:
        load_ivs = human_labels.get('load_ivs', [])
        dr_ivs   = human_labels.get('dr_ivs',   [])
    else:
        load_ivs = load_ivs_auto
        dr_ivs   = dr_ivs_auto

    # ── old filter (kept for reference) ──────────────────────────────────────
    # orb_df = orb_df.copy()
    # orb_df['Bx_mod'] = Bx_mod; orb_df['By_mod'] = By_mod; orb_df['Bz_mod'] = Bz_mod
    # loading_times = find_substorm_loading(orb_df)
    # DR_times      = find_DRs_following_substorm(orb_df, loading_times)
    # pairs = []
    # for ilt, loading_time in enumerate(loading_times):
    #     next_loading_start = loading_times[ilt + 1][0] if ilt + 1 < len(loading_times) else None
    #     for idr, DR_time in enumerate(DR_times):
    #         if DR_time[0] > loading_time[1]:
    #             if next_loading_start is None or DR_time[0] < next_loading_start:
    #                 pairs.append((ilt, idr)); break
    # paired_loading = {ilt for ilt, _ in pairs}
    # paired_dr      = {idr for _, idr in pairs}
    # ─────────────────────────────────────────────────────────────────────────

    ax_mag, ax_fac = subfig.subplots(nrows=2, sharex=True, gridspec_kw={'hspace': 0.05})
    subfig.suptitle(f'{label}: orbit {orb}', fontsize=fontsize)

    ax_mag.plot(t_obs, Bx_obs, color='red',   lw=0.7, label='Bx')
    ax_mag.plot(t_obs, Bx_mod, color='red',   lw=0.7, linestyle='dashed')
    ax_mag.plot(t_obs, By_obs, color='green',  lw=0.7, label='By')
    ax_mag.plot(t_obs, By_mod, color='green',  lw=0.7, linestyle='dashed')
    ax_mag.plot(t_obs, Bz_obs, color='blue',   lw=0.7, label='Bz')
    ax_mag.plot(t_obs, Bz_mod, color='blue',   lw=0.7, linestyle='dashed')
    ax_mag.legend(loc='lower right', fontsize=fontsize)
    ax_mag.set_ylabel('B (nT)', fontsize=fontsize)
    ax_mag.tick_params(axis='both', labelsize=fontsize)
    ax_mag.grid()

    ax_fac.plot(t_obs, B_perp, color='red',   lw=0.7, label=r'$\Delta B_\perp$')
    ax_fac.plot(t_obs, B_phi,  color='green',  lw=0.7, label=r'$\Delta B_\phi$')
    ax_fac.plot(t_obs, B_para, color='blue',   lw=0.7, label=r'$\Delta B_\parallel$')
    ax_fac.axhline(y=0, color='black', lw=0.5)
    ax_fac.legend(loc='lower right', fontsize=fontsize)
    ax_fac.set_ylim(-75, 50)
    ax_fac.set_ylabel(r'$\Delta$B (nT)', fontsize=fontsize)
    ax_fac.tick_params(axis='both', labelsize=fontsize)
    ax_fac.grid()
    if ephemeris_labels:
        ax_fac.set_xlabel('')
        set_ephemeris_ticklabels(ax_fac, orb_df, fontsize=fontsize,
                                 coords=ephemeris_coords)
    else:
        ax_fac.set_xlabel('Time', fontsize=fontsize)
        plt.setp(ax_fac.get_xticklabels(), rotation=30, ha='right')

    for ax in [ax_mag, ax_fac]:
        for lt in load_ivs:
            ax.axvspan(lt[0], lt[1], color='green', alpha=0.35, zorder=0)
        for iv in dr_ivs:
            ax.axvspan(iv[0], iv[1], color='gold', alpha=0.35, zorder=0)

    if bx_zero_line:
        t_s      = (t_obs - t_obs.iloc[0]).dt.total_seconds().to_numpy()
        dt_s     = np.median(np.diff(t_s)) if len(t_s) > 1 else 1.0
        win      = max(1, int(round(60.0 / dt_s)))
        bx_roll  = (pd.Series(Bx_obs)
                    .rolling(win, center=True, min_periods=1).mean()
                    .to_numpy())
        signs    = np.sign(bx_roll)
        signs[signs == 0] = 1
        crossings = np.where(np.diff(signs) != 0)[0]
        for ci in crossings:
            denom = bx_roll[ci] - bx_roll[ci + 1]
            frac  = bx_roll[ci] / denom if denom != 0 else 0.5
            t_cross = t_obs.iloc[ci] + frac * (t_obs.iloc[ci + 1] - t_obs.iloc[ci])
            for ax in [ax_mag, ax_fac]:
                ax.axvline(t_cross, color='black', lw=1.0, ls='--', alpha=0.8, zorder=5)

    ax_mag.set_xlim(t_obs.iloc[0], t_obs.iloc[-1])

    # -- tiny YZ orbit inset (upper-left of ax_mag) --
    ax_inset = ax_mag.inset_axes([0.01, 0.89, 0.25, 0.25])
    # nightside (X<=0) behind planet, dayside (X>0) in front
    # use masked arrays to preserve gaps — boolean indexing collapses arrays and
    # causes matplotlib to connect non-contiguous segments with spurious lines
    day   = X_full > 0
    night = ~day
    Y_night = np.ma.masked_where(day,   Y_full)
    Z_night = np.ma.masked_where(day,   Z_full)
    Y_day   = np.ma.masked_where(night, Y_full)
    Z_day   = np.ma.masked_where(night, Z_full)
    # viewed anti-sunward (-X direction): dayside (X>0) is the far side → behind planet
    # use an opaque white circle to fully occlude the dayside line before the
    # coloured (semi-transparent) patch is drawn — zorder alone is insufficient
    # because a semi-transparent patch lets lines bleed through regardless
    ax_inset.plot(Y_day,   Z_day,   color='steelblue', lw=0.6, zorder=0)
    ax_inset.add_patch(plt.Circle((0, -0.2), 1.0, color='white',       zorder=1))
    ax_inset.add_patch(plt.Circle((0, -0.2), 1.0, color='saddlebrown', alpha=0.35, zorder=2))
    ax_inset.plot(Y_night, Z_night, color='steelblue', lw=0.6, zorder=3)
    ax_inset.plot(Y, Z, color='gold', lw=1.0, zorder=4)
    ax_inset.scatter(Y_full[0],  Z_full[0],  marker='o', s=8, color='lime',   zorder=5)
    ax_inset.scatter(Y_full[-1], Z_full[-1], marker='s', s=8, color='tomato', zorder=5)
    ax_inset.set_xlim(1.5, -1.5)
    ax_inset.set_ylim(-2, 1)
    ax_inset.set_aspect('equal')
    ax_inset.tick_params(labelsize=4, length=2, pad=1)
    ax_inset.set_xlabel('Y (R$_M$)', fontsize=4, labelpad=1)
    ax_inset.set_ylabel('Z (R$_M$)', fontsize=4, labelpad=1)
    ax_inset.grid(True, alpha=0.2, lw=0.3)

    # pair each DR to its closest loading interval for the return value
    results = []
    for iv in dr_ivs:
        if not load_ivs:
            break
        paired = min(load_ivs, key=lambda lt: abs((iv[0] - lt[1]).total_seconds()))
        results.append([paired[0], paired[1], iv[0], iv[1]])
    return results

def event_filtering_toolkit_v1(orbit_start=None, orbit_end=None, fontsize=8,
                               human_labels=False, json_path=None):
    '''Interactive, development toolkit for refining our event selection criteria.
    Shows set examples, and then summarizes all the orbits in sets of 8.

    Parameters
    ----------
    orbit_start, orbit_end : int, optional
        Inclusive range of orbits to plot. Required unless human_labels=True.
    human_labels : bool
        If True, plot only the orbits marked as DR=true in the human labels JSON,
        ignoring orbit_start/orbit_end.
    json_path : str, optional
        Path to the human labels JSON. Defaults to <script_dir>/human_dr_labels.json.
    '''

    # --- Example figures (fixed case studies) ---
    example_positives = [3451, 3455, 3963, 3965]
    #example_positives = [2159, 2163, 2687, 2689, 2902, 2937, 2941, 2957, 3189,
    example_negatives = [3433, 3443, 3690, 3921]

    ncols = np.max([len(example_negatives), len(example_positives)])
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("DR filter test", fontsize=fontsize + 2)
    subfigs = fig.subfigures(nrows=2, ncols=ncols, hspace=0.05, wspace=0.05).reshape(2, ncols)

    for irow, (label, orb_list) in enumerate([('DR', example_positives), ('Non-DR', example_negatives)]):
        for iorb, orb in enumerate(orb_list):
            _plot_orbit_into_subfig(subfigs[irow, iorb], orb, label, fontsize=fontsize)

    plt.show()
    plt.close()

    # --- Build orbit list ---
    if human_labels:
        if json_path is None:
            json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'human_dr_labels.json')
        with open(json_path) as f:
            raw = json.load(f)
        orbits = sorted(
            int(k) for k, v in raw.items()
            if (v if isinstance(v, bool) else v.get('dr', False))
        )
        print(f"Plotting {len(orbits)} human-labelled DR orbits from {json_path}")
    else:
        if orbit_start is None or orbit_end is None:
            raise ValueError("Provide orbit_start and orbit_end, or set human_labels=True.")
        orbits = list(range(orbit_start, orbit_end + 1))

    # --- Batch figures: loop through orbit list, 8 per figure ---
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    chunk_size = 8
    ncols_batch, nrows_batch = 4, 2

    # pre-parse human times for quick lookup
    # supports new interval format (loading_start/stop, dr_start/stop)
    # and old point format (loading_time, dr_time) for backward compatibility
    # value: (load_start, load_stop, dr_start, dr_stop)  — stop=start for old format
    human_times = {}
    if human_labels and raw:
        for k, v in raw.items():
            entry = v if isinstance(v, dict) else {'dr': v}
            if not entry.get('dr'):
                continue
            if 'loading_start' in entry and 'dr_start' in entry:
                human_times[int(k)] = (
                    pd.Timestamp(entry['loading_start']),
                    pd.Timestamp(entry['loading_stop']),
                    pd.Timestamp(entry['dr_start']),
                    pd.Timestamp(entry['dr_stop']),
                )
            elif 'loading_time' in entry and 'dr_time' in entry:
                # old single-point format — use same time for start and stop
                lt = pd.Timestamp(entry['loading_time'])
                dt = pd.Timestamp(entry['dr_time'])
                human_times[int(k)] = (lt, lt, dt, dt)

    events = {}  # {orbit: [[loading_start, loading_stop, DR_start, DR_stop], ...]}

    for chunk_start in range(0, len(orbits), chunk_size):
        chunk = orbits[chunk_start:chunk_start + chunk_size]

        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(f"Orbits {chunk[0]}–{chunk[-1]}", fontsize=fontsize + 2)
        subfigs = fig.subfigures(nrows=nrows_batch, ncols=ncols_batch,
                                 hspace=0.05, wspace=0.05).reshape(nrows_batch, ncols_batch)

        for idx, orb in enumerate(chunk):
            irow, icol = divmod(idx, ncols_batch)
            sf = subfigs[irow, icol]
            orb_pairs = _plot_orbit_into_subfig(sf, orb, f'Orbit {orb}', fontsize=fontsize)
            if orb_pairs:
                events[orb] = orb_pairs
            # overlay human-labeled intervals (or single-point fallback)
            if orb in human_times:
                ls, le, ds, de = human_times[orb]
                for ax in sf.get_axes():
                    if ax.get_subplotspec() is None:
                        continue   # skip inset
                    ax.axvline(ls, color='darkgreen',  lw=1.0, ls='-',  zorder=5)
                    ax.axvline(le, color='darkgreen',  lw=1.0, ls='--', zorder=5)
                    ax.axvline(ds, color='darkorange', lw=1.0, ls='-',  zorder=5)
                    ax.axvline(de, color='darkorange', lw=1.0, ls='--', zorder=5)
                    if ls != le:   # shade interval if we have start≠stop
                        ax.axvspan(ls, le, color='darkgreen',  alpha=0.10, zorder=4)
                        ax.axvspan(ds, de, color='darkorange', alpha=0.10, zorder=4)

        # Hide unused panels if chunk < 8
        for idx in range(len(chunk), chunk_size):
            irow, icol = divmod(idx, ncols_batch)
            subfigs[irow, icol].set_visible(False)

        save_path = os.path.join(fig_dir, f'orbits_{chunk[0]:04d}_{chunk[-1]:04d}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")

    '''
    # --- Superposed epoch plot of all detected events ---
    epoch_window  = 600   # seconds either side of DR start
    n_grid        = 500   # points in common epoch grid
    t_grid        = np.linspace(-epoch_window, epoch_window, n_grid)

    # Accumulate interpolated detrended traces onto the common grid
    all_dBx, all_dBy, all_dBz = [], [], []

    fig_ep, axes_ep = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    fig_ep.suptitle('Superposed epoch: all DR events (t=0 = DR start)', fontsize=fontsize + 2)
    colors = {'dBx': 'red', 'dBy': 'green', 'dBz': 'blue'}

    for orb, orb_pairs in events.items():
        orb_df = load_bowers_data_pkl(orbit_number=orb)
        orb_df = filter_orbit_segment(orb_df)
        t_obs  = pd.to_datetime(orb_df['time'])
        Bx_obs = orb_df['magx'].to_numpy()
        By_obs = orb_df['magy'].to_numpy()
        Bz_obs = orb_df['magz'].to_numpy()

        _, Bx_mod, By_mod, Bz_mod = get_kt17_along_track(df=orb_df)
        dBx = Bx_obs - Bx_mod
        dBy = By_obs - By_mod
        dBz = Bz_obs - Bz_mod

        for event in orb_pairs:
            DR_start = pd.Timestamp(event[2])
            t_epoch  = (t_obs - DR_start).dt.total_seconds().to_numpy()

            mask = (t_epoch >= -epoch_window) & (t_epoch <= epoch_window)
            axes_ep[0].plot(t_epoch[mask], dBx[mask], color=colors['dBx'], lw=0.4, alpha=0.2)
            axes_ep[1].plot(t_epoch[mask], dBy[mask], color=colors['dBy'], lw=0.4, alpha=0.2)
            axes_ep[2].plot(t_epoch[mask], dBz[mask], color=colors['dBz'], lw=0.4, alpha=0.2)

            # Interpolate onto common grid for statistics
            all_dBx.append(np.interp(t_grid, t_epoch[mask], dBx[mask]))
            all_dBy.append(np.interp(t_grid, t_epoch[mask], dBy[mask]))
            all_dBz.append(np.interp(t_grid, t_epoch[mask], dBz[mask]))

    # Mean ± 1 std overlay
    for arr, ax, color, label in [
        (all_dBx, axes_ep[0], colors['dBx'], r'$\Delta B_x$'),
        (all_dBy, axes_ep[1], colors['dBy'], r'$\Delta B_y$'),
        (all_dBz, axes_ep[2], colors['dBz'], r'$\Delta B_z$'),
    ]:
        if arr:
            stack = np.array(arr)
            mean  = np.mean(stack, axis=0)
            std   = np.std(stack,  axis=0)
            ax.plot(t_grid, mean, color=color, lw=2.0, label=f'{label} mean (n={len(arr)})')
            ax.fill_between(t_grid, mean - std, mean + std, color=color, alpha=0.2)

    axes_ep[0].set_ylabel(r'$\Delta B_x$ (nT)', fontsize=fontsize)
    axes_ep[1].set_ylabel(r'$\Delta B_y$ (nT)', fontsize=fontsize)
    axes_ep[2].set_ylabel(r'$\Delta B_z$ (nT)', fontsize=fontsize)
    axes_ep[2].set_xlabel('Epoch time (s)', fontsize=fontsize)

    for ax in axes_ep:
        ax.axvline(x=0, color='black', lw=0.8, ls='--')
        ax.axhline(y=0, color='black', lw=0.4, ls=':')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.legend(loc='upper left', fontsize=fontsize)

    plt.tight_layout()
    epoch_path = os.path.join(fig_dir, 'superposed_epoch.png')
    fig_ep.savefig(epoch_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig_ep)
    print(f"  Saved: {epoch_path}")
    '''

    return events

    '''Print a step-by-step trace of why an orbit does or does not produce a detected event.'''

def event_filtering_toolkit_v2(orbit_start, orbit_end, fontsize=8, json_path=None):
    """
    Human-in-the-loop DR labelling toolkit to gather a small dataset

    Page view  : 8 subfigures per page.  Click a subfigure to expand it.
    Expanded   : full-size plot fills the figure.
                 1st click → marks the substorm LOADING time (green dashed line).
                 2nd click → marks the DR time (orange dashed line); orbit saved as DR.
                 3rd click anywhere in the plot → collapses back to page view.
    Back button: collapses without recording the orbit.
    Confirm    : saves the page and advances.
    Quit       : stops early, saves everything done so far.

    JSON format: {"orbit_number": {"dr": true/false,
                                   "loading_time": <ISO>,   # only for DR=true
                                   "dr_time":      <ISO>}}

    Parameters
    ----------
    orbit_start, orbit_end : int — inclusive range of orbits to review
    fontsize               : int — base font size
    json_path              : str — output JSON path
                             (default: <script_dir>/human_dr_labels.json)
    """
    from matplotlib.widgets import Button

    if json_path is None:
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'human_dr_labels.json'
        )
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if os.path.exists(json_path):
        with open(json_path) as f:
            raw = json.load(f)
        # migrate old bool-only format {"orb": true} → {"orb": {"dr": true}}
        labels = {k: ({"dr": v} if isinstance(v, bool) else v) for k, v in raw.items()}
    else:
        labels = {}

    orbits           = list(range(orbit_start, orbit_end + 1))
    chunk_size       = 8
    ncols_b, nrows_b = 4, 2
    n_pages          = -(-len(orbits) // chunk_size)

    print(f"\nReviewing orbits {orbit_start}–{orbit_end}  ({len(orbits)} total, {n_pages} pages)")
    print("Click a subfigure to expand → click loading time → click DR time → click to return.\n")

    for chunk_start in range(0, len(orbits), chunk_size):
        chunk = orbits[chunk_start:chunk_start + chunk_size]
        page  = chunk_start // chunk_size + 1

        fig = plt.figure(figsize=(16, 8))
        _page_title = (
            f"Orbits {chunk[0]}–{chunk[-1]}  |  page {page}/{n_pages}"
            "  —  click a subfigure to expand & label"
        )
        fig.suptitle(_page_title, fontsize=fontsize + 2)

        subfigs = fig.subfigures(
            nrows=nrows_b, ncols=ncols_b, hspace=0.05, wspace=0.05
        ).reshape(nrows_b, ncols_b)

        for row in subfigs:
            for sf in row:
                sf.patch.set_visible(False)

        ax_to_idx     = {}   # axes object → chunk index
        idx_to_subfig = {}   # chunk index → subfigure
        valid_indices = set()

        for idx, orb in enumerate(chunk):
            irow, icol = divmod(idx, ncols_b)
            sf    = subfigs[irow, icol]
            pairs = _plot_orbit_into_subfig(sf, orb, f'[{idx}] Orbit {orb}', fontsize=fontsize)
            if pairs is not None:
                valid_indices.add(idx)
                idx_to_subfig[idx] = sf
                for ax in sf.get_axes():
                    ax_to_idx[ax] = idx

        for idx in range(len(chunk), chunk_size):
            irow, icol = divmod(idx, ncols_b)
            subfigs[irow, icol].set_visible(False)

        selected      = set()   # chunk indices confirmed as DR
        false_indices = set()   # chunk indices marked as NOT a DR
        quit_flag = [False]

        # --- expansion state (mutable container shared by all closures) ---
        exp = {
            'active':   False,
            'idx':      None,
            # n_clicks:  0 = loading_start  1 = loading_stop
            #            2 = dr_start       3 = dr_stop (auto-saves & collapses)
            'n_clicks': 0,
            'times':    [None, None, None, None],  # [ls, le, ds, de]
            'axes':     [],     # overlay axes removed on collapse
            'vlines':   [],     # 8 lines: ls×2, le×2, ds×2, de×2 (top+bot each)
            'spans':    [],     # axvspan patches added after clicks 1 and 3
            'back_ax':    None,
            'back_btn':   None,
            'notadr_ax':  None,
            'notadr_btn': None,
        }

        def _set_shade(idx, on, color='limegreen'):
            sf = idx_to_subfig[idx]
            sf.patch.set_facecolor(color)
            sf.patch.set_alpha(0.3)
            sf.patch.set_visible(on)

        # pre-shade any orbits already recorded in the JSON
        for idx, orb in enumerate(chunk):
            entry = labels.get(str(orb))
            if entry is None or idx not in valid_indices:
                continue
            if entry.get('dr', False):
                selected.add(idx)
                _set_shade(idx, True, color='limegreen')
            elif entry.get('dr') is False:
                false_indices.add(idx)
                _set_shade(idx, True, color='salmon')

        def _collapse():
            for sp in exp['spans']:
                sp.remove()
            exp['spans'].clear()
            for ax in exp['axes']:
                ax.remove()
            exp['axes'].clear()
            exp['vlines'].clear()
            if exp['back_ax'] is not None:
                exp['back_ax'].remove()
            exp['back_ax']    = None
            exp['back_btn']   = None
            if exp['notadr_ax'] is not None:
                exp['notadr_ax'].remove()
            exp['notadr_ax']  = None
            exp['notadr_btn'] = None
            for idx2 in valid_indices:
                idx_to_subfig[idx2].set_visible(True)
            for idx2 in selected:
                _set_shade(idx2, True, color='limegreen')
            for idx2 in false_indices:
                _set_shade(idx2, True, color='salmon')
            for idx2 in range(len(chunk), chunk_size):
                r2, c2 = divmod(idx2, ncols_b)
                subfigs[r2, c2].set_visible(False)
            exp['active']   = False
            exp['idx']      = None
            exp['n_clicks'] = 0
            exp['times']    = [None, None, None, None]
            fig.suptitle(_page_title, fontsize=fontsize + 2, color='black')
            fig.canvas.draw_idle()

        def _expand(idx):
            orb = chunk[idx]
            # hide subfigures and show a loading message while data is fetched
            for idx2 in valid_indices:
                idx_to_subfig[idx2].set_visible(False)
            fig.suptitle(f'Orbit {orb}  —  loading data…', fontsize=fontsize + 2, color='gray')
            fig.canvas.draw()   # force synchronous render before the slow load

            orb_df = load_bowers_data_pkl(orbit_number=orb)
            orb_df = filter_orbit_segment(orb_df)
            if orb_df.empty:
                _collapse()
                return

            t_e  = pd.to_datetime(orb_df['time'])
            Xe   = orb_df['ephx'].to_numpy()
            Ye   = orb_df['ephy'].to_numpy()
            Ze   = orb_df['ephz'].to_numpy()
            Bx_e = orb_df['magx'].to_numpy()
            By_e = orb_df['magy'].to_numpy()
            Bz_e = orb_df['magz'].to_numpy()

            _, Bxm, Bym, Bzm = get_kt17_along_track(df=orb_df)
            Bmod = np.sqrt(Bxm**2 + Bym**2 + Bzm**2)
            orb_df = orb_df.copy()
            orb_df['Bx_mod'] = Bxm
            orb_df['By_mod'] = Bym
            orb_df['Bz_mod'] = Bzm

            Bp, Bphi, Bpar = transform_to_fac(Bx_e, By_e, Bz_e, Bxm, Bym, Bzm, Xe, Ye, Ze)
            Bpar = Bpar - Bmod

            lt_e = find_substorm_loading(orb_df)
            dr_e = find_DRs_following_substorm(orb_df, lt_e)

            # create two full-width overlay axes on the page figure
            ax_top = fig.add_axes([0.07, 0.48, 0.88, 0.44])
            ax_bot = fig.add_axes([0.07, 0.09, 0.88, 0.35])
            exp['axes'] = [ax_top, ax_bot]

            ax_top.plot(t_e, Bx_e, color='red',   lw=0.8, label='Bx')
            ax_top.plot(t_e, Bxm,  color='red',   lw=0.8, ls='--')
            ax_top.plot(t_e, By_e, color='green',  lw=0.8, label='By')
            ax_top.plot(t_e, Bym,  color='green',  lw=0.8, ls='--')
            ax_top.plot(t_e, Bz_e, color='blue',   lw=0.8, label='Bz')
            ax_top.plot(t_e, Bzm,  color='blue',   lw=0.8, ls='--')
            ax_top.legend(fontsize=fontsize, loc='lower right')
            ax_top.set_ylabel('B (nT)', fontsize=fontsize)
            ax_top.tick_params(labelbottom=False)
            ax_top.grid()

            ax_bot.plot(t_e, Bp,   color='red',   lw=0.8, label=r'$\Delta B_\perp$')
            ax_bot.plot(t_e, Bphi, color='green',  lw=0.8, label=r'$\Delta B_\phi$')
            ax_bot.plot(t_e, Bpar, color='blue',   lw=0.8, label=r'$\Delta B_\parallel$')
            ax_bot.axhline(0, color='k', lw=0.5)
            ax_bot.legend(fontsize=fontsize, loc='lower right')
            ax_bot.set_ylim(-100, 100)
            ax_bot.set_ylabel(r'$\Delta$B (nT)', fontsize=fontsize)
            ax_bot.set_xlabel('Time', fontsize=fontsize)
            ax_bot.grid()
            plt.setp(ax_bot.get_xticklabels(), rotation=30, ha='right')

            for lt in lt_e:
                ax_top.axvspan(lt[0], lt[1], color='green', alpha=0.25, zorder=0)
                ax_bot.axvspan(lt[0], lt[1], color='green', alpha=0.25, zorder=0)
            for dr in dr_e:
                ax_top.axvspan(dr[0], dr[1], color='gold', alpha=0.25, zorder=0)
                ax_bot.axvspan(dr[0], dr[1], color='gold', alpha=0.25, zorder=0)

            xlim = (t_e.iloc[0], t_e.iloc[-1])
            ax_top.set_xlim(*xlim)
            ax_bot.set_xlim(*xlim)

            # 8 invisible marker lines (top+bot for each of the 4 click points)
            # order: load_start×2, load_stop×2, dr_start×2, dr_stop×2
            t0_num = mdates.date2num(t_e.iloc[0].to_pydatetime())
            exp['vlines'] = [
                ax_top.axvline(t0_num, color='darkgreen',  lw=2, ls='-',  visible=False),
                ax_bot.axvline(t0_num, color='darkgreen',  lw=2, ls='-',  visible=False),
                ax_top.axvline(t0_num, color='darkgreen',  lw=2, ls='--', visible=False),
                ax_bot.axvline(t0_num, color='darkgreen',  lw=2, ls='--', visible=False),
                ax_top.axvline(t0_num, color='darkorange', lw=2, ls='-',  visible=False),
                ax_bot.axvline(t0_num, color='darkorange', lw=2, ls='-',  visible=False),
                ax_top.axvline(t0_num, color='darkorange', lw=2, ls='--', visible=False),
                ax_bot.axvline(t0_num, color='darkorange', lw=2, ls='--', visible=False),
            ]
            exp['spans'] = []

            ax_back       = fig.add_axes([0.62, 0.005, 0.08, 0.04])
            btn_back      = Button(ax_back, 'Back', color='#e8e8e8', hovercolor='#c0c0c0')
            btn_back.on_clicked(lambda _ev: _collapse())
            exp['back_ax']  = ax_back
            exp['back_btn'] = btn_back

            ax_notadr  = fig.add_axes([0.71, 0.005, 0.10, 0.04])
            btn_notadr = Button(ax_notadr, 'Not a DR', color='#f0d4d4', hovercolor='#e08080')
            def _on_not_a_dr(_ev):
                idx_done = exp['idx']
                orb_done = chunk[idx_done]
                false_indices.add(idx_done)
                selected.discard(idx_done)          # un-DR if previously marked
                labels[str(orb_done)] = {'dr': False}
                _collapse()
                _set_shade(idx_done, True, color='salmon')
                fig.canvas.draw_idle()
            btn_notadr.on_clicked(_on_not_a_dr)
            exp['notadr_ax']  = ax_notadr
            exp['notadr_btn'] = btn_notadr

            exp['active']   = True
            exp['idx']      = idx
            exp['n_clicks'] = 0
            exp['times']    = [None, None, None, None]

            fig.suptitle(
                f'Orbit {orb}  —  1/4) click LOADING start  (left edge of green region)',
                fontsize=fontsize + 2, color='darkgreen',
            )
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes is None:
                return

            if not exp['active']:
                # ── page mode: click to expand ──────────────────────────────
                idx = ax_to_idx.get(event.inaxes)
                if idx is None or idx not in valid_indices:
                    return
                _expand(idx)

            else:
                # ── expanded mode: collect four time clicks ──────────────────
                ax_top, ax_bot = exp['axes']
                if event.inaxes not in (ax_top, ax_bot) or event.xdata is None:
                    return

                t_clicked = pd.Timestamp(mdates.num2date(event.xdata).replace(tzinfo=None))
                t_num     = mdates.date2num(t_clicked.to_pydatetime())
                n         = exp['n_clicks']
                orb_cur   = chunk[exp['idx']]

                if n == 0:
                    # loading start
                    exp['times'][0] = t_clicked
                    for v in exp['vlines'][0:2]:
                        v.set_xdata([t_num, t_num]); v.set_visible(True)
                    exp['n_clicks'] = 1
                    fig.suptitle(
                        f'Orbit {orb_cur}  —  2/4) click LOADING stop  (right edge of green)',
                        fontsize=fontsize + 2, color='darkgreen',
                    )
                    fig.canvas.draw_idle()

                elif n == 1:
                    # loading stop → shade the loading window
                    exp['times'][1] = t_clicked
                    for v in exp['vlines'][2:4]:
                        v.set_xdata([t_num, t_num]); v.set_visible(True)
                    for ax in (ax_top, ax_bot):
                        exp['spans'].append(
                            ax.axvspan(exp['times'][0], t_clicked,
                                       color='darkgreen', alpha=0.15, zorder=6)
                        )
                    exp['n_clicks'] = 2
                    fig.suptitle(
                        f'Orbit {orb_cur}  —  3/4) click DR start  (left edge of gold)',
                        fontsize=fontsize + 2, color='darkorange',
                    )
                    fig.canvas.draw_idle()

                elif n == 2:
                    # DR start
                    exp['times'][2] = t_clicked
                    for v in exp['vlines'][4:6]:
                        v.set_xdata([t_num, t_num]); v.set_visible(True)
                    exp['n_clicks'] = 3
                    fig.suptitle(
                        f'Orbit {orb_cur}  —  4/4) click DR stop  (right edge of gold)',
                        fontsize=fontsize + 2, color='darkorange',
                    )
                    fig.canvas.draw_idle()

                elif n == 3:
                    # DR stop → shade, save, auto-collapse
                    exp['times'][3] = t_clicked
                    for v in exp['vlines'][6:8]:
                        v.set_xdata([t_num, t_num]); v.set_visible(True)
                    for ax in (ax_top, ax_bot):
                        exp['spans'].append(
                            ax.axvspan(exp['times'][2], t_clicked,
                                       color='darkorange', alpha=0.15, zorder=6)
                        )
                    fig.canvas.draw_idle()

                    idx_done = exp['idx']
                    orb_done = chunk[idx_done]
                    selected.add(idx_done)
                    false_indices.discard(idx_done)   # un-false if previously marked
                    labels[str(orb_done)] = {
                        'dr':            True,
                        'loading_start': exp['times'][0].isoformat(),
                        'loading_stop':  exp['times'][1].isoformat(),
                        'dr_start':      exp['times'][2].isoformat(),
                        'dr_stop':       t_clicked.isoformat(),
                    }
                    _collapse()
                    _set_shade(idx_done, True, color='limegreen')
                    fig.canvas.draw_idle()

        def on_confirm(_ev):
            plt.close(fig)

        def on_quit(_ev):
            quit_flag[0] = True
            plt.close(fig)

        cid = fig.canvas.mpl_connect('button_press_event', on_click)

        ax_confirm  = fig.add_axes([0.40, 0.005, 0.13, 0.04])
        ax_quit_btn = fig.add_axes([0.54, 0.005, 0.07, 0.04])
        btn_confirm = Button(ax_confirm,  'Confirm  ✓', color='#d4f0d4', hovercolor='#90e090')
        btn_quit    = Button(ax_quit_btn, 'Quit',       color='#f0d4d4', hovercolor='#e09090')
        btn_confirm.on_clicked(on_confirm)
        btn_quit.on_clicked(on_quit)

        plt.show(block=True)
        fig.canvas.mpl_disconnect(cid)

        with open(json_path, 'w') as f:
            json.dump(labels, f, indent=2, sort_keys=True)

        dr_orbits = [chunk[i] for i in sorted(selected)]
        print(f"  Page {page}: DR orbits = {dr_orbits}")

        if quit_flag[0]:
            break

    dr_count = sum(1 for v in labels.values() if v.get('dr', False))
    print(f"\nDone. {dr_count} DR orbits across {len(labels)} reviewed.  Labels → {json_path}")
    return labels

_FILTER_DEFAULTS = dict(
    DERIV_SMOOTH_S     = 120.0,
    LOADING_MIN_DUR_S  =  20.0,
    LOADING_GAP_FILL_S =   5.0,
    LOADING_BX_NEG     = True,
    LOADING_BPERP_POS  = True,
    LOADING_BZ_BELOW   = False,
    LOADING_BPARA_POS  = False,
    LOADING_DBX_DECR   = False,
    DR_BPARA_FRAC      =  0.05,
    DR_GAP_FILL_S      =  10.0,
    DR_MIN_DUR_S       =  45.0,
    DR_MAX_DUR_S       =  7 * 60.0,
    DR_PERP_RATIO      =   1.0,
    DR_PHI_RATIO       =   1.0,
    DR_MAX_DELAY_S     =  4 * 60.0,
    DR_OVERLAP_ALLOW_S =  10.0,
    DR_CONTEXT_S       =  60.0,
    DR_CONTEXT_RATIO   =   1.1,
)

def _apply_dr_filter(orb_df, p=None):
    """
    Core DR detection filter.

    Parameters
    ----------
    orb_df : DataFrame returned by load_bowers_data_pkl / filter_orbit_segment
    p      : dict of parameter overrides (keys match _FILTER_DEFAULTS).
             Missing keys fall back to _FILTER_DEFAULTS.

    Returns
    -------
    loading_ivs : list of [t_start, t_end] Timestamp pairs
    dr_ivs      : list of [t_start, t_end] Timestamp pairs (accepted DRs)
    debug       : dict of intermediate arrays (for diagnostic plots)
    """
    if p is None:
        p = {}
    def _g(k):
        return p.get(k, _FILTER_DEFAULTS[k])

    DERIV_SMOOTH_S     = _g('DERIV_SMOOTH_S')
    LOADING_MIN_DUR_S  = _g('LOADING_MIN_DUR_S')
    LOADING_GAP_FILL_S = _g('LOADING_GAP_FILL_S')
    LOADING_BX_NEG     = _g('LOADING_BX_NEG')
    LOADING_BPERP_POS  = _g('LOADING_BPERP_POS')
    LOADING_BZ_BELOW   = _g('LOADING_BZ_BELOW')
    LOADING_BPARA_POS  = _g('LOADING_BPARA_POS')
    LOADING_DBX_DECR   = _g('LOADING_DBX_DECR')
    DR_BPARA_FRAC      = _g('DR_BPARA_FRAC')
    DR_GAP_FILL_S      = _g('DR_GAP_FILL_S')
    DR_MIN_DUR_S       = _g('DR_MIN_DUR_S')
    DR_MAX_DUR_S       = _g('DR_MAX_DUR_S')
    DR_PERP_RATIO      = _g('DR_PERP_RATIO')
    DR_PHI_RATIO       = _g('DR_PHI_RATIO')
    DR_MAX_DELAY_S     = _g('DR_MAX_DELAY_S')
    DR_OVERLAP_ALLOW_S = _g('DR_OVERLAP_ALLOW_S')
    DR_CONTEXT_S       = _g('DR_CONTEXT_S')
    DR_CONTEXT_RATIO   = _g('DR_CONTEXT_RATIO')

    t_obs  = pd.to_datetime(orb_df['time'])
    dt_sec = (t_obs.iloc[1] - t_obs.iloc[0]).total_seconds() if len(t_obs) > 1 else 1.0
    X  = orb_df['ephx'].to_numpy()
    Y  = orb_df['ephy'].to_numpy()
    Z  = orb_df['ephz'].to_numpy()
    Bx = orb_df['magx'].to_numpy()
    By = orb_df['magy'].to_numpy()
    Bz = orb_df['magz'].to_numpy()

    if 'Bx_mod' in orb_df.columns:
        Bxm = orb_df['Bx_mod'].to_numpy()
        Bym = orb_df['By_mod'].to_numpy()
        Bzm = orb_df['Bz_mod'].to_numpy()
    else:
        _, Bxm, Bym, Bzm = get_kt17_along_track(df=orb_df)
    Bmag_mod = np.sqrt(Bxm**2 + Bym**2 + Bzm**2)

    B_perp, B_phi, B_para = transform_to_fac(Bx, By, Bz, Bxm, Bym, Bzm, X, Y, Z)
    B_para = B_para - Bmag_mod

    sm        = max(1, int(DERIV_SMOOTH_S / dt_sec))
    kernel    = np.ones(sm) / sm
    Bx_smooth = np.convolve(Bx, kernel, mode='same')
    By_smooth = np.convolve(By, kernel, mode='same')
    Bz_smooth = np.convolve(Bz, kernel, mode='same')
    Bx1       = Bx_smooth - Bxm
    By1       = By_smooth - Bym
    Bz1       = Bz_smooth - Bzm
    dBx1_dt   = np.convolve(np.gradient(Bx1, dt_sec), kernel, mode='same')
    dBy1_dt   = np.convolve(np.gradient(By1, dt_sec), kernel, mode='same')
    dBz1_dt   = np.convolve(np.gradient(Bz1, dt_sec), kernel, mode='same')

    load_mask = (Bx < Bxm)
    if LOADING_BX_NEG:    load_mask &= (Bxm < 0)
    if LOADING_BPERP_POS: load_mask &= (B_perp > 0)
    if LOADING_BZ_BELOW:  load_mask &= (Bz < Bzm)
    if LOADING_BPARA_POS: load_mask &= (B_para > 0)

    min_s = max(1, int(LOADING_MIN_DUR_S  / dt_sec))
    gap_s = max(1, int(LOADING_GAP_FILL_S / dt_sec))
    pad = np.concatenate([[False], load_mask, [False]])
    d   = np.diff(pad.astype(int))
    sts, ens = np.where(d == 1)[0], np.where(d == -1)[0]
    msts, mens = [], []
    for s, e in zip(sts, ens):
        if msts and (s - mens[-1]) <= gap_s:
            mens[-1] = e
        else:
            msts.append(s); mens.append(e)
    loading_ivs = []
    for s, e in zip(msts, mens):
        if (e - s) >= min_s:
            if LOADING_DBX_DECR and dBx1_dt[s] >= 0:
                continue
            loading_ivs.append([t_obs.iloc[s], t_obs.iloc[min(e, len(t_obs)-1)]])

    dr_mask = (B_para > DR_BPARA_FRAC * Bmag_mod)
    pad2 = np.concatenate([[False], dr_mask, [False]])
    d2   = np.diff(pad2.astype(int))
    sts2, ens2 = np.where(d2 == 1)[0], np.where(d2 == -1)[0]
    dr_gap_s = max(1, int(DR_GAP_FILL_S / dt_sec))
    msts2, mens2 = [], []
    for s, e in zip(sts2, ens2):
        if msts2 and (s - mens2[-1]) <= dr_gap_s:
            mens2[-1] = e
        else:
            msts2.append(s); mens2.append(e)
    sub_starts, sub_ends = [], []
    dr_breakpoint_times = []
    for s, e in zip(msts2, mens2):
        seg = dBz1_dt[s:e]
        if len(seg) < 3:
            sub_starts.append(s); sub_ends.append(e)
            continue
        dseg  = np.diff(seg)
        signs = np.sign(dseg)
        for k in range(1, len(signs)):
            if signs[k] == 0:
                signs[k] = signs[k - 1]
        tp = np.where(np.diff(signs) != 0)[0] + 1
        if len(tp) == 0:
            sub_starts.append(s); sub_ends.append(e)
            continue
        breaks = [0] + tp.tolist() + [e - s]
        for i in range(len(breaks) - 1):
            sub_s = s + breaks[i]
            sub_e = s + breaks[i + 1]
            if sub_e > sub_s:
                sub_starts.append(sub_s); sub_ends.append(sub_e)
        for bp in tp:
            dr_breakpoint_times.append(t_obs.iloc[s + bp])
    msts2, mens2 = sub_starts, sub_ends

    dr_ivs      = []
    dr_rejected = []
    tdelay           = pd.Timedelta(seconds=DR_MAX_DELAY_S)
    min_dr           = max(1, int(DR_MIN_DUR_S / dt_sec))
    max_dr           = max(1, int(DR_MAX_DUR_S / dt_sec))
    overlap_allowance = pd.Timedelta(seconds=DR_OVERLAP_ALLOW_S)
    for s, e in zip(msts2, mens2):
        dur       = e - s
        t_s       = t_obs.iloc[s]
        t_e       = t_obs.iloc[min(e, len(t_obs)-1)]
        para_mean = np.mean(np.abs(B_para[s:e]))
        perp_mean = np.mean(np.abs(B_perp[s:e]))
        phi_mean  = np.mean(np.abs(B_phi[s:e]))
        delays    = [abs(t_s - lt[1]) for lt in loading_ivs]
        min_delay = min(delays) if delays else None

        if dur < min_dr:
            dr_rejected.append((t_s, t_e, f'too short ({dur*dt_sec:.0f}s < {DR_MIN_DUR_S:.0f}s)'))
            continue
        if dur > max_dr:
            dr_rejected.append((t_s, t_e, f'too long ({dur*dt_sec:.0f}s > {DR_MAX_DUR_S:.0f}s)'))
            continue
        if para_mean <= DR_PERP_RATIO * perp_mean:
            dr_rejected.append((t_s, t_e, f'|B∥|={para_mean:.2f} ≤ {DR_PERP_RATIO}×|B⊥|={perp_mean:.2f}'))
            continue
        if para_mean <= DR_PHI_RATIO * phi_mean:
            dr_rejected.append((t_s, t_e, f'|B∥|={para_mean:.2f} ≤ {DR_PHI_RATIO}×|Bφ|={phi_mean:.2f}'))
            continue
        if not any(d <= tdelay for d in delays):
            dr_rejected.append((t_s, t_e, f'no loading within delay (closest={min_delay})'))
            continue
        overlapping = [
            (lt,
             (lt[0] < t_s and t_s < lt[1] - overlap_allowance)
             or
             (lt[0] >= t_s and lt[0] < t_e and lt[1] <= t_e)
            )
            for lt in loading_ivs
        ]
        if any(ov for _, ov in overlapping):
            detail = '; '.join(f'[{lt[0].strftime("%H:%M:%S")}–{lt[1].strftime("%H:%M:%S")}]'
                               for lt, ov in overlapping if ov)
            dr_rejected.append((t_s, t_e, f'DR starts during loading: {detail}'))
            continue
        B_para_abs  = B_para + Bmag_mod
        ctx_samples = max(1, int(DR_CONTEXT_S / dt_sec))
        pre_s       = max(0, s - ctx_samples)
        post_e      = min(len(B_para_abs), e + ctx_samples)
        pre_idx     = np.arange(pre_s, s)
        if len(pre_idx) > 0 and loading_ivs:
            t_pre      = t_obs.iloc[pre_idx]
            in_loading = np.zeros(len(pre_idx), dtype=bool)
            for lt in loading_ivs:
                in_loading |= (t_pre >= lt[0]).to_numpy() & (t_pre <= lt[1]).to_numpy()
            pre_vals = B_para_abs[pre_idx[~in_loading]]
        else:
            pre_vals = B_para_abs[pre_s:s]
        ctx_vals    = np.concatenate([pre_vals, B_para_abs[e:post_e]])
        ctx_mean    = np.mean(ctx_vals) if len(ctx_vals) > 0 else 0.0
        dr_mean_abs = np.mean(B_para_abs[s:e])
        if dr_mean_abs <= DR_CONTEXT_RATIO * ctx_mean:
            dr_rejected.append((t_s, t_e,
                f'B∥ not elevated vs context '
                f'(DR={dr_mean_abs:.2f} ≤ {DR_CONTEXT_RATIO}×ctx={ctx_mean:.2f})'))
            continue
        dr_ivs.append([t_s, t_e])

    debug = dict(
        t_obs=t_obs, Bx=Bx, By=By, Bz=Bz, Bxm=Bxm, Bym=Bym, Bzm=Bzm,
        Bx_smooth=Bx_smooth, By_smooth=By_smooth, Bz_smooth=Bz_smooth,
        Bmag_mod=Bmag_mod, B_para=B_para, B_perp=B_perp, B_phi=B_phi,
        Bx1=Bx1, By1=By1, Bz1=Bz1,
        dBx1_dt=dBx1_dt, dBy1_dt=dBy1_dt, dBz1_dt=dBz1_dt,
        c_bx_below=Bx < Bxm,
        c_bx_neg=Bxm < 0,
        c_bperp_pos=B_perp > 0,
        c_bz_below=Bz < Bzm,
        c_bpara_pos=B_para > 0,
        load_mask=load_mask,
        dr_mask=dr_mask,
        dr_rejected=dr_rejected,
        dr_breakpoint_times=dr_breakpoint_times,
    )
    return loading_ivs, dr_ivs, debug

def event_filtering_toolkit_v3(json_path=None, params=None, silent=False):
    """
    Evaluate the DR detection filter against every labeled event in the JSON.
    Prints a TP/TN/FP/FN report, then shows a diagnostic plot for each
    mischaracterised event so you can see exactly which criteria failed.

    Parameters
    ----------
    params  : optional dict of parameter overrides (same names as the caps
              variables in the FILTER PARAMETERS block below).
    silent  : if True, suppress all prints and plots (for use by the optimizer).

    Returns
    -------
    fp + fn : int — total misclassifications (the value the optimizer minimises).
    """

    # ── FILTER PARAMETERS ── edit these freely ──────────────────────────────
    DERIV_SMOOTH_S      = 120.0      # smoothing window for B components and their derivatives (s)

    LOADING_MIN_DUR_S   = 20.0      # minimum loading interval duration (s)
    LOADING_GAP_FILL_S  = 5.0      # merge loading intervals closer than this (s)
    LOADING_BX_NEG      = True      # require Bx_mod < 0 (implies Bx_obs < 0 via base condition)
    LOADING_BPERP_POS   = True      # Require Bperp > 0
    LOADING_BZ_BELOW    = False      # require Bz_obs < Bz_mod
    LOADING_BPARA_POS   = False      # require B_para > 0
    LOADING_DBX_DECR    = False      # require dBx1/dt < 0 at interval start

    DR_BPARA_FRAC       =  0.05     # B_para > FRAC * |B_model|
    DR_GAP_FILL_S       =  10.0      # merge DR candidate intervals closer than this (s)
    DR_MIN_DUR_S        =  45.0      # minimum DR duration (s)
    DR_MAX_DUR_S        = 7 * 60.0  # maximum DR duration (s)
    DR_PERP_RATIO       =  1.0      # require mean|B_para| > RATIO * mean|B_perp|
    DR_PHI_RATIO        =  1.0      # require mean|B_para| > RATIO * mean|B_phi|
    DR_MAX_DELAY_S      =  4 * 60.0 # DR start must be within this of loading end
    DR_OVERLAP_ALLOW_S  =  10.0      # DR may start this many seconds before loading end
    DR_CONTEXT_S        =  60.0      # seconds before/after DR for context (loading times excluded from pre-window)
    DR_CONTEXT_RATIO    =  1.1      # mean B_para in DR must exceed this × mean in context windows
    # ─────────────────────────────────────────────────────────────────────────
    # Apply any overrides passed by the optimizer
    if params:
        DERIV_SMOOTH_S      = params.get('DERIV_SMOOTH_S',      DERIV_SMOOTH_S)
        LOADING_MIN_DUR_S   = params.get('LOADING_MIN_DUR_S',   LOADING_MIN_DUR_S)
        LOADING_GAP_FILL_S  = params.get('LOADING_GAP_FILL_S',  LOADING_GAP_FILL_S)
        LOADING_BX_NEG      = params.get('LOADING_BX_NEG',       LOADING_BX_NEG)
        LOADING_BPERP_POS   = params.get('LOADING_BPERP_POS',    LOADING_BPERP_POS)
        LOADING_BZ_BELOW    = params.get('LOADING_BZ_BELOW',     LOADING_BZ_BELOW)
        LOADING_BPARA_POS   = params.get('LOADING_BPARA_POS',    LOADING_BPARA_POS)
        LOADING_DBX_DECR    = params.get('LOADING_DBX_DECR',     LOADING_DBX_DECR)
        DR_BPARA_FRAC       = params.get('DR_BPARA_FRAC',        DR_BPARA_FRAC)
        DR_GAP_FILL_S       = params.get('DR_GAP_FILL_S',        DR_GAP_FILL_S)
        DR_MIN_DUR_S        = params.get('DR_MIN_DUR_S',         DR_MIN_DUR_S)
        DR_MAX_DUR_S        = params.get('DR_MAX_DUR_S',         DR_MAX_DUR_S)
        DR_PERP_RATIO       = params.get('DR_PERP_RATIO',        DR_PERP_RATIO)
        DR_PHI_RATIO        = params.get('DR_PHI_RATIO',         DR_PHI_RATIO)
        DR_MAX_DELAY_S      = params.get('DR_MAX_DELAY_S',       DR_MAX_DELAY_S)
        DR_OVERLAP_ALLOW_S  = params.get('DR_OVERLAP_ALLOW_S',   DR_OVERLAP_ALLOW_S)
        DR_CONTEXT_S        = params.get('DR_CONTEXT_S',         DR_CONTEXT_S)
        DR_CONTEXT_RATIO    = params.get('DR_CONTEXT_RATIO',     DR_CONTEXT_RATIO)

    if json_path is None:
        json_path = os.path.join(_SCRIPT_DIR, 'human_dr_labels.json')

    with open(json_path) as f:
        labels = json.load(f)

    positives = {int(k): v for k, v in labels.items()
                 if isinstance(v, dict) and v.get('dr') is True}
    negatives = {int(k): v for k, v in labels.items()
                 if isinstance(v, dict) and v.get('dr') is False}
    if not silent:
        print(f"Loaded {len(positives)} positives, {len(negatives)} negatives from {json_path}\n")

    # ── inner filter: thin wrapper that passes the current params to _apply_dr_filter
    _p = dict(
        DERIV_SMOOTH_S=DERIV_SMOOTH_S, LOADING_MIN_DUR_S=LOADING_MIN_DUR_S,
        LOADING_GAP_FILL_S=LOADING_GAP_FILL_S, LOADING_BX_NEG=LOADING_BX_NEG,
        LOADING_BPERP_POS=LOADING_BPERP_POS, LOADING_BZ_BELOW=LOADING_BZ_BELOW,
        LOADING_BPARA_POS=LOADING_BPARA_POS, LOADING_DBX_DECR=LOADING_DBX_DECR,
        DR_BPARA_FRAC=DR_BPARA_FRAC, DR_GAP_FILL_S=DR_GAP_FILL_S,
        DR_MIN_DUR_S=DR_MIN_DUR_S, DR_MAX_DUR_S=DR_MAX_DUR_S,
        DR_PERP_RATIO=DR_PERP_RATIO, DR_PHI_RATIO=DR_PHI_RATIO,
        DR_MAX_DELAY_S=DR_MAX_DELAY_S, DR_OVERLAP_ALLOW_S=DR_OVERLAP_ALLOW_S,
        DR_CONTEXT_S=DR_CONTEXT_S, DR_CONTEXT_RATIO=DR_CONTEXT_RATIO,
    )
    def _run_filter(orb_df):
        return _apply_dr_filter(orb_df, _p)

    # ── evaluate each event ──────────────────────────────────────────────────
    results = []   # (orb, truth, predicted, entry, debug)

    for truth_val, event_dict in [(True, positives), (False, negatives)]:
        for orb, entry in event_dict.items():
            if not silent:
                print(f"  Orbit {orb} ({'POS' if truth_val else 'NEG'}) ...", end=' ', flush=True)
            orb_df = load_bowers_data_pkl(orbit_number=orb)
            orb_df = filter_orbit_segment(orb_df)
            if orb_df.empty:
                if not silent: print("no data — skipped")
                continue
            try:
                load_ivs, dr_ivs, dbg = _run_filter(orb_df)
            except Exception as ex:
                if not silent: print(f"ERROR: {ex}")
                continue
            # For positive events with timing info, require the detected DR to
            # actually overlap the hand-labelled interval, not just any detection.
            if truth_val and 'dr_start' in entry and 'dr_stop' in entry:
                h_ds = pd.Timestamp(entry['dr_start'])
                h_de = pd.Timestamp(entry['dr_stop'])
                predicted = any(iv[0] < h_de and iv[1] > h_ds for iv in dr_ivs)
                if not predicted and len(dr_ivs) > 0 and not silent:
                    print(f"(detected DR but not at labelled time "
                          f"{h_ds.strftime('%H:%M:%S')}–{h_de.strftime('%H:%M:%S')})", end=' ')
            else:
                predicted = len(dr_ivs) > 0
            outcome = ('TP' if truth_val and predicted else
                       'TN' if not truth_val and not predicted else
                       'FP' if not truth_val and predicted else 'FN')
            if not silent: print(outcome)
            results.append((orb, truth_val, predicted, entry, dbg, load_ivs, dr_ivs, outcome, orb_df))

    # ── report ───────────────────────────────────────────────────────────────
    tp = sum(1 for r in results if r[7] == 'TP')
    tn = sum(1 for r in results if r[7] == 'TN')
    fp = sum(1 for r in results if r[7] == 'FP')
    fn = sum(1 for r in results if r[7] == 'FN')
    n  = tp + tn + fp + fn
    prec   = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    f1     = 2*prec*recall / (prec+recall) if (prec+recall) else float('nan')
    if not silent:
        print(f"\n{'─'*40}")
        print(f"  Evaluated : {n}  ({tp+fn} pos / {tn+fp} neg)")
        print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        print(f"  Precision : {prec:.2f}")
        print(f"  Recall    : {recall:.2f}")
        print(f"  F1        : {f1:.2f}")
        print(f"{'─'*40}\n")

    if silent:
        return fp + fn

    # ── debug plots for misses ────────────────────────────────────────────────
    misses = [r for r in results if r[7] in ('FP', 'FN')]
    if not misses:
        print("All events correctly classified — no debug plots needed.")
        return fp + fn

    for orb, truth, predicted, entry, dbg, load_ivs, dr_ivs, outcome, orb_df in misses:
        print(f"\n{'─'*50}")
        print(f"Orbit {orb}  [{outcome}]")
        print(f"  Loading intervals detected: {len(load_ivs)}")
        for lt in load_ivs:
            print(f"    {lt[0].strftime('%H:%M:%S')} – {lt[1].strftime('%H:%M:%S')}")
        print(f"  DR candidates rejected: {len(dbg['dr_rejected'])}")
        for t_s, t_e, reason in dbg['dr_rejected']:
            print(f"    {t_s.strftime('%H:%M:%S')} – {t_e.strftime('%H:%M:%S')}  →  {reason}")
        if not dbg['dr_rejected']:
            print("    (no candidates passed the B_para threshold)")
        print(f"  DR intervals accepted: {len(dr_ivs)}")
        t    = dbg['t_obs']
        xlim = (t.iloc[0]  + pd.Timedelta(minutes=1),
                t.iloc[-1] - pd.Timedelta(minutes=1))
        fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True,
                                  gridspec_kw={'height_ratios': [2, 2, 2, 1.2, 0.8]})
        fig.suptitle(
            f'Orbit {orb}  —  {outcome}  '
            f'(truth={"DR" if truth else "NOT DR"}, '
            f'predicted={"DR" if predicted else "NOT DR"})',
            fontsize=10, color='firebrick' if outcome == 'FN' else 'darkorange',
        )

        # ── panel 1: Bx obs vs model ─────────────────────────────────────────
        ax = axes[0]
        for raw, smooth, mod, color, lbl in [
            (dbg['Bx'], dbg['Bx_smooth'], dbg['Bxm'], 'red',   'Bx'),
            (dbg['By'], dbg['By_smooth'], dbg['Bym'], 'green', 'By'),
            (dbg['Bz'], dbg['Bz_smooth'], dbg['Bzm'], 'blue',  'Bz'),
        ]:
            ax.plot(t, raw,    color=color, lw=0.4, alpha=0.3)
            ax.plot(t, smooth, color=color, lw=1.2, label=f'{lbl} (smooth)')
            ax.plot(t, mod,    color=color, lw=0.8, ls='--', label=f'{lbl} mod')
        ax.axhline(0, color='k', lw=0.4)
        ax.set_ylabel('B (nT)')
        ax.legend(fontsize=6, ncol=6, loc='upper right')
        ax.grid(True, alpha=0.2)

        # ── panel 2: FAC components ──────────────────────────────────────────
        ax = axes[1]
        ax.plot(t, dbg['B_para'], color='blue',   lw=0.8, label=r'$\Delta B_\parallel$')
        ax.plot(t, dbg['B_perp'], color='red',    lw=0.8, label=r'$\Delta B_\perp$')
        ax.plot(t, dbg['B_phi'],  color='green',  lw=0.8, label=r'$\Delta B_\phi$')
        ax.plot(t, DR_BPARA_FRAC * dbg['Bmag_mod'],
                color='blue', lw=1, ls=':', label=f'DR threshold ({DR_BPARA_FRAC:.2f}|B|)')
        ax.axhline(0, color='k', lw=0.4)
        ax.set_ylabel('ΔB (nT)')
        ax.legend(fontsize=6, ncol=4, loc='upper right')
        ax.grid(True, alpha=0.2)

        # ── panel 3: derivatives of smoothed residuals ───────────────────────
        ax = axes[2]
        for db1, color, lbl in [
            (dbg['dBx1_dt'], 'red',   'dBx1/dt'),
            (dbg['dBy1_dt'], 'green', 'dBy1/dt'),
            (dbg['dBz1_dt'], 'blue',  'dBz1/dt'),
        ]:
            ax.plot(t, db1, color=color, lw=0.8, label=lbl)
        for i, bp in enumerate(dbg['dr_breakpoint_times']):
            ax.axvline(bp, color='black', lw=1.0, ls=':', alpha=0.7,
                       label='dBz1/dt turning pt' if i == 0 else '')
        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6)
        ax.set_ylabel('nT s⁻¹')
        ax.set_ylim(-0.75, .75)
        ax.legend(fontsize=6, ncol=4, loc='upper right')
        ax.grid(True, alpha=0.2)

        # ── panel 4: individual loading boolean criteria ──────────────────────
        ax = axes[3]
        criteria_rows = [
            ('Bx<Bx_mod',  dbg['c_bx_below'],  'steelblue',    True),
            ('Bxmod<0',    dbg['c_bx_neg'],    'dodgerblue',   LOADING_BX_NEG),
            ('B⊥>0',       dbg['c_bperp_pos'], 'mediumpurple', LOADING_BPERP_POS),
            ('Bz<Bz_mod',  dbg['c_bz_below'],  'tomato',       LOADING_BZ_BELOW),
            ('B∥>0',       dbg['c_bpara_pos'], 'seagreen',     LOADING_BPARA_POS),
            ('load∩',      dbg['load_mask'],   'darkgreen',    True),
            ('DR mask',    dbg['dr_mask'],     'darkorange',   True),
        ]
        criteria_rows = [(lbl, mask, col) for lbl, mask, col, enabled in criteria_rows if enabled]
        n_rows = len(criteria_rows)
        t_arr  = t.to_numpy()
        for i, (label, mask, color) in enumerate(reversed(criteria_rows)):
            y = i / n_rows
            h = 1.0 / n_rows
            # grey background so unset regions are visible
            ax.axhspan(y + 0.02, y + h - 0.02, color='#e8e8e8', zorder=0)
            # colored patches where criterion is True
            pad  = np.concatenate([[False], mask, [False]])
            diff = np.diff(pad.astype(int))
            sts  = np.where(diff ==  1)[0]
            ens  = np.where(diff == -1)[0]
            for s, e in zip(sts, ens):
                x0 = t_arr[min(s,   len(t_arr)-1)]
                x1 = t_arr[min(e-1, len(t_arr)-1)]
                ax.axvspan(x0, x1, ymin=y + 0.02, ymax=y + h - 0.02,
                           color=color, alpha=0.75, zorder=1)
            # label on the right margin so it never overlaps data
            ax.text(1.002, y + h * 0.5, label, fontsize=8, fontweight='bold',
                    va='center', ha='left', color=color,
                    transform=ax.get_yaxis_transform(), clip_on=False)
        ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_ylabel('Criteria', fontsize=8)
        ax.grid(False)

        # ── panel 5: detected vs human-labeled intervals ─────────────────────
        ax = axes[4]
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel('Intervals', fontsize=7)

        def _shade_iv(ivs, color, ymin, ymax, label_prefix):
            for i, iv in enumerate(ivs):
                ax.axvspan(iv[0], iv[1], ymin=ymin, ymax=ymax, color=color, alpha=0.5)
                lbl = f'{label_prefix}' if i == 0 else ''
                ax.axvline(iv[0], color=color, lw=1, ls='-')
                ax.axvline(iv[1], color=color, lw=1, ls='--')

        _shade_iv(load_ivs, 'green',      0.5, 1.0, 'det. loading')
        _shade_iv(dr_ivs,   'darkorange', 0.5, 1.0, 'det. DR')
        ax.text(t.iloc[0], 0.78, '  detected', fontsize=6, color='k')

        # human labels
        if 'loading_start' in entry:
            ls = pd.Timestamp(entry['loading_start'])
            le = pd.Timestamp(entry['loading_stop'])
            ds = pd.Timestamp(entry['dr_start'])
            de = pd.Timestamp(entry['dr_stop'])
            ax.axvspan(ls, le, ymin=0.0, ymax=0.48, color='darkgreen',  alpha=0.4)
            ax.axvspan(ds, de, ymin=0.0, ymax=0.48, color='darkorange', alpha=0.4)
            ax.text(t.iloc[0], 0.22, '  human', fontsize=6, color='k')

        fig.canvas.draw()   # needed so tick locations are populated before ephemeris labels
        set_ephemeris_ticklabels(ax, orb_df, fontsize=7)
        for ax2 in axes[:-1]:
            plt.setp(ax2.get_xticklabels(), visible=False)

        # shade all panels with detected / human intervals
        for ax2 in axes[:4]:
            for iv in load_ivs:
                ax2.axvspan(iv[0], iv[1], color='green',     alpha=0.08, zorder=0)
            for iv in dr_ivs:
                ax2.axvspan(iv[0], iv[1], color='darkorange', alpha=0.10, zorder=0)
            if 'loading_start' in entry:
                ax2.axvspan(pd.Timestamp(entry['loading_start']),
                            pd.Timestamp(entry['loading_stop']),
                            color='darkgreen', alpha=0.12, zorder=0)
                ax2.axvspan(pd.Timestamp(entry['dr_start']),
                            pd.Timestamp(entry['dr_stop']),
                            color='gold', alpha=0.18, zorder=0)

        axes[0].set_xlim(*xlim)
        plt.tight_layout()
        plt.show()

    return fp + fn

def optimize_filter_v3(json_path=None, max_iter=500):
    """
    Optimize the continuous filter parameters in event_filtering_toolkit_v3
    by minimising FP + FN using derivative-free optimisation (Nelder-Mead).

    Boolean flags are left at the values hardcoded in event_filtering_toolkit_v3.
    Only continuous parameters are tuned.

    Parameters
    ----------
    json_path : path to human_dr_labels.json (defaults to the standard location)
    max_iter  : maximum optimizer iterations

    Returns
    -------
    best_params : dict of optimised parameter values
    """
    from scipy.optimize import minimize

    # ── continuous parameters to tune and their (lower, upper) bounds ────────
    param_names = [
        'DERIV_SMOOTH_S',
        'LOADING_MIN_DUR_S',
        'LOADING_GAP_FILL_S',
        'DR_BPARA_FRAC',
        'DR_GAP_FILL_S',
        'DR_MIN_DUR_S',
        'DR_MAX_DUR_S',
        'DR_PERP_RATIO',
        'DR_PHI_RATIO',
        'DR_MAX_DELAY_S',
        'DR_OVERLAP_ALLOW_S',
        'DR_CONTEXT_S',
        'DR_CONTEXT_RATIO',
    ]
    bounds = [
        (10.0,  120.0),   # DERIV_SMOOTH_S
        (5.0,   120.0),   # LOADING_MIN_DUR_S
        (0.0,    30.0),   # LOADING_GAP_FILL_S
        (0.01,   0.30),   # DR_BPARA_FRAC
        (0.0,    30.0),   # DR_GAP_FILL_S
        (10.0,  120.0),   # DR_MIN_DUR_S
        (120.0, 600.0),   # DR_MAX_DUR_S
        (0.1,    5.0),    # DR_PERP_RATIO
        (0.1,    5.0),    # DR_PHI_RATIO
        (30.0,  600.0),   # DR_MAX_DELAY_S
        (0.0,    60.0),   # DR_OVERLAP_ALLOW_S
        (10.0,  300.0),   # DR_CONTEXT_S
        (0.5,   10.0),    # DR_CONTEXT_RATIO
    ]
    # ─────────────────────────────────────────────────────────────────────────

    # starting point: defaults from event_filtering_toolkit_v3
    x0 = [120.0, 20.0, 5.0, 0.05, 10.0, 45.0, 420.0, 1.0, 1.0, 240.0, 10.0, 60.0, 1.1]

    call_count = [0]

    def objective(x):
        call_count[0] += 1
        # Clamp to bounds (Nelder-Mead can stray outside)
        x_clamped = [max(lo, min(hi, v)) for v, (lo, hi) in zip(x, bounds)]
        params = dict(zip(param_names, x_clamped))
        error  = event_filtering_toolkit_v3(json_path=json_path, params=params, silent=True)
        print(f"  [{call_count[0]:4d}]  error={error}  "
              + "  ".join(f"{k.split('_')[-1]}={v:.3g}" for k, v in params.items()),
              flush=True)
        return error

    print(f"Starting optimisation over {len(param_names)} parameters …\n")
    result = minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 0.5, 'fatol': 0.5, 'disp': True},
    )

    best_params = dict(zip(param_names, result.x))
    print(f"\n{'─'*50}")
    print(f"Optimisation finished.  Best error = {result.fun:.0f}")
    print(f"{'─'*50}")
    for k, v in best_params.items():
        print(f"  {k:25s} = {v:.4g}")

    return best_params

def run_automated_detection(orb_start, orb_end, params=None, out_path=None,
                            save_plots=False, fig_dir=None, force=False):
    """
    Run the DR detection filter over orbits orb_start..orb_end (inclusive)
    and save the results to a JSON file in the same format as human_dr_labels.json.

    For orbits with no detected DR the entry is {"dr": false}.
    For orbits with one DR the entry matches the human format exactly:
        {"dr": true, "dr_start": ..., "dr_stop": ...,
                     "loading_start": ..., "loading_stop": ...}
    For orbits with multiple detected DRs the entry stores a list:
        {"dr": true, "events": [{"dr_start":..., "dr_stop":...,
                                  "loading_start":..., "loading_stop":...}, ...]}

    Parameters
    ----------
    orb_start  : int  — first orbit number (inclusive)
    orb_end    : int  — last  orbit number (inclusive)
    params     : dict of filter-parameter overrides (same keys as _FILTER_DEFAULTS)
    out_path   : output file path; defaults to automated_dr_labels.json next to this script
    save_plots : if True, save a Bx/By/Bz overview figure for each detected DR
    fig_dir    : directory for saved figures; defaults to figures/ next to this script
    force      : if True, reprocess orbits already present in the output file

    Returns
    -------
    labels : dict  (same structure written to out_path)
    """
    if out_path is None:
        out_path = os.path.join(_SCRIPT_DIR, 'automated_dr_labels.json')

    if save_plots:
        if fig_dir is None:
            fig_dir = os.path.join(_SCRIPT_DIR, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

    # load existing results so we can resume / append without re-running orbits
    if os.path.exists(out_path) and not force:
        with open(out_path) as f:
            labels = json.load(f)
        in_range = sum(1 for o in range(orb_start, orb_end + 1) if str(o) in labels)
        print(f"Resuming from {out_path}  ({len(labels)} total, {in_range} in requested range already done)")
    elif os.path.exists(out_path) and force:
        with open(out_path) as f:
            labels = json.load(f)
        print(f"Force reprocessing {orb_end - orb_start + 1} orbits (file has {len(labels)} entries)")
    else:
        labels = {}

    def _fmt(ts):
        return ts.isoformat()

    def _pair_loading(dr_iv, loading_ivs):
        """Return the loading interval whose end is closest to and before dr_start."""
        t_s = dr_iv[0]
        candidates = [(abs((t_s - lt[1]).total_seconds()), lt)
                      for lt in loading_ivs
                      if lt[1] <= t_s + pd.Timedelta(seconds=_FILTER_DEFAULTS['DR_MAX_DELAY_S'])]
        if not candidates:
            # fall back to closest overall
            candidates = [(abs((t_s - lt[1]).total_seconds()), lt) for lt in loading_ivs]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[0])[1]

    p = params or {}

    total       = orb_end - orb_start + 1
    n_processed = 0
    n_dr        = 0

    for orb in range(orb_start, orb_end + 1):
        orb_str = str(orb)
        if orb_str in labels and not force:
            continue   # already processed

        n_processed += 1
        print(f"  Orbit {orb} ({n_processed}/{total}) ...", end=' ', flush=True)
        orb_df = load_bowers_data_pkl(orbit_number=orb)
        orb_df = filter_orbit_segment(orb_df)
        if orb_df.empty:
            print("no data — skipped")
            continue

        try:
            load_ivs, dr_ivs, _ = _apply_dr_filter(orb_df, p)
        except Exception as ex:
            print(f"ERROR: {ex}")
            continue

        if not dr_ivs:
            labels[orb_str] = {"dr": False}
            print("no DR")
        elif len(dr_ivs) == 1:
            paired = _pair_loading(dr_ivs[0], load_ivs)
            entry = {"dr": True,
                     "dr_start": _fmt(dr_ivs[0][0]),
                     "dr_stop":  _fmt(dr_ivs[0][1])}
            if paired:
                entry["loading_start"] = _fmt(paired[0])
                entry["loading_stop"]  = _fmt(paired[1])
            labels[orb_str] = entry
            n_dr += 1
            print(f"DR  {dr_ivs[0][0].strftime('%H:%M:%S')}–{dr_ivs[0][1].strftime('%H:%M:%S')}")
        else:
            events = []
            for iv in dr_ivs:
                paired = _pair_loading(iv, load_ivs)
                ev = {"dr_start": _fmt(iv[0]), "dr_stop": _fmt(iv[1])}
                if paired:
                    ev["loading_start"] = _fmt(paired[0])
                    ev["loading_stop"]  = _fmt(paired[1])
                events.append(ev)
            labels[orb_str] = {"dr": True, "events": events}
            n_dr += 1
            print(f"{len(dr_ivs)} DRs detected")

        # optional plot for orbits with detected DRs
        if save_plots and dr_ivs:
            fname = os.path.join(fig_dir, f'orbit_{orb}.png')
            plot_field_aligned_timeseries(
                df=orb_df,
                ext_dr_ivs=dr_ivs,
                ext_load_ivs=load_ivs,
                save_path=fname,
                fontsize=9,
            )
            print(f"    saved {fname}")

        # save after every orbit so progress survives interruption
        with open(out_path, 'w') as f:
            json.dump(labels, f, indent=2)

    total_dr = sum(1 for v in labels.values() if isinstance(v, dict) and v.get('dr'))
    print(f"\nDone.  {n_dr} DR orbits found this run  ({n_processed} processed).")
    print(f"File total: {total_dr} DR orbits across {len(labels)} entries.")
    print(f"Labels saved → {out_path}")
    return labels

def plot_event_locations(json_path=None, fig=None, orbit_labels=False):
    """
    Read human_dr_labels.json (or *json_path*) and plot the MSM trajectory
    segments of every loading interval (green) and every DR (yellow) in two
    panels:

        Left  — X–Y plane  (MSM, R_M)
        Right — latitude vs. east-longitude  (degrees, derived from MSM X/Y/Z)

    Parameters
    ----------
    orbit_labels : bool
        If True, annotate each DR segment on the lat-lon panel with its orbit
        number, placed at the midpoint of the segment.

    Returns the matplotlib Figure.
    """
    import os

    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), 'human_dr_labels.json')

    with open(json_path) as f:
        labels = json.load(f)

    # ------------------------------------------------------------------ #
    # Collect trajectory segments for each event
    # ------------------------------------------------------------------ #
    # Each list holds (orb, X_arr, Y_arr, Z_arr)
    loading_segs = []
    dr_segs      = []

    for orb_str, entry in labels.items():
        if not isinstance(entry, dict) or not entry.get('dr'):
            continue

        orb = int(orb_str)
        try:
            orb_df = load_bowers_data_pkl(orbit_number=orb)
        except Exception:
            continue
        if orb_df is None or orb_df.empty:
            continue

        t_obs = pd.to_datetime(orb_df['time'])
        X_orb = orb_df['ephx'].to_numpy()
        Y_orb = orb_df['ephy'].to_numpy()
        Z_orb = orb_df['ephz'].to_numpy()

        def _seg(t_start_str, t_stop_str):
            """Return (X, Y, Z) arrays for samples within [t_start, t_stop]."""
            t0   = pd.Timestamp(t_start_str)
            t1   = pd.Timestamp(t_stop_str)
            mask = (t_obs >= t0) & (t_obs <= t1)
            if mask.sum() < 2:
                # fall back to nearest single point duplicated so a dot appears
                idx  = np.argmin(np.abs((t_obs - (t0 + (t1 - t0)/2)).dt.total_seconds().to_numpy()))
                return (np.array([X_orb[idx], X_orb[idx]]),
                        np.array([Y_orb[idx], Y_orb[idx]]),
                        np.array([Z_orb[idx], Z_orb[idx]]))
            return X_orb[mask], Y_orb[mask], Z_orb[mask]

        # loading interval
        if 'loading_start' in entry and 'loading_stop' in entry:
            try:
                loading_segs.append((orb, *_seg(entry['loading_start'], entry['loading_stop'])))
            except Exception:
                pass

        # DR interval(s) — support both single-event and multi-event formats
        if 'events' in entry:
            for ev in entry['events']:
                if 'dr_start' in ev and 'dr_stop' in ev:
                    try:
                        dr_segs.append((orb, *_seg(ev['dr_start'], ev['dr_stop'])))
                    except Exception:
                        pass
        elif 'dr_start' in entry and 'dr_stop' in entry:
            try:
                dr_segs.append((orb, *_seg(entry['dr_start'], entry['dr_stop'])))
            except Exception:
                pass

    def _seg_to_latlon(X, Y, Z):
        r   = np.sqrt(X**2 + Y**2 + Z**2)
        lat = np.degrees(np.arcsin(np.clip(Z / r, -1, 1)))
        lon = np.degrees(np.arctan2(Y, X)) % 360
        return lon, lat

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    else:
        axes = fig.subplots(1, 2)

    ax_xy, ax_ll = axes

    # ---------- X–Y panel ----------
    theta = np.linspace(0, 2 * np.pi, 300)
    ax_xy.fill(np.cos(theta), np.sin(theta), color='saddlebrown', alpha=0.4, zorder=1)
    ax_xy.plot(np.cos(theta), np.sin(theta), color='saddlebrown', lw=1.0, zorder=1)

    load_label_done = dr_label_done = False
    for orb, X, Y, Z in loading_segs:
        ax_xy.plot(Y, X, color='limegreen', lw=2.0, alpha=0.8, zorder=3,
                   label='Loading' if not load_label_done else '_')
        load_label_done = True
    for orb, X, Y, Z in dr_segs:
        ax_xy.plot(Y, X, color='gold', lw=2.5, alpha=0.9, zorder=4,
                   label='DR' if not dr_label_done else '_',
                   solid_capstyle='round')
        dr_label_done = True

    ax_xy.set_xlabel('Y$_{MSM}$ (R$_M$)')
    ax_xy.set_ylabel('X$_{MSM}$ (R$_M$)')
    ax_xy.set_aspect('equal')
    ax_xy.axhline(0, color='k', lw=0.4, alpha=0.4)
    ax_xy.axvline(0, color='k', lw=0.4, alpha=0.4)
    ax_xy.legend(fontsize=9)
    ax_xy.grid(True, alpha=0.25)
    ax_xy.set_title('X–Y plane (MSM)')

    # ---------- Lat–Lon panel ----------
    ax_ll.set_facecolor('#e8f4f8')
    ax_ll.axhline(0,   color='k', lw=0.5, alpha=0.4)
    ax_ll.axvline(180, color='k', lw=0.5, alpha=0.3, ls='--')

    load_label_done = dr_label_done = False
    for orb, X, Y, Z in loading_segs:
        lon, lat = _seg_to_latlon(X, Y, Z)
        ax_ll.plot(lon, lat, color='limegreen', lw=2.0, alpha=0.8, zorder=3,
                   label='Loading' if not load_label_done else '_')
        load_label_done = True
    for orb, X, Y, Z in dr_segs:
        lon, lat = _seg_to_latlon(X, Y, Z)
        ax_ll.plot(lon, lat, color='gold', lw=2.5, alpha=0.9, zorder=4,
                   label='DR' if not dr_label_done else '_',
                   solid_capstyle='round')
        dr_label_done = True
        if orbit_labels:
            mid = len(lon) // 2
            ax_ll.text(lon[mid], lat[mid], str(orb),
                       fontsize=5, ha='center', va='bottom',
                       color='darkorange', zorder=5,
                       clip_on=True)

    lon_min, lon_max = 90, 270
    lat_min, lat_max = -60, 60
    ax_ll.set_xlabel('East longitude (°)')
    ax_ll.set_ylabel('MLat (°)')
    ax_ll.set_xticks([t for t in range(0, 361, 45) if lon_min <= t <= lon_max])
    ax_ll.set_yticks([t for t in range(-90, 91, 30) if lat_min <= t <= lat_max])
    ax_ll.legend(fontsize=9)
    ax_ll.grid(True, alpha=0.25)
    ax_ll.set_title('Latitude–Longitude (MSM)')
    # set limits last so nothing above can re-trigger autoscaling
    ax_ll.set_xlim(lon_min, lon_max)
    ax_ll.set_ylim(lat_min, lat_max)

    n_load = len(loading_segs)
    n_dr_  = len(dr_segs)
    fig.suptitle(
        f'MESSENGER DR event locations  '
        f'({n_dr_} DRs,  {n_load} loading intervals)',
        fontsize=11,
    )
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(json_path)), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'event_locations.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")

    return fig

def plot_dr_bphi_locations(json_path=None, clim=(-10, 10), orbit_labels=False):
    """
    Read human_dr_labels.json (or *json_path*) and plot each DR trajectory
    segment on a lat–lon map, coloured by ΔB_phi (azimuthal FAC component).

    The colour axis is symmetric about zero (default ±20 nT) using the
    RdBu_r colormap so negative (blue) and positive (red) are intuitive.

    Returns the matplotlib Figure.
    """
    from matplotlib.collections import LineCollection
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), 'human_dr_labels.json')

    with open(json_path) as f:
        labels = json.load(f)

    cmap  = cm.RdBu_r
    norm  = mcolors.Normalize(vmin=clim[0], vmax=clim[1])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor('#e8f4f8')
    ax.axhline(0,   color='k', lw=0.5, alpha=0.4)
    ax.axvline(180, color='k', lw=0.5, alpha=0.3, ls='--')

    for orb_str, entry in labels.items():
        if not isinstance(entry, dict) or not entry.get('dr'):
            continue

        orb = int(orb_str)

        if orb not in nice_examples:
            print("HARDCODED EXAMPLE MODE, NOT PLOTTING THIS ORBIT")
            continue

        try:
            orb_df = load_bowers_data_pkl(orbit_number=orb)
            orb_df = filter_orbit_segment(orb_df)
        except Exception:
            continue
        if orb_df is None or orb_df.empty:
            continue

        _, _, dbg = _apply_dr_filter(orb_df)
        t_obs  = dbg['t_obs']
        X_orb  = orb_df['ephx'].to_numpy()
        Y_orb  = orb_df['ephy'].to_numpy()
        Z_orb  = orb_df['ephz'].to_numpy()
        B_phi  = dbg['B_phi']

        # 60 s rolling mean of observed Bx over the full orbit
        t_s      = (t_obs - t_obs.iloc[0]).dt.total_seconds().to_numpy()
        dt_s     = np.median(np.diff(t_s)) if len(t_s) > 1 else 1.0
        win      = max(1, int(round(60.0 / dt_s)))
        Bx_smooth = (pd.Series(orb_df['magx'].to_numpy())
                     .rolling(win, center=True, min_periods=1).mean()
                     .to_numpy())

        # DR intervals — support single and multi-event formats
        dr_intervals = []
        if 'events' in entry:
            for ev in entry['events']:
                if 'dr_start' in ev and 'dr_stop' in ev:
                    dr_intervals.append((pd.Timestamp(ev['dr_start']),
                                         pd.Timestamp(ev['dr_stop'])))
        elif 'dr_start' in entry and 'dr_stop' in entry:
            dr_intervals.append((pd.Timestamp(entry['dr_start']),
                                 pd.Timestamp(entry['dr_stop'])))

        for t0, t1 in dr_intervals:
            mask = (t_obs >= t0) & (t_obs <= t1)
            if mask.sum() < 2:
                continue

            Xs = X_orb[mask]; Ys = Y_orb[mask]; Zs = Z_orb[mask]
            bp = B_phi[mask]

            r   = np.sqrt(Xs**2 + Ys**2 + Zs**2)
            lat = np.degrees(np.arcsin(np.clip(Zs / r, -1, 1)))
            lon = np.degrees(np.arctan2(Ys, Xs)) % 360

            # build line segments for LineCollection
            pts  = np.column_stack([lon, lat])
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            vals = 0.5 * (bp[:-1] + bp[1:])   # colour by midpoint value

            lc = LineCollection(segs, cmap=cmap, norm=norm, lw=3.0,
                                 alpha=0.9, zorder=3)
            lc.set_array(vals)
            ax.add_collection(lc)

            if orbit_labels:
                mid = len(lon) // 2
                ax.text(lon[mid], lat[mid], str(orb),
                        fontsize=5, ha='center', va='bottom',
                        color='k', zorder=5, clip_on=True)

            # perpendicular tick at each 60 s-smoothed Bx=0 crossing
            bx_seg = Bx_smooth[mask]
            signs  = np.sign(bx_seg)
            signs[signs == 0] = 1          # treat exact zero as positive
            crossings = np.where(np.diff(signs) != 0)[0]
            for ci in crossings:
                denom = bx_seg[ci] - bx_seg[ci + 1]
                frac  = bx_seg[ci] / denom if denom != 0 else 0.5
                cx_lon = lon[ci] + frac * (lon[ci + 1] - lon[ci])
                cx_lat = lat[ci] + frac * (lat[ci + 1] - lat[ci])
                # local trajectory direction → perpendicular (rotate 90°)
                i0 = max(ci - 1, 0);  i1 = min(ci + 2, len(lon) - 1)
                dlon = lon[i1] - lon[i0]
                dlat = lat[i1] - lat[i0]
                mag  = np.sqrt(dlon**2 + dlat**2)
                if mag > 0:
                    pdlon, pdlat = -dlat / mag, dlon / mag
                else:
                    pdlon, pdlat = 0.0, 1.0
                tick = 1.5   # half-length in degrees
                ax.plot([cx_lon - pdlon * tick, cx_lon + pdlon * tick],
                        [cx_lat - pdlat * tick, cx_lat + pdlat * tick],
                        color='black', lw=0.5, zorder=6,
                        solid_capstyle='round')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(r'$\Delta B_\phi$ (nT)', fontsize=10)

    lon_min, lon_max = 90, 270
    lat_min, lat_max = -30, 30
    ax.set_xlabel('East longitude (°)')
    ax.set_ylabel('MLat (°)')
    ax.set_xticks([t for t in range(0, 361, 45) if lon_min <= t <= lon_max])
    ax.set_yticks([t for t in range(-90, 91, 30) if lat_min <= t <= lat_max])
    ax.grid(True, alpha=0.25)
    ax.set_title(r'DR locations coloured by $\Delta B_\phi$  (MSM lat–lon)')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_aspect(1)

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(json_path)), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'dr_bphi_locations.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")

    return fig


# Fips data: https://pds-ppi.igpp.ucla.edu/collection/urn:nasa:pds:mess-epps-fips-derived:data-espec
# FIPS energy bin centres (keV), 64 channels, ascending order.
# Sourced from the DEP1 variable in the LIGHT64 CDF files (instrument constant).
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

# Column order in the FIPS ESPEC TAB file (after INDEX and MET)
_FIPS_TAB_SPECIES = ['H+', 'He++', 'He+', 'Na-group', 'O-group']


def load_fips_espec_tab(path):
    """
    Load a MESSENGER FIPS ESPEC DDR TAB file (PDS product).

    Contains differential flux [1/(cm² s keV sr)] for H+, He++, He+,
    Na-group, and O-group in 64 energy channels.

    MET is converted to UTC via SPICE (requires MESSENGER kernels);
    falls back to a linear approximation using the known epoch in the label.

    Returns
    -------
    dict with keys:
        't'             : datetime64[ns] (N,)
        '<sp>_flux'     : float32 (N, 64), NaN where fill (≤0)
        '<sp>_energy'   : float32 (64,)   energy centres (keV, ascending)
      for sp in 'H+', 'He++', 'He+', 'Na-group', 'O-group'
    """
    raw = pd.read_csv(path, skiprows=4, header=None, sep=r'\s+',
                      engine='python')
    # cols: 0=INDEX, 1=MET, then 5 groups of 64
    met = raw.iloc[:, 1].to_numpy(dtype='float64')

    # MET → UTC via SPICE; linear fallback if kernels unavailable
    try:
        load_messenger_kernels()
        et_arr  = np.array([spice.sct2e(-236, m) for m in met])
        utc_arr = np.array([spice.et2utc(e, 'ISOC', 3) for e in et_arr],
                           dtype='datetime64[ns]')
    except Exception:
        # label anchor: MET 253562681 = 2012-08-16T00:00:14.973Z
        _met0 = 253562681.0
        _t0   = np.datetime64('2012-08-16T00:00:14.973', 'ms').astype('int64')
        delta = ((met - _met0) * 1e9).astype('int64')
        utc_arr = (_t0 + delta).astype('datetime64[ns]')

    result = {'t': utc_arr}
    for i, sp in enumerate(_FIPS_TAB_SPECIES):
        col0 = 2 + i * 64
        flux = raw.iloc[:, col0:col0 + 64].to_numpy(dtype='float32')
        flux[flux <= 0] = np.nan
        result[f'{sp}_flux']   = flux
        result[f'{sp}_energy'] = _FIPS_ENERGY_KEV.copy()

    return result

def _fips_time_edges(t_ns):
    """(N+1,) datetime64[ns] bin edges from (N,) int64 nanosecond centres."""
    edges = np.empty(len(t_ns) + 1, dtype='int64')
    edges[1:-1] = (t_ns[:-1] + t_ns[1:]) // 2
    dt = int(np.median(np.diff(t_ns))) if len(t_ns) > 1 else int(60e9)
    edges[0]  = t_ns[0]  - dt // 2
    edges[-1] = t_ns[-1] + dt // 2
    return edges.astype('datetime64[ns]')


def _fips_bin_edges(centres):
    """(N+1,) log-spaced bin edges from (N,) bin centres."""
    log_c = np.log10(centres)
    dlog  = np.diff(log_c)
    edges = np.empty(len(centres) + 1)
    edges[1:-1] = 10 ** (0.5 * (log_c[:-1] + log_c[1:]))
    edges[0]    = 10 ** (log_c[0]  - 0.5 * dlog[0])
    edges[-1]   = 10 ** (log_c[-1] + 0.5 * dlog[-1])
    return edges


def plot_fips_espec_spectrogram(path, species=None, save=True):
    """
    Plot FIPS differential flux spectrograms from a PDS ESPEC TAB file.

    Parameters
    ----------
    path    : str   path to a FIPS_ESPEC_*_DDR_*.TAB file
    species : list  subset of _FIPS_TAB_SPECIES to plot; default ['H+']
    save    : bool

    Returns the matplotlib Figure.
    """
    if species is None:
        species = ['H+']

    data    = load_fips_espec_tab(path)
    t_ns    = data['t'].astype('int64')
    t_edges = _fips_time_edges(t_ns)

    cmap = plt.cm.nipy_spectral.copy()
    #cmap.set_bad('black')
    #cmap.set_under('black')

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

        T, E = np.meshgrid(t_edges, e_edges)
        pos  = flux[flux > 0]
        vmin = 1e6 #np.nanpercentile(pos,  5) if pos.size else 1.0
        vmax = 1e9 #np.nanpercentile(pos, 99) if pos.size else 1e6

        pcm = ax.pcolormesh(T, E, flux.T,
                            cmap=cmap,
                            norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                            shading='flat')
        ax.set_yscale('log')
        ax.set_ylabel('Energy (keV)', fontsize=9)
        ax.set_ylim(e_edges[0], e_edges[-1])
        cb = fig.colorbar(pcm, ax=ax, pad=0.005, fraction=0.015)
        cb.set_label(r'Flux (cm$^{-2}$ s$^{-1}$ keV$^{-1}$ sr$^{-1}$)', fontsize=6)
        ax.text(0.005, 0.96, sp, transform=ax.transAxes,
                fontsize=10, va='top', fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc='k', alpha=0.45))
        ax.grid(True, alpha=0.15, color='white', lw=0.4)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].set_xlabel('UTC 2012-08-16', fontsize=9)

    date_tag = os.path.basename(path).split('_')[2]
    fig.suptitle(f'MESSENGER FIPS ESPEC — {date_tag}', fontsize=11)
    plt.tight_layout()

    if save:
        os.makedirs('figures', exist_ok=True)
        sp_tag = '_'.join(s.replace('+', 'p').replace('-', '') for s in species)
        out = os.path.join('figures', f'fips_espec_{date_tag}_{sp_tag}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved → {out}")

    return fig


_FIPS_ESPEC_DIR = os.path.join(os.path.dirname(__file__), 'FIPS')
_FIPS_ESPEC_BASE = 'https://pds-ppi.igpp.ucla.edu/data/mess-epps-fips-derived/data-espec/'


def _fips_espec_path_for_date(date):
    """Return local path for the FIPS ESPEC TAB file covering *date*.

    *date* can be a datetime, Timestamp, or datetime64.
    Downloads from PDS FTP if the file is not present.
    """
    import urllib.request
    dt  = pd.Timestamp(date)
    doy = dt.day_of_year
    tag = f'{dt.year}{doy:03d}'
    fname = f'FIPS_ESPEC_{tag}_DDR_V03.TAB'
    local = os.path.join(_FIPS_ESPEC_DIR, fname)

    if not os.path.exists(local):
        url = _FIPS_ESPEC_BASE + fname
        print(f'Downloading {fname} …', end=' ', flush=True)
        os.makedirs(_FIPS_ESPEC_DIR, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, local)
            print('done.')
        except Exception as e:
            if os.path.exists(local):
                os.remove(local)
            raise FileNotFoundError(
                f"Could not download {fname} from PDS FTP.\n"
                f"URL tried: {url}\nError: {e}"
            ) from e

    return local

_FIPS_COLLECTION_PAGE = ('https://pds-ppi.igpp.ucla.edu/collection/'
                         'urn:nasa:pds:mess-epps-fips-derived:data-espec')
_FIPS_ITEM_BASE       = 'https://pds-ppi.igpp.ucla.edu/item/'


def _fips_tab_url_from_item_page(yyyydoy):
    """
    Fetch the PPI item page for the given day-of-year tag and return the
    direct HTTPS URL of the TAB file, or None if not found.
    """
    import urllib.request
    import re as _re

    lid    = (f'urn:nasa:pds:mess-epps-fips-derived:data-espec:'
              f'fips_espec_{yyyydoy}_ddr_v03')
    # try version suffixes 1.2, 1.1, 1.0 in order
    for ver in ('1.2', '1.1', '1.0'):
        item_url = f'{_FIPS_ITEM_BASE}{lid}::{ver}'
        try:
            req = urllib.request.Request(item_url,
                                         headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                html = resp.read().decode('utf-8', errors='replace')
            # look for href containing the TAB filename
            matches = _re.findall(
                r'href=["\']([^"\']*FIPS_ESPEC[^"\']*\.TAB)["\']', html, _re.I)
            for m in matches:
                return m if m.startswith('http') else 'https://pds-ppi.igpp.ucla.edu' + m
        except Exception:
            continue
    return None


def download_all_fips_espec(overwrite=False):
    """
    Download every FIPS ESPEC TAB file from the PDS PPI collection into FIPS/.

    Fetches the collection index page to discover all product day-of-year tags,
    then downloads each TAB file via the individual item pages.
    Skips files already present unless *overwrite=True*.
    """
    import urllib.request
    import re as _re

    os.makedirs(_FIPS_ESPEC_DIR, exist_ok=True)

    # --- discover all YYYYDOY tags from the collection page ---
    print(f'Fetching collection index …')
    req = urllib.request.Request(_FIPS_COLLECTION_PAGE,
                                 headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode('utf-8', errors='replace')

    tags = sorted(set(_re.findall(r'fips_espec_(\d{7})_ddr_v\d+', html, _re.I)))
    if not tags:
        raise RuntimeError(
            'No product tags found on the collection page.\n'
            f'Check that {_FIPS_COLLECTION_PAGE} is accessible and lists products.'
        )
    print(f'Found {len(tags)} products.')

    for i, tag in enumerate(tags, 1):
        fname = f'FIPS_ESPEC_{tag.upper()}_DDR_V03.TAB'
        local = os.path.join(_FIPS_ESPEC_DIR, fname)
        if os.path.exists(local) and not overwrite:
            print(f'[{i}/{len(tags)}] {fname} — already present, skipping.')
            continue

        tab_url = _fips_tab_url_from_item_page(tag)
        if tab_url is None:
            print(f'[{i}/{len(tags)}] {fname} — could not resolve download URL, skipping.')
            continue

        print(f'[{i}/{len(tags)}] Downloading {fname} …', end=' ', flush=True)
        try:
            urllib.request.urlretrieve(tab_url, local)
            print('done.')
        except Exception as e:
            print(f'FAILED: {e}')
            if os.path.exists(local):
                os.remove(local)

    print('Download complete.')

def plot_fips_for_orbit(orb, species=None, save=True):
    """
    Plot FIPS ESPEC spectrograms for the date of *orb*, downloading the
    data file from PDS if not already present in FIPS/.

    Parameters
    ----------
    orb     : int   orbit number
    species : list  species to plot, default ['H+']
    save    : bool
    """
    orb_df = load_bowers_data_pkl(orbit_number=orb)
    date   = pd.to_datetime(orb_df['time'].iloc[0])
    path   = _fips_espec_path_for_date(date)
    return plot_fips_espec_spectrogram(path, species=species, save=save)


# --- Usage ---

trange = ['2015-04-05/01:54:00', '2015-04-05/02:02:00']

#traj = download_messenger_trajectory(
#    trange=trange,
#    coord='mso',
#    dt_sec=60
#)

#trange = ['2012-01-02/17:02:22', '2012-01-02/23:02:22']
#data = load_bowers_data_pkl(trange=trange)
#traj = bowers_traj(trange=trange)
#print(data)
#plot_messenger_trajectory(traj)

#plot_kt17_streamplot(xlim=(-4, 2), zlim=(-3, 3), y0=0.0, nx=50, nz=50, Rsun=0.25)
#plot_kt17_streamplot(xlim=(-4, 2), zlim=(-3, 3), y0=0.0, nx=50, nz=50, Rsun=0.3, DistIndex = 90)
#plot_kt17_streamplot(xlim=(-4, 2), zlim=(-3, 3), y0=0.0, nx=50, nz=50, Rsun=0.5)
#plot_mag_timeseries(trange, show_model=True)
#plot_field_aligned_timeseries(trange)
#batch_plot_orbits(3400,3500, highlight = False, show_loading_times = True)
#traj = bowers_traj(trange=trange)
#plot_messenger_trajectory(traj)
#print(data)

# Show examples
#event_filtering_toolkit_v1(human_labels=True)
#nice_examples = [1638, 3936, 3940,1109,1118, 3964, 3946, 2687, 2909, 3451, 3206, 3458, 3485, 3158,]
nice_examples = [3158,3936,3940,2909,3946,3451,3964,3458,3206,1109,2687,1638,1118,3485,]

# Hand-label examples
#event_filtering_toolkit_v2(3653,3752) # continue from 3195

# Summarize locations of selected events
#plot_event_locations(orbit_labels=True)

# Plot their azimuthal field
#plot_dr_bphi_locations(clim=(-10, 10))

# Plot chosen orbit examples
#for orbit in nice_examples:
#    #orbit = 1638
#    fig = plt.figure(figsize=(10, 5))               
#    sf  = fig.subfigures(1, 1)                      
#    _plot_orbit_into_subfig(sf, orb=orbit, label=str(orbit),ephemeris_labels=True,human_labels=_human_labels_for_orbit(orbit),ephemeris_coords='latlon',bx_zero_line=True,)                                   
#    plt.tight_layout()
#    fig.savefig(f'figures/orbit_{orbit}.png', dpi=150, bbox_inches='tight')
#    #plt.show()
#    plt.close()

# Refine filters
#event_filtering_toolkit_v3()
#best = optimize_filter_v3()

# Perform search on orbits
#run_automated_detection(2864,2975, force=True,save_plots=True)

# Show fips data from Fraenz
#plot_fips_spectrogram('cdf/MES_FIPS_PHA_LIGHT64_20140210.cdf') 

# Show Raines FIPS data
#plot_fips_espec_spectrogram('FIPS/FIPS_ESPEC_2012229_DDR_V03.TAB', species=['H+']) 

# Download FIPS
#download_all_fips_espec()