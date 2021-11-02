import numpy as np
from lightkurve import LightCurve
import math 

num_bins = 2048
num_local_bins = 256
bin_width_factor = 0.16


def almost_same(x, y):
    delta = 1e-5
    return math.fabs(x - y) <= delta


def choose_from_center(time, val_center, val_width):
    """
    val_center, val_width should have the same units (days in general)
    """
    half_width = val_width / 2.0
    hi = np.searchsorted(time, val_center + half_width)
    lo = np.searchsorted(time, val_center - half_width)
    lo, hi = int(lo), int(hi)
    return lo, hi


def flatten_interp_transits(all_time, all_flux, period, t0, duration):
    """
    all_time, all_flux: real time of the lightcurve, before folding.\n
    duration: in Days
    return the flattened time, flux, not folding
    """
    fold_time = all_time % period
    t0 %= period
    half_duration = duration / 2.0
    mask = np.logical_and(fold_time <= t0 + half_duration,
                          fold_time >= t0 - half_duration)

    tce_time, tce = all_time[mask], all_flux[mask]

    lc = LightCurve(time=all_time, flux=all_flux).flatten(
        window_length=701, polyorder=2, break_tolerance=40, sigma=3
    )

    all_time, flat_flux = lc.time, lc.flux

    # keep the transit original
    flat_flux[mask] = tce

    # sigma clip the outliers in the transit
    # tce_time, tce = sigma_clip(tce_time, tce, sigma=3.0)

    try:
        return all_time.value, flat_flux.value
    except:
        return all_time, flat_flux


def remove_points_other_tce(all_time,
                            all_flux,
                            cur_period,
                            period_list,
                            t0_list,
                            duration_list):
    """
    all_time, all_flux: real time, not after folding \n
    return: time, flux after removing the other periods points
    """
    for period, t0, duration in zip(period_list, t0_list, duration_list):
        # don't remove the current period
        if almost_same(cur_period, period):
            continue
        fold_time = all_time % period
        t0 %= period
        half_duration = duration / 2.0
        while (t0 + half_duration) <= period:
            mask = np.logical_or(all_time > t0 + half_duration,
                                 all_time < t0 - half_duration)
            all_time = all_time[mask]
            all_flux = all_flux[mask]
            t0 += period

    return all_time, all_flux


def process_global(time, flux, period, t0, duration):
    """
    time, flux: after removing the other period values\n
    duration: in DAYs \n
    return: binned flux
    """
    t0 %= period

    time, flux = flatten_interp_transits(time, flux, period, t0, duration)

    time, flux = fold(time, flux, period, t0)

    bin_width = None
    # print('bin width', bin_width)
    # if period < 1:
    #     bin_width *= 40
    # elif period < 10:
    #     bin_width *= 20
    # elif period < 100:
    #     bin_width *= 10
    binned_flux = median_bin(
        time, flux,
        num_bins=num_bins
        # duration=duration
    )
    return binned_flux


def process_local(time, flux, period, t0, duration, center_ratio=7):
    # print('center ratio', center_ratio)
    """
    time, flux: after removing the other period values\n
    duration: in DAYs \n
    return: binned flux
    """
    t0 %= period

    time, flux = flatten_interp_transits(
        time, flux, period, t0, duration
    )

    time, flux = fold(time, flux, period, t0)

    lo, hi = choose_from_center(
        time, period / 2.0, center_ratio * duration
    )

    time, flux = time[lo:hi], flux[lo:hi]

    # experimental
    bin_width = bin_width_factor

    # if duration < 5:
    #     bin_width *= 4
    # elif duration < 10:
    #     bin_width *= 2

    binned_flux = median_bin(
        time, flux,
        num_bins=num_local_bins,
        bin_width_factor=bin_width
    )

    return binned_flux


def median_bin(x, y, num_bins, bin_width=None, normalize=True, bin_width_factor=1.0):
    """
    assume x is sorted 
    bin_width is the scale factor of (x_max-x_min)/num_bins, not the absolute value
    """
    x = x.ravel()
    y = y.ravel()
    x_min, x_max = x[0], x[-1]
    default_bin_width = (x_max - x_min) / num_bins
    bin_width = default_bin_width if bin_width is None else bin_width
    bin_width /= bin_width_factor

    bin_spacing = (x_max - x_min) / num_bins

    m = np.median(y)
    res = np.repeat(m, num_bins)
    bin_lo, bin_hi = x_min, x_min + bin_width

    for i in range(num_bins):
        indices = np.argwhere(
            (x < bin_hi) & (x >= bin_lo)).ravel()

        if len(indices) > 0:
            res[i] = np.median(y[indices])

        bin_lo += bin_spacing
        bin_hi += bin_spacing

    if normalize:
        res -= np.median(res)
        res /= np.max(np.abs(res))

    return res


def fold(time, flux, period, t0=None):
    half_period = period / 2.0
    if t0 is None:
        t0 = half_period

    t0 %= period
    time = (time + half_period - t0) % period
    indices = np.argsort(time)
    time, flux = time[indices], flux[indices]

    return time, flux
