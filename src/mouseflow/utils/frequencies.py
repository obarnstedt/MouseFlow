#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pyTreadMouse functions to analyse camera data

Created on Wed May  8 14:31:51 2019
@author: Oliver Barnstedt
"""


import os.path
import glob
import pandas as pd
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
import math
import numpy as np
from scipy.stats.mstats import zscore
from scipy import signal, optimize

def freq_analysis(x, f_s=75, M=128):
    freqs, times, Sx = signal.spectrogram(x, fs=f_s, window='hanning',
                                          nperseg=M, noverlap=M - 50,
                                          detrend=False, scaling='spectrum')
    max_freq = pd.DataFrame(data={'Time': times, 'maxf': np.argmax(Sx, axis=0)})
    org_timing = pd.DataFrame(data={'Time':np.arange(0, len(x)/f_s, 1/f_s)})
    joint = pd.merge_asof(org_timing, max_freq, on='Time').interpolate(method='linear')

    return joint.maxf
    #
    # f, ax = plt.subplots(4, 1, figsize=(4.8, 2.4), sharex=True)
    # ax[0].plot(timing, x)
    # ax[1].pcolormesh(times, freqs, 10 * np.log10(Sx), cmap='viridis')
    # ax[1].set_ylabel('Frequency [Hz]')
    # ax[1].set_xlabel('Time [s]');
    # ax[2].plot(times, np.argmax(Sx, axis=0))
    # ax[3].plot(timing, motion_frontpaw)

def freq_analysis2(x, fps, rollwin=75, min_periods=50, conf=0.5):
    if 'x' in x:
        x.loc[x.likelihood < conf] = np.nan
        y = x.x
    else:
        y = x
    xinterp = y.interpolate(method='linear')
    xz = (xinterp - xinterp.min()) / (xinterp - xinterp.min()).max()
    print('Fitting sine...')
    w = xz.rolling(int(rollwin), center=True, min_periods=min_periods).apply(fit_sin_w)
    error = xz.rolling(int(rollwin), center=True, min_periods=min_periods).apply(fit_sin_error)
    w_clean = w.copy()
    w_clean[w_clean.diff().abs()>(.75/fps)] = np.nan
    w_clean[error>(fps/100)] = np.nan
    w_clean[w_clean<0] = np.nan
    w_clean = w_clean.interpolate(method='polynomial', order=3, limit=int(fps*2))
    w_smooth = w_clean.rolling(int(rollwin/3), center=True, min_periods=int(min_periods/3)).mean(window='gaussian')
    f_smooth = w_smooth/(2.*np.pi)*fps
    return f_smooth


def dlc_pointphasecorr(point1, point2, body_conf_thresh=.5, body_interpolation_limit=300, correlation_window=75):
    point1, point2 = point1.values, point2.values
    confident = [(point1[:, 2] > body_conf_thresh), (point2[:, 2] > body_conf_thresh)]
    xpoints = pd.DataFrame(data={'x1':point1[:, 0]*confident[0], 'x2':point2[:, 0]*confident[1]})
    xpoints[xpoints==0] = np.nan
    xpoints_angle = (((xpoints.apply(lambda x: (x - x.quantile(.01)) / (x - x.quantile(.01)).quantile(.99)))*2*np.pi)-np.pi).interpolate(method='polynomial', order=3, limit=body_interpolation_limit)
    running_corr = xpoints_angle.x1.rolling(correlation_window).corr(xpoints_angle.x2)
    return pd.Series(running_corr)


def hilbert_peaks(x, fps, fc=10, butterN=5, peakprom=1.5):
    env = np.abs(signal.hilbert(x))
    # Low-Pass Butter filter:
    w = fc / (fps / 2)  # Normalize the frequency to SamplingFrequency
    b, a = signal.butter(butterN, w, 'low')
    mouth_envfilt = pd.Series(signal.filtfilt(b, a, env)).shift(12)
    peaks = signal.find_peaks(mouth_envfilt, prominence=peakprom)[0]
    peakbool = np.zeros(len(x))
    peakbool[peaks] = 1
    return mouth_envfilt, peakbool


def dlc_phasecorrX(fr, fl, br, bl):
    corr_x = pd.Series(np.nanmean([dlc_pointphasecorr(fr, bl).values, dlc_pointphasecorr(fl, br).values], axis=0))
    corr_eq = pd.Series(np.nanmean([dlc_pointphasecorr(fr, fl).values, dlc_pointphasecorr(bl, br).values], axis=0))
    return(corr_x, corr_eq)




# sine fitting written by unsym: https://stackoverflow.com/a/42322656
def fit_sin(yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(range(len(yy)))
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    try:
        popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        f = w / (2. * np.pi)
        fitfunc = lambda t: A * np.sin(w * t + p) + c
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f, "fitfunc": fitfunc,
                "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}
    except:
        return np.nan


def fit_sin_A(yy):  # amplitude
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(range(len(yy)))
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    try:
        popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        return A
    except:
        return np.nan


def fit_sin_w(yy):  # omega
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(range(len(yy)))
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c
    try:
        popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        return w
    except:
        return np.nan


def fit_sin_p(yy):  # phase
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(range(len(yy)))
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    try:
        popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        return p
    except:
        return np.nan


def fit_sin_c(yy):  # offset
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(range(len(yy)))
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    try:
        popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        return c
    except:
        return np.nan


def fit_sin_error(yy):  # mean SD error
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(range(len(yy)))
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])
    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c
    try:
        popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        return np.mean(np.sqrt(np.diag(pcov)))
    except:
        return np.nan

def fit_sin_werror(yy):  # omega and mean SD error
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(range(len(yy)))
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])
    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c
    try:
        popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        return pd.Series(w, np.mean(np.sqrt(np.diag(pcov))))
    except:
        return np.nan