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


# preallocate empty array and assign slice by chrisaycock
def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

#    if x.ndim != 1:
#        raise ValueError, "smooth only accepts 1 dimension arrays."
#
#    if x.size < window_len:
#        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x

#    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def process_raw_data(smoothing_windows_sec, na_limit, FaceCam_FPS, interpolation_limits_frames, face_raw):
    face_raw.iloc[:, (face_raw.isnull().mean()>na_limit).values] = np.nan  

    # Interpolate missing values
    face_interp = face_raw.copy()
    face_interp[['PupilX', 'PupilY', 'PupilMotion', 'PupilDiam']] = \
            face_interp[['PupilX', 'PupilY', 'PupilMotion', 'PupilDiam']].interpolate(
                method='linear', limit=interpolation_limits_frames['pupil'])

    # Smoothen data
    smoothing_windows_frames = {x: int(k * FaceCam_FPS)
                                for (x, k) in smoothing_windows_sec.items()}
    face_smooth = face_interp.copy()
    face_smooth['PupilDiam'] = face_smooth['PupilDiam'].rolling(
            window=smoothing_windows_frames['PupilDiam'], center=True).mean()
    face_smooth[['PupilX', 'PupilY', 'PupilMotion']] = \
            face_smooth[['PupilX', 'PupilY', 'PupilMotion']].rolling(
            window=smoothing_windows_frames['PupilMotion'], center=True).mean()
    face_smooth.loc[:, face_smooth.columns.str.startswith('MotionEnergy')] = \
            face_smooth.loc[:, face_smooth.columns.str.startswith('MotionEnergy')].rolling(
            window=smoothing_windows_frames['MotionEnergy'], center=True).mean()

    # Z-scoring data
    face_zscore = face_smooth.apply(lambda a: (a - a.mean())/a.std(ddof=0))

    # adding binary data
    face_zscore['Saccades'] = pd.Series(face_smooth['PupilX'].diff().abs() > 1.5).astype(int)
    face_zscore['EyeBlink'] = pd.Series(face_interp['EyeLidDist'] < face_interp['EyeLidDist'].median()*.75).astype(int)

    # Concatenating all data types into multi-level dataframe
    face = pd.concat({'raw': face_raw, 'interpolated': face_interp, 'smooth': face_smooth, 'zscore': face_zscore}, names=['Data_type'], axis=1)

    return face