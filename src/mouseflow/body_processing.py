#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pyTreadMouse functions to analyse camera data

Created on Wed May  8 14:31:51 2019
@author: Oliver Barnstedt
"""
from typing import NamedTuple

import pandas as pd
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats.mstats import zscore
from scipy import signal, optimize


plt.interactive(False)


# Calculating motion energy

class CylinderMotionResult(NamedTuple):
    raw: pd.Series
    smoothed: pd.Series

def cylinder_motion(videopath, mask, smooth_window=25, videoslice=[]):
    print("Calculating cylinder motion...")
    facemp4 = cv2.VideoCapture(videopath)
    if videoslice:
        print("Processing slice from {} to {}...".format(videoslice[0], videoslice[-1]))
        facemp4.set(1, videoslice[0])
        framelength = len(videoslice)
    else:
        framelength = int(facemp4.get(7))
    ret, current_frame = facemp4.read()
    previous_frame = current_frame
    frame_diff = np.zeros([1])
    frame_diff_current = np.zeros([1])

    with tqdm(total=framelength) as pbar:
        while facemp4.isOpened():
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            frame_diff_current[0] = (np.sum(
                cv2.absdiff(current_frame_gray * mask, previous_frame_gray * mask)) / np.sum(mask))
            frame_diff = np.vstack((frame_diff, frame_diff_current))
            pbar.update(1)
            previous_frame = current_frame.copy()
            ret, current_frame = facemp4.read()
            if current_frame is None or (videoslice and len(frame_diff) > len(videoslice)-1):
                break
    facemp4.release()

    cyl = pd.Series(zscore(frame_diff.flatten()))
    cyl_smooth = pd.Series(zscore(pd.Series(smooth(frame_diff.flatten(), window_len=smooth_window)).shift(periods=-int(smooth_window/2))[:-int(smooth_window/2)]))
    return CylinderMotionResult(cyl, cyl_smooth)


class PointMotionResult(NamedTuple):
    raw_distance: pd.Series
    proc_distance: pd.Series
    angles: pd.Series

def dlc_pointmotion(dlc, body_conf_thresh=.5, body_smooth_window=25, body_interpolation_limit=150) -> PointMotionResult:
    x = dlc['x']
    y = dlc['y']
    likelihood = dlc['likelihood']
    confident = likelihood > body_conf_thresh
    x_conf = x*confident
    y_conf = y*confident
    xy = pd.DataFrame(np.array((x_conf, y_conf)).T, columns=['x', 'y'])
    xy[xy == 0] = np.nan
    xydiff_raw = xy.diff()
    xydist_raw = pd.Series(np.linalg.norm(xydiff_raw, axis=1))
    deg = pd.Series(np.arctan2(xydiff_raw.y, xydiff_raw.x)) + np.pi
    deg.loc[deg>np.pi] = deg.loc[deg>np.pi] - 2*np.pi
    xydist_interp = xydist_raw.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
    deg_interp = deg.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
    xydist_z_raw = ((xydist_interp - xydist_interp.mean()) / xydist_interp.std(ddof=0))[:len(x)]
    xydist_smooth = pd.Series(smooth(xydist_interp, window_len=body_smooth_window)).shift(periods=-int(body_smooth_window/2))
    xydist_z = ((xydist_smooth - xydist_smooth.mean()) / xydist_smooth.std(ddof=0))[:len(x)]

    return PointMotionResult(pd.Series(xydist_raw), pd.Series(xydist_z), pd.Series(deg_interp))

    
class AngleResult(NamedTuple):
    angle3: pd.Series
    slope: pd.Series

def dlc_angle(point1, point2, point3, body_conf_thresh=.5, body_smooth_window=25, body_interpolation_limit=150) -> AngleResult:
    point1, point2, point3 = point1.values, point2.values, point3.values
    confident = [(point1[:,2] > body_conf_thresh), (point2[:,2] > body_conf_thresh), (point3[:,2] > body_conf_thresh)]
    point1_conf = pd.DataFrame(np.array((point1[:,0]*confident[0], point1[:,1]*confident[0])).T, columns = ['x', 'y'])
    point1_conf[point1_conf == 0] = np.nan
    point2_conf = pd.DataFrame(np.array((point2[:,0]*confident[1], point2[:,1]*confident[1])).T, columns = ['x', 'y'])
    point2_conf[point2_conf == 0] = np.nan
    point3_conf = pd.DataFrame(np.array((point3[:,0]*confident[2], point3[:,1]*confident[2])).T, columns = ['x', 'y'])
    point3_conf[point3_conf == 0] = np.nan
    dX1 = np.array(point2_conf['x'] - point1_conf['x'], dtype=float)
    dY1 = np.array(point2_conf['y'] - point1_conf['y'], dtype=float)
    deg1 = pd.Series(np.arctan2(dY1, dX1)) + np.pi
    deg1.loc[deg1>np.pi] = deg1.loc[deg1>np.pi] - 2*np.pi
    dX2 = np.array(point3_conf['x'] - point2_conf['x'], dtype=float)
    dY2 = np.array(point3_conf['y'] - point2_conf['y'], dtype=float)
    deg2 = pd.Series(np.arctan2(dY2, dX2))  + np.pi
    deg2.loc[deg2>np.pi] = deg2.loc[deg2>np.pi] - 2*np.pi
    angle = deg2 - deg1
    deg1_interp = deg1.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
    angle_interp = angle.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
    deg1_smooth = pd.Series(smooth(deg1_interp, window_len=body_smooth_window)).shift(periods=-int(body_smooth_window/2))
    angle_smooth = pd.Series(smooth(angle_interp, window_len=body_smooth_window)).shift(periods=-int(body_smooth_window/2))
    return AngleResult(angle3=pd.Series(angle_smooth), slope=pd.Series(deg1_smooth))

def dlc_pointdistance(point1, point2, body_conf_thresh=.5, body_interpolation_limit=300):
    point1, point2 = point1.values, point2.values
    confident = [(point1[:, 2] > body_conf_thresh), (point2[:, 2] > body_conf_thresh)]
    point1_conf = pd.DataFrame(np.array((point1[:,0]*confident[0], point1[:,1]*confident[0])).T, columns = ['x', 'y'])
    point1_conf[point1_conf == 0] = np.nan
    point2_conf = pd.DataFrame(np.array((point2[:,0]*confident[1], point2[:,1]*confident[1])).T, columns = ['x', 'y'])
    point2_conf[point2_conf == 0] = np.nan
    xydiff_raw = point1_conf - point2_conf
    xydist_raw = pd.Series(np.linalg.norm(xydiff_raw, axis=1))
    xydist_interp = xydist_raw.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
    xydist_z = ((xydist_interp - xydist_interp.mean()) / xydist_interp.std(ddof=0))[:len(point1)]

    return pd.Series(xydist_raw), pd.Series(xydist_z)
