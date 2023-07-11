#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pyTreadMouse functions to analyse camera data

Created on Wed May  8 14:31:51 2019
@author: Oliver Barnstedt
"""
from typing import NamedTuple

import glob
import math
import os.path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from mouseflow.utils.generic import smooth

plt.interactive(False)


class PupilResult(NamedTuple):
    pupil_x_raw: pd.Series
    pupil_y_raw: pd.Series
    pupil_xydist_raw: pd.Series
    pupil_diam_raw: pd.Series

def pupilextraction(pupil_markers_xy_confident):
    pupil_circle = np.zeros(shape=(len(pupil_markers_xy_confident), 2), dtype=object)
    print("Fitting circle to pupil...")

    # 2D points from DLC MouseFace
    for i in tqdm(range(len(pupil_markers_xy_confident))):
        pupilpoints = np.float32(pupil_markers_xy_confident[i].reshape(6, 2))  
        pupil_circle[i, :] = cv2.minEnclosingCircle(pupilpoints)

    # extract pupil centroid
    pupil_centre = np.asarray(tuple(pupil_circle[:, 0]), dtype=np.float32)
    pupil_x_raw = pd.Series(pupil_centre[:, 0])
    pupil_y_raw = pd.Series(pupil_centre[:, 1])

    # extract pupil movement
    pupil_xy = pd.DataFrame(np.array((pupil_x_raw, pupil_y_raw)).T, columns=['x', 'y'])
    pupil_xy[pupil_xy == 0] = np.nan
    pupil_xydist_raw = pd.Series(np.linalg.norm(pupil_xy.diff(), axis=1))

    # extract pupil diameter
    pupil_diam_raw = pd.Series(pupil_circle[:, 1], dtype=np.float32)

    return PupilResult(pd.Series(pupil_x_raw), pd.Series(pupil_y_raw), pd.Series(pupil_xydist_raw), pd.Series(pupil_diam_raw))


def eyeblink(dlc_face, eyelid_conf_thresh=.999, eyelid_smooth_window=25, eyelid_interpolation_limit=150):
    eyelid_labels_top = np.array(dlc_face.values[:, [15, 16]]) * (dlc_face.values[:, [17, 17]] > eyelid_conf_thresh)
    eyelid_labels_bot = np.array(dlc_face.values[:, [18, 19]]) * (dlc_face.values[:, [20, 20]] > eyelid_conf_thresh)
    eyelid_labels_top[eyelid_labels_top == 0] = np.nan
    eyelid_labels_bot[eyelid_labels_bot == 0] = np.nan
    eyelid_dist_raw = pd.Series(np.linalg.norm(eyelid_labels_top - eyelid_labels_bot, axis=1))
    eyelid_dist_interp = eyelid_dist_raw.interpolate(method='linear', limit=eyelid_interpolation_limit)  # linear interpolation, 2sec maximum
    eyelid_dist_smooth = pd.Series(smooth(eyelid_dist_interp, window_len=eyelid_smooth_window)).shift(periods=-int(eyelid_smooth_window/2))
    eyelid_dist_z = (eyelid_dist_smooth - eyelid_dist_smooth.mean()) / eyelid_dist_smooth.std(ddof=0)
    eyeblinks = eyelid_dist_interp < eyelid_dist_interp.median()*.75

    return eyelid_dist_z, eyeblinks



def faceregions(dlc_face, facevid, faceregion_conf_thresh,
              faceregion_size_whiskers, faceregion_size_nose, faceregion_size_mouth, faceregion_size_cheek, dlc_file):
    # checking on video
    facemp4 = cv2.VideoCapture(facevid)
    firstframe = np.array(facemp4.read()[1][:, :, 0], dtype = np.uint8)
    facemp4.release()
    plt.imshow(firstframe, cmap='gray')

    # create empty canvas
    canvas = np.zeros(firstframe.shape)
    face_anchor = pd.DataFrame(index=['x', 'y'], columns=['nosetip', 'forehead', 'mouthtip', 'chin', 'tearduct', 'eyelid_bottom'])

    # scaling of distances
    scale_width = firstframe.shape[1] / 782
    scale_height = firstframe.shape[0] / 582
    scaling = np.mean([scale_width, scale_height])
    faceregion_size_whiskers, faceregion_size_nose, faceregion_size_mouth, faceregion_size_cheek = faceregion_size_whiskers*scaling, \
        faceregion_size_nose*scaling, faceregion_size_mouth*scaling, faceregion_size_cheek*scaling

    # nosetip
    nosetip = pd.DataFrame(np.array(dlc_face.values[:, [0, 1]]) * (dlc_face.values[:, [2, 2]] > faceregion_conf_thresh), columns=['x', 'y'])
    nosetip[nosetip == 0] = np.nan
    if pd.isna(nosetip).mean()['x'] == 1: #  if more than 90% missing values, leave empty
        face_anchor.nosetip = np.nan
    else:
        face_anchor.nosetip = np.nanmean(nosetip, axis=0)
        plt.scatter(nosetip['x'], nosetip['y'], alpha=.002)
        plt.scatter(face_anchor.nosetip[0], face_anchor.nosetip[1])

    # forehead
    forehead = pd.DataFrame(np.array(dlc_face.values[:, [3, 4]]) * (dlc_face.values[:, [5, 5]] > faceregion_conf_thresh), columns=['x', 'y'])
    forehead[forehead == 0] = np.nan
    if pd.isna(forehead).mean()['x'] == 1:  # if more than 90% missing values, leave empty
        face_anchor.forehead = np.nan
    else:
        face_anchor.forehead = np.nanmean(forehead, axis=0)
        plt.scatter(forehead['x'], forehead['y'], alpha=.002)
        plt.scatter(face_anchor.forehead[0], face_anchor.forehead[1])

    # mouthtip
    mouthtip = pd.DataFrame(np.array(dlc_face.values[:, [6, 7]]) * (dlc_face.values[:, [8, 8]] > faceregion_conf_thresh), columns=['x', 'y'])
    mouthtip[mouthtip == 0] = np.nan
    if pd.isna(mouthtip).mean()['x'] == 1:  # if more than 90% missing values, leave empty
        face_anchor.mouthtip = np.nan
    else:
        face_anchor.mouthtip = np.nanmean(mouthtip, axis=0)
        plt.scatter(mouthtip['x'], mouthtip['y'], alpha=.005)
        plt.scatter(face_anchor.mouthtip[0], face_anchor.mouthtip[1])

    # chin
    chin = pd.DataFrame(np.array(dlc_face.values[:, [9, 10]]) * (dlc_face.values[:, [11, 11]] > faceregion_conf_thresh), columns=['x', 'y'])
    chin[chin == 0] = np.nan
    if pd.isna(chin).mean()['x'] == 1:  # if more than 90% missing values, leave empty
        face_anchor.chin = np.nan
    else:
        face_anchor.chin = np.nanmean(chin, axis=0)
        plt.scatter(chin['x'], chin['y'], alpha=.01)
        plt.scatter(face_anchor.chin[0], face_anchor.chin[1])

    # tearduct
    tearduct = pd.DataFrame(np.array(dlc_face.values[:, [12, 13]]) * (dlc_face.values[:, [14, 14]] > faceregion_conf_thresh), columns=['x', 'y'])
    tearduct[tearduct == 0] = np.nan
    if pd.isna(tearduct).mean()['x'] == 1:  # if more than 90% missing values, leave empty
        face_anchor.tearduct = np.nan
    else:
        face_anchor.tearduct = np.nanmean(tearduct, axis=0)
        plt.scatter(tearduct['x'], tearduct['y'], alpha=.01)
        plt.scatter(face_anchor.tearduct[0], face_anchor.tearduct[1])

    # eyelid_bottom
    eyelid_bottom = pd.DataFrame(np.array(dlc_face.values[:, [18, 19]]) * (dlc_face.values[:, [20, 20]] > faceregion_conf_thresh), columns=['x', 'y'])
    eyelid_bottom[eyelid_bottom == 0] = np.nan
    if pd.isna(eyelid_bottom).mean()['x'] == 1:  # if more than 90% missing values, leave empty
        face_anchor.eyelid_bottom = np.nan
    else:
        face_anchor.eyelid_bottom = np.nanmean(eyelid_bottom, axis=0)
        plt.scatter(eyelid_bottom['x'], eyelid_bottom['y'], alpha=.01)
        plt.scatter(face_anchor.eyelid_bottom[0], face_anchor.eyelid_bottom[1])

    # whisker inference
    whiskercentre = np.array(cv2.minEnclosingCircle(np.array(np.vstack(([np.nanmean(nosetip, axis=0)], [face_anchor.mouthtip], [face_anchor.tearduct])), dtype=np.float32))[0])
    if any([np.isnan(t) for t in whiskercentre]):
        whiskermask = np.nan
    else:
        whiskercentre = tuple(np.round(whiskercentre).astype(int) + [int(s*scaling) for s in [-10, 70]])
        whiskers = cv2.circle(firstframe, whiskercentre, int(100*faceregion_size_whiskers), color=(255, 0, 0), thickness=5)
        whiskermask = cv2.circle(canvas.copy(), whiskercentre, int(100*faceregion_size_whiskers), color=(1, 0, 0), thickness=-1).astype(bool)
        plt.imshow(whiskers, cmap='gray')

    # nose inference
    if not pd.isna(face_anchor.nosetip)[0]:
        nosecentre = tuple(np.round(face_anchor.nosetip).astype(int) + [int(s*scaling) for s in [50, 0]])
        nose = cv2.ellipse(firstframe, nosecentre, (int(70*faceregion_size_nose*scaling), int(50*faceregion_size_nose)), -60.0, 0.0, 360.0, (255, 0, 0), 5)
        nosemask = cv2.ellipse(canvas.copy(), nosecentre, (int(70*faceregion_size_nose), int(50*faceregion_size_nose)), -60.0, 0.0, 360.0, (1, 0, 0), -1).astype(bool)
        plt.imshow(nose, cmap='gray')
    else:
        nosemask = np.nan

    # mouth inference ellipse
    if any([np.isnan(t) for t in face_anchor.mouthtip]):
        mouthmask = np.nan
    else:
        mouthcentre = tuple(np.round(face_anchor.mouthtip + (np.nanmean(chin, axis=0) - face_anchor.mouthtip)/3).astype(int))
        mouthangle = math.degrees(math.atan2((np.nanmean(chin, axis=0)[1] - face_anchor.mouthtip[1]), (np.nanmean(chin, axis=0)[0] - face_anchor.mouthtip[0])))
        mouth = cv2.ellipse(firstframe, mouthcentre, (int(160*faceregion_size_mouth), int(60*faceregion_size_nose)), mouthangle, 0.0, 360.0, (255, 0, 0), 5)
        mouthmask = cv2.ellipse(canvas.copy(), mouthcentre, (int(160*faceregion_size_mouth), int(60*faceregion_size_nose)), mouthangle, 0.0, 360.0, (1, 0, 0), -1).astype(bool)
        plt.imshow(mouth, cmap='gray')

    # cheek inference ellipse
    if any([np.isnan(t) for t in face_anchor.chin]):
        cheekmask = np.nan
    else:
        cheek_centre = tuple(np.round(face_anchor.eyelid_bottom + (face_anchor.chin - face_anchor.eyelid_bottom)/2).astype(int))
        cheek = cv2.ellipse(firstframe, cheek_centre, (int(180*faceregion_size_cheek), int(100*faceregion_size_cheek)), 0.0, 0.0, 360.0, (255, 0, 0), 5)
        cheekmask = cv2.ellipse(canvas.copy(), cheek_centre, (int(180*faceregion_size_cheek), int(100*faceregion_size_cheek)), 0.0, 0.0, 360.0, (1, 0, 0), -1).astype(bool)
        plt.imshow(cheek, cmap='gray')

    masks = [nosemask, whiskermask, mouthmask, cheekmask]
    if dlc_file:
        plt.savefig(dlc_file[:-4] + "face_regions.png")
    plt.close('all')

    return masks, face_anchor


# Calculating optical flow and motion energy
def facemotion(videopath, masks, videoslice=[], total=False):
    if total:
        print("Calculating whole frame-to-frame differences...")
    else:
        print(f"Calculating optical flow and motion energy for video {videopath}...")
    facemp4 = cv2.VideoCapture(videopath)
    if videoslice:
        print("Processing slice from {} to {}...".format(videoslice[0], videoslice[-1]))
        facemp4.set(1, videoslice[0])
        framelength = videoslice[1]-videoslice[0]
    else:
        framelength = int(facemp4.get(7))
    ret, current_frame = facemp4.read()
    previous_frame = current_frame
    if total:
        total_frame_diff = np.zeros([1, 1])
    else:
        gpu_masks = []
        maskpx = []
        masks = [m.astype('float32') for m in masks]
        for m in range(len(masks)):
            gpu_mask = cv2.cuda_GpuMat()
            gpu_mask.upload(masks[m])
            gpu_masks.append(gpu_mask)
            masks[m][masks[m]==0] = np.nan
            maskpx.append(np.nansum(masks[m]))

        frame_diff = np.empty((framelength, 4))
        frame_diffmag = np.empty((framelength, 4))
        frame_diffang = np.empty((framelength, 4))
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(numLevels=5, pyrScale=.5, fastPyramids=True, winSize=25,
                                                        numIters=3, polyN=5, polySigma=1.2, flags=0)

    i = 0
    with tqdm(total=framelength) as pbar:
        while facemp4.isOpened():
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(current_frame_gray)

            previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            gpu_previous = cv2.cuda_GpuMat()
            gpu_previous.upload(previous_frame_gray)

            if total:
                total_frame_diff_current = (np.sum(cv2.absdiff(current_frame_gray, previous_frame_gray)))
                total_frame_diff = np.vstack((total_frame_diff, total_frame_diff_current))
            else:
                flow = gpu_flow.calc(gpu_frame, gpu_previous, None)
                gpu_flow_x = cv2.cuda_GpuMat(flow.size(), cv2.CV_32FC1)
                gpu_flow_y = cv2.cuda_GpuMat(flow.size(), cv2.CV_32FC1)
                cv2.cuda.split(flow, [gpu_flow_x, gpu_flow_y])
                gpu_mag, gpu_ang = cv2.cuda.cartToPolar(gpu_flow_x, gpu_flow_y, angleInDegrees=True)
                gpu_frame32 = gpu_frame.convertTo(cv2.CV_32FC1, gpu_frame)
                gpu_previous32 = gpu_previous.convertTo(cv2.CV_32FC1, gpu_previous)
                for index, (mask, gpu_mask, px) in enumerate(zip(masks, gpu_masks, maskpx)):
                    mag_mask = cv2.cuda.multiply(gpu_mag, gpu_mask)
                    ang_mask = cv2.cuda.multiply(gpu_ang, gpu_mask)
                    frame_mask = cv2.cuda.multiply(gpu_frame32, gpu_mask)
                    previous_mask = cv2.cuda.multiply(gpu_previous32, gpu_mask)
                    frame_diffmag[i, index] = cv2.cuda.absSum(mag_mask)[0] / px
                    frame_diffang[i, index] = cv2.cuda.absSum(ang_mask)[0] / px
                    frame_diff[i, index] = cv2.cuda.absSum(cv2.cuda.absdiff(frame_mask, previous_mask))[0] / px
            pbar.update(1)
            i += 1
            previous_frame = current_frame.copy()
            ret, current_frame = facemp4.read()
            if current_frame is None or (videoslice and len(frame_diff) > len(videoslice)-1):
                break
    facemp4.release()

    if total:
        return total_frame_diff
    else:
        motion = pd.DataFrame(np.hstack([frame_diff, frame_diffmag, frame_diffang]),
                              columns=['MotionEnergy_Nose', 'MotionEnergy_Whiskerpad', 'MotionEnergy_Mouth', 'MotionEnergy_Cheek',
                                       'OFmag_Nose', 'OFmag_Whiskerpad', 'OFmag_Mouth', 'OFmag_Cheek',
                                       'OFang_Nose', 'OFang_Whiskerpad', 'OFang_Mouth', 'OFang_Cheek'])
        return motion


# Calculating motion energy
def facemotion_nocuda(videopath, masks, videoslice=[], total=False):
    if total:
        print("Calculating whole frame-to-frame differences...")
    else:
        print(f"Calculating motion energy for video {videopath}...")
    facemp4 = cv2.VideoCapture(videopath)
    if videoslice:
        print("Processing slice from {} to {}...".format(videoslice[0], videoslice[-1]))
        facemp4.set(1, videoslice[0])
        framelength = videoslice[1]-videoslice[0]
    else:
        framelength = int(facemp4.get(7))
    ret, current_frame = facemp4.read()
    previous_frame = current_frame
    masks = [m.astype('float32') for m in masks]
    for m in range(len(masks)):
        masks[m][masks[m]==0] = np.nan

    frame_diff = np.empty((framelength, 4))

    i = 0
    with tqdm(total=framelength) as pbar:
        while facemp4.isOpened():
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

            for index, mask in enumerate(masks):
                frame_diff[i, index] = np.nanmean(cv2.absdiff(current_frame_gray * mask, previous_frame_gray * mask))
            pbar.update(1)
            i += 1
            previous_frame = current_frame.copy()
            ret, current_frame = facemp4.read()
            if current_frame is None or (videoslice and len(frame_diff) > len(videoslice)-1):
                break
    facemp4.release()

    motion = pd.DataFrame(frame_diff,
                            columns=['MotionEnergy_Nose', 'MotionEnergy_Whiskerpad', 
                                     'MotionEnergy_Mouth', 'MotionEnergy_Cheek',])

    return motion

