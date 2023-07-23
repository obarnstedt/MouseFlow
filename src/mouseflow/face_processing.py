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

# PUPIL
def pupilextraction(pupil_markers_xy_confident):
    pupil_circle = np.zeros(
        shape=(len(pupil_markers_xy_confident), 2), dtype=object)
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
    pupil_xy = pd.DataFrame(
        np.array((pupil_x_raw, pupil_y_raw)).T, columns=['x', 'y'])
    pupil_xy[pupil_xy == 0] = np.nan
    pupil_xydist_raw = pd.Series(np.linalg.norm(pupil_xy.diff(), axis=1))

    # extract pupil diameter
    pupil_diam_raw = pd.Series(pupil_circle[:, 1], dtype=np.float32)

    return pd.DataFrame({'PupilX': pupil_x_raw, 
                         'PupilY': pupil_y_raw, 
                         'PupilMotion': pupil_xydist_raw, 
                         'PupilDiam': pupil_diam_raw})


# FACE REGIONS
def define_faceregions(dlc_face, facevid, faceregions_sizes, dlc_file):
    # checking on video
    facemp4 = cv2.VideoCapture(facevid)
    firstframe = np.array(facemp4.read()[1][:, :, 0], dtype=np.uint8)
    facemp4.release()
    plt.imshow(firstframe, cmap='gray');

    # create empty canvas
    canvas = np.zeros(firstframe.shape)

    # define face anchors
    anchor_names = ['nosetip', 'forehead', 'mouthtip',
                    'chin', 'tearduct', 'eyelid2']
    face_anchor = pd.DataFrame(index=['x', 'y'], columns=anchor_names)
    for anchor_name in anchor_names:
        if dlc_face[anchor_name, 'x'].isnull().mean() == 1:  # if only missing values, leave empty
            face_anchor[anchor_name] = np.nan
        else:
            face_anchor[anchor_name] = dlc_face[anchor_name].mean()
            plt.scatter(dlc_face[anchor_name]['x'],
                        dlc_face[anchor_name]['y'], alpha=.002)
            plt.scatter(face_anchor[anchor_name]['x'],
                        face_anchor[anchor_name]['y'])

    # scaling of distances
    scale_width = firstframe.shape[1] / 782
    scale_height = firstframe.shape[0] / 582
    scaling = np.mean([scale_width, scale_height])
    faceregions_sizes = {x: int(k * scaling)
                         for (x, k) in faceregions_sizes.items()}

    # creating face region masks
    def create_mask(centre, scaling, size1, size2, angle):
        outline = cv2.ellipse(firstframe, centre, (int(size1*scaling),
                                                   int(size2*scaling)), angle, 0.0, 360.0, (255, 0, 0), 5)
        mask = cv2.ellipse(canvas.copy(), centre, (int(size1*faceregions_sizes['nose']), int(
            size2*scaling)), angle, 0.0, 360.0, (1, 0, 0), -1).astype(bool)
        plt.imshow(outline, cmap='gray');
        return mask

    # whisker inference
    centre_whiskers = np.array(cv2.minEnclosingCircle(np.array(np.vstack(([face_anchor.nosetip], [
        face_anchor.mouthtip], [face_anchor.tearduct])), dtype=np.float32))[0])
    if any([np.isnan(t) for t in centre_whiskers]):
        mask_whiskers = canvas.astype(bool)
    else:
        centre_whiskers = tuple(np.round(centre_whiskers).astype(
            int) + [int(s*scaling) for s in [-10*scaling, 70*scaling]])
        mask_whiskers = create_mask(
            centre_whiskers, faceregions_sizes['whiskers'], 100, 100, 0)

    # nose inference
    if not pd.isna(face_anchor.nosetip)[0]:
        centre_nose = tuple(np.round(face_anchor.nosetip).astype(
            int) + [int(s*scaling) for s in [50*scaling, 0]])
        mask_nose = create_mask(
            centre_nose, faceregions_sizes['nose'], 70, 50, -60.0)
    else:
        mask_nose = canvas.astype(bool)

    # mouth inference ellipse
    if any([np.isnan(t) for t in face_anchor.mouthtip]):
        mask_mouth = canvas.astype(bool)
    else:
        centre_mouth = tuple(np.round(
            face_anchor.mouthtip + (face_anchor.chin - face_anchor.mouthtip)/3).astype(int))
        angle_mouth = math.degrees(math.atan2((face_anchor.chin[
            1] - face_anchor.mouthtip[1]), (face_anchor.chin[0] - face_anchor.mouthtip[0])))
        mask_mouth = create_mask(
            centre_mouth, faceregions_sizes['mouth'], 160, 60, angle_mouth)

    # cheek inference ellipse
    if any([np.isnan(t) for t in face_anchor.chin]):
        mask_cheek = canvas.astype(bool)
    else:
        centre_cheek = tuple(np.round(face_anchor.eyelid_bottom +
                             (face_anchor.chin - face_anchor.eyelid_bottom)/2).astype(int))
        mask_cheek = create_mask(
            centre_cheek, faceregions_sizes['cheek'], 180, 100, 0)
        
    masks = [mask_nose, mask_whiskers, mask_mouth, mask_cheek]
    if dlc_file:
        plt.savefig(dlc_file[:-4] + "face_regions.png");
    plt.close('all');

    return masks, face_anchor


# Calculating optical flow and motion energy
def facemotion(videopath, masks, videoslice=[]):
    print(f"Calculating optical flow and motion energy for video {videopath}...")
    facemp4 = cv2.VideoCapture(videopath)
    if videoslice:
        print("Processing slice from {} to {}...".format(
            videoslice[0], videoslice[-1]))
        facemp4.set(1, videoslice[0])
        framelength = videoslice[1]-videoslice[0]
    else:
        framelength = int(facemp4.get(7))
    _, current_frame = facemp4.read()
    previous_frame = current_frame
    gpu_masks = []
    maskpx = []
    masks = [m.astype('float32') for m in masks]
    for m in range(len(masks)):
        gpu_mask = cv2.cuda_GpuMat()
        gpu_mask.upload(masks[m])
        gpu_masks.append(gpu_mask)
        masks[m][masks[m] == 0] = np.nan
        maskpx.append(np.nansum(masks[m]))

    frame_diff = np.empty((framelength, 4))
    frame_diffmag = np.empty((framelength, 4))
    frame_diffang = np.empty((framelength, 4))
    gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(numLevels=5, pyrScale=.5, fastPyramids=True, winSize=25,
                                                    numIters=3, polyN=5, polySigma=1.2, flags=0)

    i = 0
    with tqdm(total=framelength) as pbar:
        while facemp4.isOpened():
            current_frame_gray = cv2.cvtColor(
                current_frame, cv2.COLOR_BGR2GRAY)
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(current_frame_gray)

            previous_frame_gray = cv2.cvtColor(
                previous_frame, cv2.COLOR_BGR2GRAY)
            gpu_previous = cv2.cuda_GpuMat()
            gpu_previous.upload(previous_frame_gray)
            flow = gpu_flow.calc(gpu_frame, gpu_previous, None)
            gpu_flow_x = cv2.cuda_GpuMat(flow.size(), cv2.CV_32FC1)
            gpu_flow_y = cv2.cuda_GpuMat(flow.size(), cv2.CV_32FC1)
            cv2.cuda.split(flow, [gpu_flow_x, gpu_flow_y])
            gpu_mag, gpu_ang = cv2.cuda.cartToPolar(
                gpu_flow_x, gpu_flow_y, angleInDegrees=True)
            gpu_frame32 = gpu_frame.convertTo(cv2.CV_32FC1, gpu_frame)
            gpu_previous32 = gpu_previous.convertTo(
                cv2.CV_32FC1, gpu_previous)
            for index, (mask, gpu_mask, px) in enumerate(zip(masks, gpu_masks, maskpx)):
                mag_mask = cv2.cuda.multiply(gpu_mag, gpu_mask)
                ang_mask = cv2.cuda.multiply(gpu_ang, gpu_mask)
                frame_mask = cv2.cuda.multiply(gpu_frame32, gpu_mask)
                previous_mask = cv2.cuda.multiply(gpu_previous32, gpu_mask)
                frame_diffmag[i, index] = cv2.cuda.absSum(mag_mask)[0] / px
                frame_diffang[i, index] = cv2.cuda.absSum(ang_mask)[0] / px
                frame_diff[i, index] = cv2.cuda.absSum(
                    cv2.cuda.absdiff(frame_mask, previous_mask))[0] / px
            pbar.update(1)
            i += 1
            previous_frame = current_frame.copy()
            _, current_frame = facemp4.read()
            if current_frame is None or (videoslice and len(frame_diff) > len(videoslice)-1):
                break
    facemp4.release()

    motion = pd.DataFrame(np.hstack([frame_diff, frame_diffmag, frame_diffang]),
                            columns=['MotionEnergy_Nose', 'MotionEnergy_Whiskerpad', 'MotionEnergy_Mouth', 'MotionEnergy_Cheek',
                                    'OFmag_Nose', 'OFmag_Whiskerpad', 'OFmag_Mouth', 'OFmag_Cheek',
                                    'OFang_Nose', 'OFang_Whiskerpad', 'OFang_Mouth', 'OFang_Cheek'])
    return motion


# Calculating motion energy
def facemotion_nocuda(videopath, masks, videoslice=[]):
    print(f"Calculating motion energy for video {videopath}...")
    facemp4 = cv2.VideoCapture(videopath)
    if videoslice:
        print("Processing slice from {} to {}...".format(
            videoslice[0], videoslice[-1]))
        facemp4.set(1, videoslice[0])
        framelength = videoslice[1]-videoslice[0]
    else:
        framelength = int(facemp4.get(7))
    _, current_frame = facemp4.read()
    previous_frame = current_frame
    masks = [m.astype('float32') for m in masks]
    for m in range(len(masks)):
        masks[m][masks[m] == 0] = np.nan

    frame_diff = np.empty((framelength, 4))

    i = 0
    with tqdm(total=framelength) as pbar:
        while facemp4.isOpened():
            current_frame_gray = cv2.cvtColor(
                current_frame, cv2.COLOR_BGR2GRAY)
            previous_frame_gray = cv2.cvtColor(
                previous_frame, cv2.COLOR_BGR2GRAY)

            for index, mask in enumerate(masks):
                frame_diff[i, index] = np.nanmean(cv2.absdiff(
                    current_frame_gray * mask, previous_frame_gray * mask))
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
