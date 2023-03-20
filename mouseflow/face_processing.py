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
from mouseflow.utils.generic import smooth


plt.interactive(False)

def pupilextraction(dlc_face, pupil_conf_thresh=.99, pupil_smooth_window=75, pupil_interpolation_limit=150, pupil_na_limit=.25):
    pupil_labels_xy = dlc_face.values[:, [21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37]]
    pupil_confidence = dlc_face.values[:, [23, 26, 29, 32, 35, 38]]
    pupil_confident = pupil_confidence[:, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]] > pupil_conf_thresh
    pupil_labels_xy_confident = pupil_labels_xy * pupil_confident
    pupil_labels_xy_confident[pupil_labels_xy_confident == 0] = np.nan
    pupil_circle = np.zeros(shape=(len(pupil_labels_xy), 2), dtype=object)
    print("Fitting circle to pupil...")
    for i in tqdm(range(len(pupil_labels_xy))):
        pupilpoints = np.float32(pupil_labels_xy_confident[i].reshape(6, 2))  # 2D points from DLC MouseFace
        pupil_circle[i, :] = np.asarray(cv2.minEnclosingCircle(pupilpoints))

    pupil_centre = np.asarray(tuple(pupil_circle[:, 0]), dtype=np.float32)
    pupil_x_raw = pd.Series(pupil_centre[:, 0])
    pupil_x_interp = pupil_x_raw.interpolate(method='linear', limit=pupil_interpolation_limit)
    pupil_x_smooth = pd.Series(smooth(pupil_x_interp, window_len=round(pupil_smooth_window/10))).shift(
        periods=-int(pupil_smooth_window/10 / 2))
    pupil_x_z = pd.Series((pupil_x_smooth - pupil_x_smooth.mean()) / pupil_x_smooth.std(ddof=0))

    pupil_y_raw = pd.Series(pupil_centre[:, 1])
    pupil_y_interp = pupil_y_raw.interpolate(method='linear', limit=pupil_interpolation_limit)
    pupil_y_smooth = pd.Series(smooth(pupil_y_interp, window_len=round(pupil_smooth_window/10))).shift(
        periods=-int(pupil_smooth_window/10 / 2))
    pupil_y_z = pd.Series((pupil_y_smooth - pupil_y_smooth.mean()) / pupil_y_smooth.std(ddof=0))

    pupil_xy = pd.DataFrame(np.array((pupil_x_raw, pupil_y_raw)).T, columns=['x', 'y'])
    pupil_xy[pupil_xy == 0] = np.nan
    pupil_xydiff_raw = pupil_xy.diff()
    pupil_xydist_raw = pd.Series(np.linalg.norm(pupil_xydiff_raw, axis=1))
    pupil_xydist_interp = pupil_xydist_raw.interpolate(method='linear', limit=pupil_interpolation_limit)  # linear interpolation
    pupil_xydist_smooth = pd.Series(smooth(pupil_xydist_interp, window_len=round(pupil_smooth_window/10))).shift(
        periods=-int(pupil_smooth_window/10 / 2))
    pupil_xydist_z = ((pupil_xydist_interp - pupil_xydist_interp.mean()) / pupil_xydist_interp.std(ddof=0))[:len(pupil_x_raw)]

    pupil_diam_raw = pd.Series(pupil_circle[:, 1], dtype=np.float32)
    pupil_diam_interp = pupil_diam_raw.interpolate(method='linear',
                                                   limit=pupil_interpolation_limit)  # linear interpolation
    pupil_diam_smooth = pd.Series(smooth(pupil_diam_interp, window_len=pupil_smooth_window)).shift(
        periods=-int(pupil_smooth_window / 2))
    pupil_diam_z = pd.Series((pupil_diam_smooth - pupil_diam_smooth.mean()) / pupil_diam_smooth.std(ddof=0))

    pupil_saccades = pd.Series(pupil_x_smooth.diff().abs() > 1.5)

    pupil = pd.concat([pupil_x_raw, pupil_x_interp, pupil_x_smooth, pupil_x_z, pupil_y_raw, pupil_y_interp,
                       pupil_y_smooth, pupil_y_z, pupil_diam_raw, pupil_diam_interp, pupil_diam_smooth,
                       pupil_diam_z, pupil_xydist_z, pupil_saccades], axis=1).rename(columns={0:'x_raw', 1:'x_interp', 2:'x_smooth', 3:'x_z',
                                                                      4:'y_raw', 5:'y_interp', 6:'y_smooth', 7:'y_z',
                                                                      8:'diam_raw', 9:'diam_interp', 10:'diam_smooth', 11:'diam_z',
                                                                      12:'shift_z', 13:'saccades'})

    if np.mean(pupil_diam_raw.isna()) > pupil_na_limit:  # if too many missing values present, fill everything with NANs
        pupil = np.nan

    return pupil


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
              faceregion_size_whiskers, faceregion_size_nose, faceregion_size_mouth, faceregion_size_cheek):
    # checking on video
    facemp4 = cv2.VideoCapture(facevid)
    firstframe = np.array(facemp4.read()[1][:, :, 0], dtype = np.uint8)
    facemp4.release()
    plt.imshow(firstframe, cmap='gray')

    # create empty canvas
    canvas = np.zeros([582, 782])
    face_anchor = pd.DataFrame(index=['x', 'y'], columns=['nosetip', 'forehead', 'mouthtip', 'chin', 'tearduct', 'eyelid_bottom'])

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
        whiskercentre = tuple(np.round(whiskercentre).astype(int) + [-10, 70])
        whiskers = cv2.circle(firstframe, whiskercentre, int(100*faceregion_size_whiskers), color=(255, 0, 0), thickness=5)
        whiskermask = cv2.circle(canvas.copy(), whiskercentre, int(100*faceregion_size_whiskers), color=(1, 0, 0), thickness=-1).astype(bool)
        plt.imshow(whiskers, cmap='gray')

    # nose inference
    if not pd.isna(face_anchor.nosetip)[0]:
        nosecentre = tuple(np.round(face_anchor.nosetip).astype(int) + [50, 0])
        nose = cv2.ellipse(firstframe, nosecentre, (int(70*faceregion_size_nose), int(50*faceregion_size_nose)), -60.0, 0.0, 360.0, (255, 0, 0), 5)
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
    # plt.savefig(os.path.join(params['paths']['Results_Cam_Dir'], '') + "face_regions.png")
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
        for m in range(len(masks)):
            masks[m] = masks[m].astype('float32')
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
                    # frame_diff[i, index] = np.nanmean(cv2.absdiff(current_frame_gray * mask, previous_frame_gray * mask))
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
        # nose_smooth = pd.Series(smooth(frame_diff[:, 0], window_len=smooth_window)).shift(periods=-int(smooth_window/2))
        # whisker_smooth = pd.Series(smooth(frame_diff[:, 1], window_len=smooth_window)).shift(periods=-int(smooth_window/2))
        # mouth_smooth = pd.Series(smooth(frame_diff[:, 2], window_len=smooth_window)).shift(periods=-int(smooth_window/2))
        # cheek_smooth = pd.Series(smooth(frame_diff[:, 3], window_len=smooth_window)).shift(periods=-int(smooth_window/2))
        # frame_diff_smooth = pd.DataFrame({'MotionEnergy_Nose_raw': frame_diff[:, 0],
        #                                   'MotionEnergy_Whiskerpad_raw': frame_diff[:, 1],
        #                                   'MotionEnergy_Mouth_raw': frame_diff[:, 2],
        #                                   'MotionEnergy_Cheek_raw': frame_diff[:, 3],
        #                                   'MotionEnergy_Nose': nose_smooth[:frame_diff.shape[0]],
        #                                   'MotionEnergy_Whiskerpad': whisker_smooth[:frame_diff.shape[0]],
        #                                   'MotionEnergy_Mouth': mouth_smooth[:frame_diff.shape[0]],
        #                                   'MotionEnergy_Cheek': cheek_smooth[:frame_diff.shape[0]]}).apply(zscore)
        return motion
