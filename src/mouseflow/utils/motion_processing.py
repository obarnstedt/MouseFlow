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



def faceregions(params, dlc_face):
    # checking on video
    facemp4 = cv2.VideoCapture(params['paths']['Data_FaceCam'])
    firstframe = np.array(facemp4.read()[1][:, :, 0], dtype = np.uint8)
    facemp4.release()
    plt.imshow(firstframe, cmap='gray')

    # create empty canvas
    canvas = np.zeros([582, 782])
    face_anchor = pd.DataFrame(index=['x', 'y'], columns=['nosetip', 'forehead', 'mouthtip', 'chin', 'tearduct', 'eyelid_bottom'])

    # nosetip
    nosetip = pd.DataFrame(np.array(dlc_face.values[:, [0, 1]]) * (dlc_face.values[:, [2, 2]] > params['cameras']['faceregion_conf_thresh']), columns=['x', 'y'])
    nosetip[nosetip == 0] = np.nan
    if pd.isna(nosetip).mean()['x'] == 1: #  if more than 90% missing values, leave empty
        face_anchor.nosetip = np.nan
    else:
        face_anchor.nosetip = np.nanmean(nosetip, axis=0)
        plt.scatter(nosetip['x'], nosetip['y'], alpha=.002)
        plt.scatter(face_anchor.nosetip[0], face_anchor.nosetip[1])

    # forehead
    forehead = pd.DataFrame(np.array(dlc_face.values[:, [3, 4]]) * (dlc_face.values[:, [5, 5]] > params['cameras']['faceregion_conf_thresh']), columns=['x', 'y'])
    forehead[forehead == 0] = np.nan
    if pd.isna(forehead).mean()['x'] == 1:  # if more than 90% missing values, leave empty
        face_anchor.forehead = np.nan
    else:
        face_anchor.forehead = np.nanmean(forehead, axis=0)
        plt.scatter(forehead['x'], forehead['y'], alpha=.002)
        plt.scatter(face_anchor.forehead[0], face_anchor.forehead[1])

    # mouthtip
    mouthtip = pd.DataFrame(np.array(dlc_face.values[:, [6, 7]]) * (dlc_face.values[:, [8, 8]] > params['cameras']['faceregion_conf_thresh']), columns=['x', 'y'])
    mouthtip[mouthtip == 0] = np.nan
    if pd.isna(mouthtip).mean()['x'] == 1:  # if more than 90% missing values, leave empty
        face_anchor.mouthtip = np.nan
    else:
        face_anchor.mouthtip = np.nanmean(mouthtip, axis=0)
        plt.scatter(mouthtip['x'], mouthtip['y'], alpha=.005)
        plt.scatter(face_anchor.mouthtip[0], face_anchor.mouthtip[1])

    # chin
    chin = pd.DataFrame(np.array(dlc_face.values[:, [9, 10]]) * (dlc_face.values[:, [11, 11]] > params['cameras']['faceregion_conf_thresh']), columns=['x', 'y'])
    chin[chin == 0] = np.nan
    if pd.isna(chin).mean()['x'] == 1:  # if more than 90% missing values, leave empty
        face_anchor.chin = np.nan
    else:
        face_anchor.chin = np.nanmean(chin, axis=0)
        plt.scatter(chin['x'], chin['y'], alpha=.01)
        plt.scatter(face_anchor.chin[0], face_anchor.chin[1])

    # tearduct
    tearduct = pd.DataFrame(np.array(dlc_face.values[:, [12, 13]]) * (dlc_face.values[:, [14, 14]] > params['cameras']['faceregion_conf_thresh']), columns=['x', 'y'])
    tearduct[tearduct == 0] = np.nan
    if pd.isna(tearduct).mean()['x'] == 1:  # if more than 90% missing values, leave empty
        face_anchor.tearduct = np.nan
    else:
        face_anchor.tearduct = np.nanmean(tearduct, axis=0)
        plt.scatter(tearduct['x'], tearduct['y'], alpha=.01)
        plt.scatter(face_anchor.tearduct[0], face_anchor.tearduct[1])

    # eyelid_bottom
    eyelid_bottom = pd.DataFrame(np.array(dlc_face.values[:, [18, 19]]) * (dlc_face.values[:, [20, 20]] > params['cameras']['faceregion_conf_thresh']), columns=['x', 'y'])
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
        whiskers = cv2.circle(firstframe, whiskercentre, int(100*params['cameras']['faceregion_size_whiskers']), color=(255, 0, 0), thickness=5)
        whiskermask = cv2.circle(canvas.copy(), whiskercentre, int(100*params['cameras']['faceregion_size_whiskers']), color=(1, 0, 0), thickness=-1).astype(bool)
        plt.imshow(whiskers, cmap='gray')

    # nose inference
    if not pd.isna(face_anchor.nosetip)[0]:
        nosecentre = tuple(np.round(face_anchor.nosetip).astype(int) + [50, 0])
        nose = cv2.ellipse(firstframe, nosecentre, (int(70*params['cameras']['faceregion_size_nose']), int(50*params['cameras']['faceregion_size_nose'])), -60.0, 0.0, 360.0, (255, 0, 0), 5)
        nosemask = cv2.ellipse(canvas.copy(), nosecentre, (int(70*params['cameras']['faceregion_size_nose']), int(50*params['cameras']['faceregion_size_nose'])), -60.0, 0.0, 360.0, (1, 0, 0), -1).astype(bool)
        plt.imshow(nose, cmap='gray')
    else:
        nosemask = np.nan

    # mouth inference ellipse
    if any([np.isnan(t) for t in face_anchor.mouthtip]):
        mouthmask = np.nan
    else:
        mouthcentre = tuple(np.round(face_anchor.mouthtip + (np.nanmean(chin, axis=0) - face_anchor.mouthtip)/3).astype(int))
        mouthangle = math.degrees(math.atan2((np.nanmean(chin, axis=0)[1] - face_anchor.mouthtip[1]), (np.nanmean(chin, axis=0)[0] - face_anchor.mouthtip[0])))
        mouth = cv2.ellipse(firstframe, mouthcentre, (int(160*params['cameras']['faceregion_size_mouth']), int(60*params['cameras']['faceregion_size_nose'])), mouthangle, 0.0, 360.0, (255, 0, 0), 5)
        mouthmask = cv2.ellipse(canvas.copy(), mouthcentre, (int(160*params['cameras']['faceregion_size_mouth']), int(60*params['cameras']['faceregion_size_nose'])), mouthangle, 0.0, 360.0, (1, 0, 0), -1).astype(bool)
        plt.imshow(mouth, cmap='gray')

    # cheek inference ellipse
    if any([np.isnan(t) for t in face_anchor.chin]):
        cheekmask = np.nan
    else:
        cheek_centre = tuple(np.round(face_anchor.eyelid_bottom + (face_anchor.chin - face_anchor.eyelid_bottom)/2).astype(int))
        cheek = cv2.ellipse(firstframe, cheek_centre, (int(180*params['cameras']['faceregion_size_cheek']), int(100*params['cameras']['faceregion_size_cheek'])), 0.0, 0.0, 360.0, (255, 0, 0), 5)
        cheekmask = cv2.ellipse(canvas.copy(), cheek_centre, (int(180*params['cameras']['faceregion_size_cheek']), int(100*params['cameras']['faceregion_size_cheek'])), 0.0, 0.0, 360.0, (1, 0, 0), -1).astype(bool)
        plt.imshow(cheek, cmap='gray')

    masks = [nosemask, whiskermask, mouthmask, cheekmask]
    plt.savefig(os.path.join(params['paths']['Results_Cam_Dir'], '') + "face_regions.png")
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

# Calculating motion energy
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
    return cyl, cyl_smooth



# Calculating motion energy
def get_mp4timestamps(videopath):
    print("Extracting video time stamps for each frame...")
    cap = cv2.VideoCapture(videopath)
    framelength = int(cap.get(7))
    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]

    with tqdm(total=framelength) as pbar:
        while (cap.isOpened()):
            frame_exists, curr_frame = cap.read()
            if frame_exists:
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            else:
                break
            pbar.update(1)
        cap.release()

    return timestamps

def create_labeled_video_face(params, dlc_face, pupil, face, facemasks, face_anchor, write_mp4=False):
    facemp4 = cv2.VideoCapture(params['paths']['Data_FaceCam'])
    if params['cameras']['face_output'] > 1:
        vidlength = round(params['cameras']['face_output'])
    else:
        vidlength = int(facemp4.get(7))
    print("Labelling face video...")
    ret, current_frame = facemp4.read()
    index = 0

    plt.close("all")
    # plt.ioff()  # hide figures
    plt.ion()  # show figures
    plt.figure(figsize=(14, 10), dpi=80)
    if params['cameras']['use_dark_background']:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    if write_mp4:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(params['paths']['Results_FaceVid'], fourcc, int(params['cameras']['FaceCam_FPS']), (1120, 800))
    elif not os.path.exists(os.path.join(params['paths']['Results_Cam_Dir'], 'face_output_vid')):
        os.makedirs(os.path.join(params['paths']['Results_Cam_Dir'], 'face_output_vid'))

    facenames = list(face.columns)
    facedatapoints = face.values[:]
    minvalue = min(np.array(face.quantile(q=.05)) - np.array(range(len(face.columns))))
    maxvalue = max(np.array(face.quantile(q=.95)) - np.array(range(len(face.columns))))
    showseconds = 10
    showtimepoints = int(params['cameras']['FaceCam_FPS']) * showseconds
    midpoint = int(showtimepoints/2)
    cmap = plt.get_cmap('Dark2')

    with tqdm(total=vidlength) as pbar:
        while facemp4.isOpened():
            # shift X window of data
            plt.subplot(212)
            plt.cla()
            plt.xlim(0, showtimepoints)
            plt.xticks(np.arange(0, showtimepoints, int(params['cameras']['FaceCam_FPS'])), [str(a) for a in range(-5, 5)])
            plt.ylim(minvalue, maxvalue)
            plt.yticks(np.arange(-len(face.columns)+1, 1), reversed(facenames), fontsize=12, fontweight='bold')
            facedatapointsshifted = shift5(facedatapoints, midpoint-index)
            for i in range(len(facedatapoints.transpose())):
                plt.axhline(y=-i, linestyle='--', color=cmap(i), alpha=.5, linewidth=1.2)
                plt.plot(facedatapointsshifted.transpose()[i] - i, color=cmap(i))
                # plt.legend(facenames, loc=2)
            if params['cameras']['use_dark_background']:
                plt.axvline(x=midpoint, color='white')
            else:
                plt.axvline(x=midpoint, color='black')
            j = len(face.columns)-1
            for i in plt.gca().get_yticklabels():
                i.set_color(cmap(j))
                j -= 1
            plt.xlabel('Time [sec]')

            # plot face frame
            plt.subplot(221)
            plt.cla()
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            plt.imshow(current_frame_gray, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.text(10, 30, str(params['paths']['Data_FaceCam'][-22:]), color='w', size=8)
            plt.text(10, 60, "%06d" % index, color='w')

            # plot fix points
            for key in face_anchor:
                plt.scatter(face_anchor[key][0], face_anchor[key][1], s=200, color='w', alpha=.5)

            # plot skeleton
            plt.plot([face_anchor.nosetip[0], face_anchor.forehead[0]], [face_anchor.nosetip[1], face_anchor.forehead[1]], color='w')
            plt.plot([face_anchor.nosetip[0], face_anchor.mouthtip[0]], [face_anchor.nosetip[1], face_anchor.mouthtip[1]], color='w')
            plt.plot([face_anchor.nosetip[0], face_anchor.tearduct[0]], [face_anchor.nosetip[1], face_anchor.tearduct[1]], color='w')
            plt.plot([face_anchor.mouthtip[0], face_anchor.tearduct[0]], [face_anchor.mouthtip[1], face_anchor.tearduct[1]], color='w')
            plt.plot([face_anchor.eyelid_bottom[0], face_anchor.chin[0]], [face_anchor.eyelid_bottom[1], face_anchor.chin[1]], color='w')

            # plot face regions
            for i in range(len(facemasks)):
                plt.contour(facemasks[i], colors=[cmap(i+2)], linestyles='--', size=5)

            # plot dynamic DLC points whole face
            for i in range(np.uint8(len(dlc_face.columns)/3)-6):
                plt.scatter(dlc_face.values[index][i*3], dlc_face.values[index][i*3+1], alpha=0.7, s=100, color='w')
            plt.scatter(pupil['x_raw'][index], pupil['y_raw'][index], alpha=0.7, s=30, color='w')
            plt.xlim(0, params['cameras']['FaceCam_Width'])
            plt.ylim(params['cameras']['FaceCam_Height'], 0)

            # plot eye detail
            plt.subplot(222)
            plt.cla()
            pupilmeanx = np.round(np.mean(pupil['x_smooth'])).astype(int)
            pupilmeany = np.round(np.mean(pupil['y_smooth'])).astype(int)
            plt.imshow(current_frame_gray[pupilmeany-90:pupilmeany+90, pupilmeanx-120:pupilmeanx+120], cmap='gray')
            plt.axis('off')
            plt.tight_layout()

            # plot eye DLC points
            for i in range(4, np.uint8(len(dlc_face.columns)/3)):
                plt.scatter(dlc_face.values[index][i*3]-(pupilmeanx-120), dlc_face.values[index][i*3+1]-(pupilmeany-90), alpha=1, s=100, color='w')
            plt.plot([(dlc_face.values[index][12]-(pupilmeanx-120)), (dlc_face.values[index][15]-(pupilmeanx-120))], [(dlc_face.values[index][13]-(pupilmeany-90)), (dlc_face.values[index][16]-(pupilmeany-90))], color=cmap(1))
            plt.plot([(dlc_face.values[index][12]-(pupilmeanx-120)), (dlc_face.values[index][18]-(pupilmeanx-120))], [(dlc_face.values[index][13]-(pupilmeany-90)), (dlc_face.values[index][19]-(pupilmeany-90))], color=cmap(1))
            plt.scatter(pupil['x_raw'][index]-(pupilmeanx-120), pupil['y_raw'][index]-(pupilmeany-90), alpha=0.7, s=100, color='w')

            # plot pupil
            pupilplot = plt.Circle((np.round(pupil['x_interp'][index]).astype(int)-(pupilmeanx-120), np.round(pupil['y_interp'][index]).astype(int)-(pupilmeany-90)), np.round(pupil['diam_interp'][index]).astype(int), color=cmap(0), alpha=0.5, fill=False, linewidth=3)
            fig = plt.gcf()
            ax = fig.gca()
            ax.add_artist(pupilplot)

            if write_mp4:
                out.write(mplfig_to_npimage(fig))
            else:
                plt.savefig(os.path.join(params['paths']['Results_Cam_Dir'], 'face_output_vid', '') + "file%06d.png" % index)

            pbar.update(1)
            index += 1
            if index > vidlength:
                break
            ret, current_frame = facemp4.read()
            if current_frame is None:
                break
    if write_mp4:
       out.release()
    facemp4.release()


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


def dlc_pointmotion(dlc, body_conf_thresh=.5, body_smooth_window=25, body_interpolation_limit=150):
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
    return pd.Series(xydist_raw), pd.Series(xydist_z), pd.Series(deg_interp)


def dlc_angle(point1, point2, point3, body_conf_thresh=.5, body_smooth_window=25, body_interpolation_limit=150):
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
    return pd.Series(angle_smooth), pd.Series(deg1_smooth)

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


def create_labeled_video_body(params, body, write_mp4=False):
    dlc_bodyvid = cv2.VideoCapture(glob.glob(os.path.join(params['paths']['Results_DLC_Body'], '') + '*DeepCut*.mp4')[0])
    if params['cameras']['body_output'] > 1:
        vidlength = round(params['cameras']['body_output'])
    else:
        vidlength = int(dlc_bodyvid.get(7))
    print("Labelling body video...")
    ret, current_frame = dlc_bodyvid.read()
    index = 0

    plt.close("all")
    plt.ioff()  # hide figures
    # plt.ion()  # show figures
    fig = plt.figure(figsize=(10, 8), dpi=80)

    if write_mp4:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(params['paths']['Results_BodyVid'], fourcc, int(params['cameras']['BodyCam_FPS']), (800, 640))
    elif not os.path.exists(os.path.join(params['paths']['Results_Cam_Dir'], 'body_output_vid')):
        os.makedirs(os.path.join(params['paths']['Results_Cam_Dir'], 'body_output_vid'))

    with tqdm(total=vidlength) as pbar:
        while dlc_bodyvid.isOpened():
            # shift X window of data
            plt.subplot(212)
            plt.cla()
            bodynames = list(body.columns)
            bodydatapoints = (body.values[:] - np.nanmean(np.array(body.values, dtype=float), axis=0)) / np.sqrt(np.nanstd(np.array(body.values, dtype=float), axis=0))
            plt.xlim(0, 750)
            plt.ylim(-25, 50)
            bodydatapointsshifted = shift5(bodydatapoints, 375-index)
            for i in range(len(bodydatapoints.transpose())):
                plt.plot(bodydatapointsshifted.transpose()[i] + i)
                plt.legend(bodynames, loc=2)
            plt.axvline(x=375, color='black')

            # plot body frame
            plt.subplot(211)
            plt.cla()
            plt.imshow(current_frame)
            plt.axis('off')
            plt.tight_layout()
            plt.text(10, 30, str(params['paths']['Data_BodyCam'][-22:]), color='w', size=8)
            plt.text(10, 60, "%06d" % index, color='w')

            if write_mp4:
                out.write(mplfig_to_npimage(fig))
            else:
                plt.savefig(os.path.join(params['paths']['Results_Cam_Dir'], 'body_output_vid', '') + "file%06d.png" % index)

            pbar.update(1)
            index = index+1
            if index > vidlength:
                break
            ret, current_frame = dlc_bodyvid.read()
            if current_frame is None:
                break
    if write_mp4:
       out.release()
    dlc_bodyvid.release()

def plot_sweep_inference(params, which_camera, triggersum, inferredtriggersum, total_frame_diff_diff):
    plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    plotrows = round(len(triggersum) / 5) + 2
    plotcols = 5
    plt.subplot(plotrows,1,1)
    dashsize = int(total_frame_diff_diff.max()/10)
    for t in triggersum:
        plt.axvline(t, dashes=(dashsize, dashsize), color='r')
    for u in inferredtriggersum:
        plt.axvline(u, color='g')
    total_frame_diff_diff.plot(color='b')
    plt.axis('off')
    plt.title('Recorded (red) and inferred (green) sweep transitions for {} camera; max sweep inference {} frames'.format(which_camera, params['cameras']['infer_sweep_maxframes']))
    for idx in range(len(triggersum)-1):
        triggerrange = np.arange(inferredtriggersum[idx]-params['cameras']['infer_sweep_maxframes'], inferredtriggersum[idx]+params['cameras']['infer_sweep_maxframes'])
        plt.subplot(plotrows, plotcols, idx+6)
        plt.axvline(triggersum[idx], dashes=(dashsize, dashsize), color='r')
        plt.axvline(inferredtriggersum[idx], color='g')
        total_frame_diff_diff[triggerrange].plot(color='b')
        plt.axis('off')
        plt.title('{}: Diff: {} frames'.format(idx, triggersum[idx]-inferredtriggersum[idx]))
        plt.ylim(total_frame_diff_diff[triggerrange].min()*2, total_frame_diff_diff[triggerrange].max()*2)
    plt.savefig(os.path.join(params['paths']['Results_Cam_Dir'], '') + which_camera + "_sweep_inference.pdf")
    plt.close('all')





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