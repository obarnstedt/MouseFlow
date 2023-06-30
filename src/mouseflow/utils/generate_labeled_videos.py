#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pyTreadMouse functions to analyse camera data

Created on Wed May  8 14:31:51 2019
@author: Oliver Barnstedt
"""


import glob
import math
import os.path

import cv2
import flow_vis  # visualisation from https://github.com/tomrunia/OpticalFlow_Visualization
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize, signal
from scipy.stats.mstats import zscore
from tqdm import tqdm

plt.interactive(False)

# path_vid_face = '/media/oliver/Oliver_SSD1/Musall/2202_cam1_20211213.avi'
# path_dlc_face = '/media/oliver/Oliver_SSD1/Musall/mouseflow/2202_cam1_20211213DeepCut_resnet50_MouseFaceAug21shuffle1_1030000.h5'
# path_mf = '/media/oliver/Oliver_SSD1/Musall/mouseflow/2202_cam1_20211213DeepCut_resnet50_MouseFaceAug21shuffle1_1030000_analysis.h5'
path_vid_face = '/media/oliver/Oliver_SSD1/Ziyan/Basler_acA1920-150um__40032679__20230413_113011645_crop.mp4'
path_dlc_face = '/media/oliver/Oliver_SSD1/Ziyan/mouseflow/Basler_acA1920-150um__40032679__20230413_113011645_cropDeepCut_resnet50_MouseFaceAug21shuffle1_1030000.h5'
path_mf = '/media/oliver/Oliver_SSD1/Ziyan/mouseflow/Basler_acA1920-150um__40032679__20230413_113011645_cropDeepCut_resnet50_MouseFaceAug21shuffle1_1030000_analysis.h5'
startframe=500
cols_to_plot=['PupilDiam', 'PupilX', 'MotionEnergy_Mouth', 'OFang_Whiskerpad', 'OFang_Nose']
generate_frames=1000
blend_gray_optflow=0.5
smooth_data=15
dlc_conf_thresh=0.99
use_dark_background = True
resultsdir=False

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


def create_labeled_video_face(path_vid_face, path_dlc_face, path_mf,
                              generate_frames=1000, startframe=500, use_dark_background=True, resultsdir=False,
                              cols_to_plot=['PupilDiam', 'PupilX', 'MotionEnergy_Mouth', 'OFang_Whiskerpad', 'OFang_Nose'],
                              blend_gray_optflow=0.5, smooth_data=15, dlc_conf_thresh=0.99):
    facemp4 = cv2.VideoCapture(path_vid_face)
    facemp4.set(1, startframe)
    FaceCam_FPS = facemp4.get(cv2.CAP_PROP_FPS)
    index = startframe
    if type(generate_frames)==int:
        vidlength = generate_frames
    elif generate_frames==True:
        vidlength = int(facemp4.get(7))
    elif len(generate_frames)==2:
        facemp4.set(1, generate_frames[0])
        vidlength = generate_frames[1] - generate_frames[0]
        index = generate_frames[0]
    print("Labelling face video...")
    ret, current_frame = facemp4.read()
    previous_frame = current_frame

    gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(numLevels=5, pyrScale=.5, fastPyramids=True, winSize=25,
                                                        numIters=3, polyN=5, polySigma=1.2, flags=0)

    dlc_face = pd.read_hdf(path_dlc_face)
    hfg = h5py.File(path_mf)
    facemasks = (np.array(hfg['facemasks'])).astype('float32')
    pupil = pd.read_hdf(path_mf, 'pupil')
    face = pd.read_hdf(path_mf, 'face')
    face_anchor = pd.read_hdf(path_mf, 'face_anchor')
    
    plt.close("all")
    # plt.ioff()  # hide figures
    plt.ion()  # show figures
    fig, axd = plt.subplot_mosaic([['upper left', 'upper centre', 'upper right'],
                                ['bottom', 'bottom', 'bottom']], figsize=(14, 8), dpi=80)
    if use_dark_background:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    if not resultsdir:
        resultsdir = os.path.join(os.path.dirname(path_dlc_face), 'mouseflow_'+os.path.basename(path_vid_face).split('.')[0])
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    showseconds = 10
    showtimepoints = int(FaceCam_FPS) * showseconds
    midpoint = int(showtimepoints/2)
    facedata = face.iloc[index-midpoint:index+generate_frames+midpoint][cols_to_plot].apply(zscore).rolling(smooth_data).mean()
    facedatapoints = shift5(facedata.values[:], -index)
    minvalue = min(np.array(facedata.quantile(q=.01)) - np.array(range(len(facedata.columns))))
    maxvalue = max(np.array(facedata.quantile(q=.99)) - np.array(range(len(facedata.columns))))
    cmap = plt.get_cmap('Dark2')

    with tqdm(total=vidlength) as pbar:
        while facemp4.isOpened():
            # shift X window of data
            axd['bottom'].cla()
            axd['bottom'].set_xlim(0, showtimepoints);
            axd['bottom'].set_xticks(ticks=np.arange(0, showtimepoints, int(FaceCam_FPS)));
            axd['bottom'].set_xticklabels([str(a) for a in range(-5, 5)]);
            axd['bottom'].set_ylim(minvalue, maxvalue)
            axd['bottom'].set_yticks(np.arange(-len(facedata.columns)+1, 1))
            axd['bottom'].set_yticklabels(reversed(cols_to_plot), fontsize=12, fontweight='bold')
            facedatapointsshifted = shift5(facedatapoints, midpoint-index)
            for i in range(len(facedatapoints.transpose())):
                axd['bottom'].axhline(y=-i, linestyle='--', color=cmap(i), alpha=.5, linewidth=1.2)
                axd['bottom'].plot(facedatapointsshifted.transpose()[i] - i, color=cmap(i))
                # plt.legend(facenames, loc=2)
            if use_dark_background:
                axd['bottom'].axvline(x=midpoint, color='white')
            else:
                axd['bottom'].axvline(x=midpoint, color='black')
            j = len(face.columns)-1
            for i in axd['bottom'].get_yticklabels():
                i.set_color(cmap(j))
                j -= 1
            axd['bottom'].set_xlabel('Time [sec]')

            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            # plot optical flow face frame
            axd['upper left'].cla()
            previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(current_frame_gray)
            gpu_previous = cv2.cuda_GpuMat()
            gpu_previous.upload(previous_frame_gray)
            flow = gpu_flow.calc(gpu_frame, gpu_previous, None)
            flow_color = flow_vis.flow_to_color(flow.download(), convert_to_bgr=True)
            if blend_gray_optflow:
                dst = cv2.addWeighted(current_frame, blend_gray_optflow, flow_color, 1-blend_gray_optflow, 0)
            else:
                dst = flow_color
            axd['upper left'].imshow(dst)
            axd['upper left'].axis('off')
            # plt.text(10, 30, str(params['paths']['Data_FaceCam'][-22:]), color='w', size=8)
            axd['upper left'].text(10, 30, "%06d" % index, color='w')

            # plot face regions
            for i in range(len(facemasks)):
                axd['upper left'].contour(facemasks[i], colors=[cmap(i+2)], linestyles='--', size=5)

            # plot face frame
            axd['upper centre'].cla()
            axd['upper centre'].imshow(current_frame_gray, cmap='gray')
            axd['upper centre'].axis('off')
            # plt.text(10, 30, str(params['paths']['Data_FaceCam'][-22:]), color='w', size=8)
            axd['upper centre'].text(10, 30, "%06d" % index, color='w')

            # plot fix points
            for key in face_anchor:
                axd['upper centre'].scatter(face_anchor[key][0], face_anchor[key][1], s=200, color='w', alpha=.5)

            # plot skeleton
            axd['upper centre'].plot([face_anchor.nosetip[0], face_anchor.forehead[0]], [face_anchor.nosetip[1], face_anchor.forehead[1]], color='w')
            axd['upper centre'].plot([face_anchor.nosetip[0], face_anchor.mouthtip[0]], [face_anchor.nosetip[1], face_anchor.mouthtip[1]], color='w')
            axd['upper centre'].plot([face_anchor.nosetip[0], face_anchor.tearduct[0]], [face_anchor.nosetip[1], face_anchor.tearduct[1]], color='w')
            axd['upper centre'].plot([face_anchor.mouthtip[0], face_anchor.tearduct[0]], [face_anchor.mouthtip[1], face_anchor.tearduct[1]], color='w')
            axd['upper centre'].plot([face_anchor.eyelid_bottom[0], face_anchor.chin[0]], [face_anchor.eyelid_bottom[1], face_anchor.chin[1]], color='w')

            # plot dynamic DLC points whole face
            for i in range(np.uint8(len(dlc_face.columns)/3)-6):
                if dlc_face.values[index][i*3+2] > dlc_conf_thresh:
                    axd['upper centre'].scatter(dlc_face.values[index][i*3], dlc_face.values[index][i*3+1], alpha=0.7, s=100, color='w')
            axd['upper centre'].scatter(pupil['x_raw'][index], pupil['y_raw'][index], alpha=0.7, s=30, color='w')
            axd['upper centre'].set_xlim(0, facemp4.get(3))
            axd['upper centre'].set_ylim(facemp4.get(4), 0)

            # plot eye detail
            axd['upper right'].cla()
            pupilmeanx = np.round(np.mean(pupil['x_smooth'])).astype(int)
            pupilmeany = np.round(np.mean(pupil['y_smooth'])).astype(int)
            axd['upper right'].imshow(current_frame_gray[pupilmeany-90:pupilmeany+90, pupilmeanx-120:pupilmeanx+120], cmap='gray')
            axd['upper right'].axis('off')

            # plot eye DLC points
            for i in range(4, np.uint8(len(dlc_face.columns)/3)):
                axd['upper right'].scatter(dlc_face.values[index][i*3]-(pupilmeanx-120), dlc_face.values[index][i*3+1]-(pupilmeany-90), alpha=1, s=100, color='w')
            axd['upper right'].plot([(dlc_face.values[index][12]-(pupilmeanx-120)), (dlc_face.values[index][15]-(pupilmeanx-120))], [(dlc_face.values[index][13]-(pupilmeany-90)), (dlc_face.values[index][16]-(pupilmeany-90))], color=cmap(1))
            axd['upper right'].plot([(dlc_face.values[index][12]-(pupilmeanx-120)), (dlc_face.values[index][18]-(pupilmeanx-120))], [(dlc_face.values[index][13]-(pupilmeany-90)), (dlc_face.values[index][19]-(pupilmeany-90))], color=cmap(1))
            axd['upper right'].scatter(pupil['x_raw'][index]-(pupilmeanx-120), pupil['y_raw'][index]-(pupilmeany-90), alpha=0.7, s=100, color='w')

            # plot pupil
            pupilplot = plt.Circle((np.round(pupil['x_interp'][index]).astype(int)-(pupilmeanx-120), np.round(pupil['y_interp'][index]).astype(int)-(pupilmeany-90)), np.round(pupil['diam_interp'][index]).astype(int), color=cmap(0), alpha=0.5, fill=False, linewidth=3)
            axd['upper right'].add_artist(pupilplot)

            plt.tight_layout()
            plt.savefig(os.path.join(resultsdir, "file%06d.png" % index))

            pbar.update(1)
            index += 1
            if index > vidlength:
                break
            ret, current_frame = facemp4.read()
            if current_frame is None:
                break
    facemp4.release()



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

