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

