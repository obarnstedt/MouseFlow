#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import glob
import pandas as pd
import fun.imh_cameras as imh_cameras
import h5py
import numpy as np
import curl
import cv2

plt.interactive(False)

#  To evade cuDNN error message:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.logging.set_verbosity(tf.logging.ERROR)

def mouseflow(dir=os.getcwd(), facekey='', bodykey='', dgp=True, batch=True, of_type='farneback', filetype='.mp4',
              dlc_faceyaml='', dlc_bodyyaml='', pupil_conf_thresh = .99, pupil_smooth_window = 1, 
              pupil_interpolation_limit = 2, pupil_na_limit = .25, eyelid_conf_thresh = .999, eyelid_smooth_window = 1,
              eyelid_interpolation_limit = 2, faceregion_smooth_window = .333, faceregion_conf_thresh = .99,
              faceregion_size_whiskers=1, faceregion_size_nose=1.5, faceregion_size_mouth=1, faceregion_size_cheek=1):
    # dir defines directory to detect face/body videos, standard: current working directory
    # facekey defines unique string that is contained in all face videos. If none, no face videos will be considered.
    # bodykey defines unique string that is contained in all face videos. If none, no face videos will be considered.
    # dgp defines whether to use DeepGraphPose (if True), otherwise resorts to DLC
    # batch defines how many videos to analyse (True for all, integer for the first n videos)
    # of_type sets the optical flow algorithm

    dir_out = os.path.join(dir, 'mouseflow')
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    dlc_faceyaml = '/media/oliver/Oliver_SSD/MouseFace-Barnstedt-2019-08-21/config.yaml'
    dlc_bodyyaml = '/media/oliver/Oliver_SSD/MouseBody-Barnstedt-2019-09-09/config.yaml'

    facefiles = glob.glob(dir+'/*'+facekey+'*'+filetype)
    bodyfiles = glob.glob(dir+'/*'+bodykey+'*'+filetype)

    for facefile in facefiles:
        #  FACE DEEPGRAPHPOSE / DEEPLABCUT
        print("Applying ", dlc_faceyaml, " on FACE video: ", facefile)
        if dgp:
            from deepgraphpose.models.fitdgp_util import get_snapshot_path
            from deepgraphpose.models.eval import estimate_pose
            snapshot_path, _ = get_snapshot_path('snapshot-step2-final--0', os.path.dirname(dlc_faceyaml), shuffle=1)
            estimate_pose(proj_cfg_file=dlc_faceyaml,
                        dgp_model_file=str(snapshot_path),
                        video_file=facefile,
                        output_dir=dir_out,
                        shuffle=1,
                        save_pose=True,
                        save_str='',
                        new_size=None)
        else:
            import deeplabcut
            deeplabcut.analyze_videos(config=dlc_faceyaml, videos=[facefile], shuffle=1, videotype=filetype, destfolder=dir_out)
        print("DGP/DLC face labels saved in ", dir_out)

        #  FACE ANALYSIS
        faceDLC = glob.glob(os.path.join(dir_out, os.path.basename(facefile)[:-4] + '*labeled.h5'))
        print('Processing DLC data from '+faceDLC)
        facevidcap = cv2.VideoCapture(facefile)
        FaceCam_FPS = facevidcap.get(cv2.CAP_PROP_FPS)
        dlc_face = pd.read_hdf(faceDLC, mode='r')
        FaceCam_FrameTotal = len(dlc_face)
        face = pd.DataFrame()
        pupil = imh_cameras.pupilextraction(dlc_face, pupil_conf_thresh, round(pupil_smooth_window * FaceCam_FPS), round(pupil_interpolation_limit * FaceCam_FPS), pupil_na_limit)
        pupil.to_hdf(faceDLC, key='pupil')
        face[['PupilDiam', 'PupilX', 'PupilY', 'Saccades']] = pupil[['diam_z', 'x_raw', 'y_raw', 'saccades']]
        face['EyeLidDist'], face['EyeBlinks'] = imh_cameras.eyeblink(dlc_face, eyelid_conf_thresh, round(eyelid_smooth_window * FaceCam_FPS), round(eyelid_interpolation_limit * FaceCam_FPS))

        # Face region motion
        facemasks, face_anchor = imh_cameras.faceregions(params, dlc_face)
        face_anchor.to_hdf(faceDLC, key='face_anchor')
        hfg = h5py.File(faceDLC, 'a')
        try:
            hfg.create_dataset('facemasks', data=facemasks)
        except:
            None
        hfg.close()

        frame_diff = imh_cameras.facemotion(facefile, facemasks, round(faceregion_smooth_window * FaceCam_FPS))
        face = pd.concat([face, frame_diff], axis=1)[:FaceCam_FrameTotal]
        face.to_hdf(faceDLC, key='face')


    for bodyfile in bodyfiles:
        #  BODY DEEPGRAPHPOSE / DEEPLABCUT
        print("Applying ", dlc_bodyyaml, " on BODY video: ", bodyfile)
        if dgp:
            snapshot_path, _ = get_snapshot_path('snapshot-step2-final--0', os.path.dirname(dlc_bodyyaml), shuffle=1)
            estimate_pose(proj_cfg_file=dlc_bodyyaml,
                        dgp_model_file=str(snapshot_path),
                        video_file=bodyfile,
                        output_dir=dir_out,
                        shuffle=1,
                        save_pose=True,
                        save_str='',
                        new_size=None)
        else:
            deeplabcut.analyze_videos(config=dlc_bodyyaml, videos=[bodyfile], shuffle=1, videotype=filetype, destfolder=dir_out)
        print("DGP/DLC body labels saved in ", dir_out)

    #  BODY ANALYSIS
    bodyDLC = glob.glob(os.path.join(dir_out, os.path.basename(bodyfile)[:-4] + '*labeled.h5'))
    dlc_body = pd.read_hdf(bodyDLC, mode='r')
    bodyvidcap = cv2.VideoCapture(bodyfile)
    BodyCam_FPS = bodyvidcap.get(cv2.CAP_PROP_FPS)
    BodyCam_FrameTotal = len(dlc_body)
    motion_frontpaw_raw, motion_frontpaw, motionangle_frontpaw = imh_cameras.dlc_pointmotion(dlc_body.values[:, 12],
                                                                                            dlc_body.values[:, 13],
                                                                                            dlc_body.values[:, 14])
    motion_backpaw_raw, motion_backpaw, motionangle_backpaw = imh_cameras.dlc_pointmotion(dlc_body.values[:, 24],
                                                                                        dlc_body.values[:, 25],
                                                                                        dlc_body.values[:, 26])
    frontpaw_lrdiff = imh_cameras.dlc_pointdistance(dlc_body.values[:, [9, 10, 11]], dlc_body.values[:, [18, 19, 20]])
    backpaw_lrdiff = imh_cameras.dlc_pointdistance(dlc_body.values[:, [21, 22, 23]], dlc_body.values[:, [30, 31, 32]])
    rightpaws_fbdiff = imh_cameras.dlc_pointdistance(dlc_body.values[:, [12, 13, 14]], dlc_body.values[:, [24, 25, 26]])
    stride_freq = imh_cameras.freq_analysis(rightpaws_fbdiff, BodyCam_FPS, M=128)
    motion_mouth_raw, motion_mouth, motionangle_mouth = imh_cameras.dlc_pointmotion(dlc_body.values[:, 3],
                                                                                    dlc_body.values[:, 4],
                                                                                    dlc_body.values[:, 5])
    angle_tail_3, angle_tail = imh_cameras.dlc_angle(dlc_body.values[:, [33, 34, 35]], dlc_body.values[:, [36, 37, 38]],
                                                    dlc_body.values[:, [39, 40, 41]])
    angle_paws_front_3, angle_paws_front = imh_cameras.dlc_angle(dlc_body.values[:, [9, 10, 11]],
                                                                dlc_body.values[:, [12, 13, 14]],
                                                                dlc_body.values[:, [15, 16, 17]])
    angle_paws_back_3, angle_paws_back = imh_cameras.dlc_angle(dlc_body.values[:, [21, 22, 23]],
                                                            dlc_body.values[:, [24, 25, 26]],
                                                            dlc_body.values[:, [27, 28, 29]])
    tailroot_level = -pd.Series(
        (dlc_body.values[:, 34] - np.nanmean(dlc_body.values[:, 34])) / np.nanstd(dlc_body.values[:, 34]))
    cylinder_mask = np.zeros([582, 782])
    cylinder_mask[int(np.nanpercentile(dlc_body.values[:, 22], 99) + 30):, :250] = 1
    cylinder_motion_raw, cylinder_motion = imh_cameras.cylinder_motion(bodyfile, cylinder_mask)
    body = pd.DataFrame({'PointMotion_FrontPaw': motion_frontpaw_raw, 'AngleMotion_FrontPaw': motionangle_frontpaw,
                        'PointMotion_Mouth': motion_mouth_raw, 'AngleMotion_Mouth': motionangle_mouth,
                        'PointMotion_BackPaw': motion_backpaw_raw, 'AngleMotion_BackPaw': motionangle_backpaw,
                        'Angle_Tail_3': angle_tail_3,
                        'Angle_Tail': angle_tail, 'Angle_Paws_Front_3': angle_paws_front_3,
                        'Angle_Paws_Front': angle_paws_front,
                        'Angle_Paws_Back': angle_paws_back, 'Angle_Paws_Back_3': angle_paws_back_3,
                        'Tailroot_Level': tailroot_level,
                        'Cylinder_Motion': cylinder_motion_raw, 'Stride_Frequency': stride_freq})
    body = body[:BodyCam_FrameTotal]
    body.to_hdf(bodyDLC, key='body')
