#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse body camera data

Created on Wed May  8 14:31:51 2019
@author: Oliver Barnstedt
"""

import sys
import glob
import os.path
sys.path.append(os.path.join(os.getcwd(), 'fun'))
import pandas as pd
import fun.imh_cameras as imh_cameras
import numpy as np
import fun.imh_base as imh_base
import yaml

#  LOAD EXPERIMENT YAML
params = imh_base.readyaml(sys.argv[1], sys.argv[2])

#  check what's contained in behaviour.h5
if os.path.exists(params['paths']['Results_Behaviour']):
    try:
        behaviour_store = pd.HDFStore(params['paths']['Results_Behaviour'], 'r')
        behaviour_store_keys = behaviour_store.keys()
        behaviour_store.close()
    except:
        behaviour_store_keys = {}

if 'camtriggers' in params['treadmill']:
    sweeps = int(params['xlsmeta']['Sweeps'].split('_')[1]) - int(params['xlsmeta']['Sweeps'].split('_')[0]) + 1
else:
    sweeps = 1

#  Only continue if body file does not exist yet
if 'Gait_x' in pd.read_hdf(params['paths']['Results_Behaviour'], 'body'):#sum(["body" in s for s in behaviour_store_keys]) == sweeps:
    sys.exit("Body data already recently analysed. Skipping ahead.")
else:
    print("Extracting body features from BodyCam DLC labels...")
    #  LOAD BODY DLC OUTPUT
    try:
        dlc_bodypath = glob.glob(os.path.join(params['paths']['Results_DLC_Body'], '') + '*labeled.h5')[0]
    except:
        sys.exit("Could not find body camera DLC labels.")
    dlc_body = pd.read_hdf(dlc_bodypath, mode='r')
    dlc_body.columns = pd.MultiIndex.from_arrays(
        [dlc_body.columns.get_level_values(1), dlc_body.columns.get_level_values(2)], names=dlc_body.columns.names[1:])

    # Inferring body camera triggers
    if 'camtriggers' in params['treadmill']:
        startsweep = int(params['xlsmeta']['Sweeps'].split('_')[0])
        triggersum = np.cumsum(params['treadmill']['camtriggers'])
        bodytriggersum = []
        cum_deviations = [0]
        if params['cameras']['infer_sweep_frames'] and params['cameras']['BodyCam_FrameTotal'] < sum(params['treadmill']['camtriggers']):
            print("Checking camera trigger frame count accuracy...")
            params['treadmill']['bodycamtriggers'] = params['treadmill']['camtriggers']
            total_frame_diff = imh_cameras.facemotion(params['paths']['Data_BodyCam'], masks=[], total=True)
            total_frame_diff_diff = pd.Series(total_frame_diff[:, 0]).diff()
            for idx, triggers in enumerate(params['treadmill']['camtriggers']):
                if idx < len(params['treadmill']['camtriggers'])-1:
                    slice = np.arange((triggersum[idx] - int(params['cameras']['infer_sweep_maxframes']) - cum_deviations[-1]), (triggersum[idx] + int(params['cameras']['infer_sweep_maxframes']) - cum_deviations[-1]))
                    bodytriggersum.append(total_frame_diff_diff[slice].idxmax() - 1)
                    cum_deviations.append(triggersum[idx] - bodytriggersum[-1])
            bodytriggersum.append(params['cameras']['BodyCam_FrameTotal'])
            params['treadmill']['bodycamtriggers'] = [int(bodytriggersum[0])] + np.diff(bodytriggersum).tolist()
            print("Recorded triggers: {}".format(params['treadmill']['camtriggers']))
            print("Inferred body camera frames: {}".format(params['treadmill']['bodycamtriggers']))
            print("Deviation: {}".format(
                [x - y for x, y in zip(params['treadmill']['bodycamtriggers'], params['treadmill']['camtriggers'])]))
            print('Updating', params['paths']['yaml_experiment'])
            yaml.dump(params, open(params['paths']['yaml_experiment'], "w"), default_flow_style=False)
            imh_cameras.plot_sweep_inference(params, 'body', triggersum, bodytriggersum, total_frame_diff_diff)
        else:
            bodytriggersum = triggersum
            params['treadmill']['bodycamtriggers'] = [bodytriggersum[0]] + np.diff(bodytriggersum).tolist()

        dlc_body_total = dlc_body.copy()
        for idx, triggers in enumerate(params['treadmill']['bodycamtriggers']):
            print("Sweep {}/{}...".format(idx + startsweep, int(params['xlsmeta']['Sweeps'].split('_')[-1])))
            dlc_body = dlc_body_total.iloc[bodytriggersum[idx] - triggers:bodytriggersum[idx]]
            motion_frontpaw = imh_cameras.dlc_pointmotion(dlc_body.values[:, 12], dlc_body.values[:, 13],
                                                          dlc_body.values[:, 14])
            motion_mouth = imh_cameras.dlc_pointmotion(dlc_body.values[:, 3], dlc_body.values[:, 4],
                                                       dlc_body.values[:, 5])
            print("Calculating tail angle...")
            angle_tail_3, angle_tail = imh_cameras.dlc_angle(dlc_body.values[:, [33, 34, 35]],
                                                             dlc_body.values[:, [36, 37, 38]],
                                                             dlc_body.values[:, [39, 40, 41]])
            print("Calculating front paw angle...")
            angle_paws_front_3, angle_paws_front = imh_cameras.dlc_angle(dlc_body.values[:, [9, 10, 11]],
                                                                         dlc_body.values[:, [12, 13, 14]],
                                                                         dlc_body.values[:, [15, 16, 17]])
            print("Calculating back paw angle...")
            angle_paws_back_3, angle_paws_back = imh_cameras.dlc_angle(dlc_body.values[:, [21, 22, 23]],
                                                                       dlc_body.values[:, [24, 25, 26]],
                                                                       dlc_body.values[:, [27, 28, 29]])
            tailroot_level = -pd.Series(
                (dlc_body.values[:, 34] - np.nanmean(dlc_body.values[:, 34])) / np.nanstd(dlc_body.values[:, 34]))
            body = pd.DataFrame({'PointMotion_FrontPaw': motion_frontpaw, 'PointMotion_Mouth': motion_mouth,
                                 'Angle_Tail_3': angle_tail_3,
                                 'Angle_Tail': angle_tail, 'Angle_Paws_Front_3': angle_paws_front_3,
                                 'Angle_Paws_Front': angle_paws_front,
                                 'Angle_Paws_Back': angle_paws_back, 'Angle_Paws_Back_3': angle_paws_back_3,
                                 'Tailroot_Level': tailroot_level})
            body = body[:triggers]
            body.to_hdf(params['paths']['Results_Behaviour'], key='w{:02d}/body/'.format(idx + startsweep))
    else:
        motion_frontpaw_raw, motion_frontpaw, motionangle_frontpaw = imh_cameras.dlc_pointmotion(dlc_body['paw_front-right2'])
        motion_backpaw_raw, motion_backpaw, motionangle_backpaw = imh_cameras.dlc_pointmotion(dlc_body['paw_back-right2'])

        frontpaws_lrdiff_raw, frontpaws_lrdiff = imh_cameras.dlc_pointdistance(dlc_body['paw_front-right2'], dlc_body['paw_front-left'])
        backpaws_lrdiff_raw, backpaws_lrdiff = imh_cameras.dlc_pointdistance(dlc_body['paw_back-right2'], dlc_body['paw_back-left'])
        rightpaws_fbdiff_raw, rightpaws_fbdiff = imh_cameras.dlc_pointdistance(dlc_body['paw_front-right2'], dlc_body['paw_back-right2'])

        stride_freq = imh_cameras.freq_analysis(rightpaws_fbdiff, params['cameras']['FaceCam_FPS'], M=128)
        frontpaw_stride_freq = imh_cameras.freq_analysis2(dlc_body['paw_front-right2'], params['cameras']['BodyCam_FPS'], rollwin=params['cameras']['BodyCam_FPS'], min_periods=int(params['cameras']['BodyCam_FPS']*.67))
        backpaw_stride_freq = imh_cameras.freq_analysis2(dlc_body['paw_back-right2'], params['cameras']['BodyCam_FPS'], rollwin=params['cameras']['BodyCam_FPS'], min_periods=int(params['cameras']['BodyCam_FPS']*.67))

        frontpaws_phasecorr = imh_cameras.dlc_pointphasecorr(dlc_body['paw_front-right2'], dlc_body['paw_front-left'])
        backpaws_phasecorr = imh_cameras.dlc_pointphasecorr(dlc_body['paw_back-right2'], dlc_body['paw_back-left'])
        rightpaws_phasecorr = imh_cameras.dlc_pointphasecorr(dlc_body['paw_front-right2'], dlc_body['paw_back-right2'])
        phasecorr_x, phasecorr_eq = imh_cameras.dlc_phasecorrX(dlc_body['paw_front-right2'], dlc_body['paw_front-left'],
                                                              dlc_body['paw_back-right2'], dlc_body['paw_back-left'])

        motion_mouth_raw, motion_mouth, motionangle_mouth = imh_cameras.dlc_pointmotion(dlc_body['mouth'])

        angle_tail_3, angle_tail = imh_cameras.dlc_angle(dlc_body['tail1'], dlc_body['tail2'], dlc_body['tail3'])
        angle_paws_front_3, angle_paws_front = imh_cameras.dlc_angle(dlc_body['paw_front-right1'], dlc_body['paw_front-right2'], dlc_body['paw_front-right3'])
        angle_paws_back_3, angle_paws_back = imh_cameras.dlc_angle(dlc_body['paw_back-right1'], dlc_body['paw_back-right2'], dlc_body['paw_back-right3'])

        tailroot_level = -pd.Series((dlc_body.values[:, 34] - np.nanmean(dlc_body.values[:, 34])) / np.nanstd(dlc_body.values[:, 34]))
        cylinder_mask = np.zeros([582, 782])
        cylinder_mask[int(np.nanpercentile(dlc_body.values[:, 22], 99)+30):, :250] = 1
        cylinder_motion_raw, cylinder_motion = imh_cameras.cylinder_motion(params['paths']['Data_BodyCam'], cylinder_mask)
        body = pd.DataFrame({'PointMotion_FrontPaw': motion_frontpaw_raw, 'AngleMotion_FrontPaw':motionangle_frontpaw,
                             'PointMotion_Mouth': motion_mouth_raw, 'AngleMotion_Mouth':motionangle_mouth,
                             'PointMotion_BackPaw': motion_backpaw_raw, 'AngleMotion_BackPaw':motionangle_backpaw,
                             'Angle_Tail_3': angle_tail_3,
                             'Angle_Tail': angle_tail, 'Angle_Paws_Front_3': angle_paws_front_3, 'Angle_Paws_Front': angle_paws_front,
                             'Angle_Paws_Back': angle_paws_back, 'Angle_Paws_Back_3': angle_paws_back_3, 'Tailroot_Level': tailroot_level,
                             'Cylinder_Motion': cylinder_motion_raw, 'Stride_Frequency': stride_freq,
                             'Gait_x': phasecorr_x, 'Gait_eq': phasecorr_eq,
                             'GaitCorr_front': frontpaws_phasecorr, 'GaitCorr_back': backpaws_phasecorr, 'GaitCorr_right': rightpaws_phasecorr,
                             'PawDistF_LR_raw': frontpaws_lrdiff_raw, 'PawDistF_LR': frontpaws_lrdiff, 'PawDistB_LR_raw': backpaws_lrdiff_raw, 'PawDistF_LR': backpaws_lrdiff,
                             'PawDistR_FB_raw': rightpaws_fbdiff_raw, 'PawDistR_FB': rightpaws_fbdiff, 'Stride_Frequency_F': frontpaw_stride_freq, 'Stride_Frequency_B': backpaw_stride_freq})
        body = body[:params['cameras']['BodyCam_FrameTotal']]
        body.to_hdf(params['paths']['Results_Behaviour'], key='body')

    #   if params['cameras']['body_output'] and not 'camtriggers' in params['treadmill']:
#        imh_cameras.create_labeled_video_body(params, body)

# Create labeled video from body data
# if params['cameras']['body_output'] and not 'camtriggers' in params['treadmill']:
#     body = pd.read_hdf(params['paths']['Results_Behaviour'], key='body')
#     imh_cameras.create_labeled_video_body(params, body)

