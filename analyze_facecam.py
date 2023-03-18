#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse face camera data, extracting pupil data and motion energy of distinct face regions

Created on Wed May  8 14:31:51 2019
@author: Oliver Barnstedt
"""

import sys
import glob
import os.path
import pandas as pd
import fun.imh_cameras as imh_cameras
import h5py
import fun.imh_base as imh_base
import numpy as np
import yaml
import datetime
import logging

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
else:
    behaviour_store_keys = {}

if 'camtriggers' in params['treadmill']:
    sweeps = int(params['xlsmeta']['Sweeps'].split('_')[1]) - int(params['xlsmeta']['Sweeps'].split('_')[0]) + 1
else:
    sweeps = 1

# #  Only continue if face file does not exist yet
# if sum(["face" in s for s in behaviour_store_keys]) == sweeps+1:
#     sys.exit("Face data already analysed. Skipping ahead.")
# else:
#     print("Extracting facial features from FaceCam DLC labels...")
if 'Chew' in pd.read_hdf(params['paths']['Results_Behaviour'], 'face'):
    sys.exit("Face data already recently analysed. Skipping ahead.")


#  LOAD FACE DLC OUTPUT
try:
    dlc_facepath = glob.glob(os.path.join(params['paths']['Results_DLC_Face'], '') + '*.h5')[0]
except:
    sys.exit("Could not find face camera DLC labels.")
dlc_face = pd.read_hdf(dlc_facepath, mode='r')

# Inferring face camera triggers
if 'camtriggers' in params['treadmill']:
    startsweep = int(params['xlsmeta']['Sweeps'].split('_')[0])
    triggersum = np.cumsum(params['treadmill']['camtriggers'])
    facetriggersum = []
    cum_deviations = [0]
    if params['cameras']['infer_sweep_frames'] and params['cameras']['FaceCam_FrameTotal'] < sum(params['treadmill']['camtriggers']):
        print("Checking camera trigger frame count accuracy...")
        params['treadmill']['facecamtriggers'] = params['treadmill']['camtriggers']
        total_frame_diff = imh_cameras.facemotion(params['paths']['Data_FaceCam'], masks=[], total=True)
        total_frame_diff_diff = pd.Series(total_frame_diff[:, 0]).diff()
        for idx, triggers in enumerate(params['treadmill']['camtriggers']):
            if idx < len(params['treadmill']['camtriggers'])-1:
                slice = np.arange((triggersum[idx] - int(params['cameras']['infer_sweep_maxframes']) - cum_deviations[-1]), (triggersum[idx] + int(params['cameras']['infer_sweep_maxframes']) - cum_deviations[-1]))
                facetriggersum.append(total_frame_diff_diff[slice].idxmax() - 1)
                cum_deviations.append(triggersum[idx] - facetriggersum[-1])
        facetriggersum.append(params['cameras']['FaceCam_FrameTotal'])
        params['treadmill']['facecamtriggers'] = [facetriggersum[0]] + np.diff(facetriggersum).astype(int).tolist()
        print("Recorded triggers: {}".format(params['treadmill']['camtriggers']))
        print("Inferred face camera frames: {}".format(params['treadmill']['facecamtriggers']))
        print("Deviation: {}".format(
            [x - y for x, y in zip(params['treadmill']['facecamtriggers'], params['treadmill']['camtriggers'])]))
        print('Updating', params['paths']['yaml_experiment'])
        yaml.dump(params, open(params['paths']['yaml_experiment'], "w"), default_flow_style=False)
        imh_cameras.plot_sweep_inference(params, 'face', triggersum, facetriggersum, total_frame_diff_diff)
    else:
        facetriggersum = triggersum
        params['treadmill']['facecamtriggers'] = [facetriggersum[0]] + np.diff(facetriggersum).tolist()


# Pupil
if 1==0:#sum(["pupil" in s for s in behaviour_store_keys]) == sweeps:
    print("Pupil already calculated. Skipping.")
    if sweeps == 1:
        pupil = pd.read_hdf(params['paths']['Results_Behaviour'], key='pupil')
else:
    print("Calculating pupil...")
    if 'camtriggers' in params['treadmill']:
        for idx, triggers in enumerate(params['treadmill']['facecamtriggers']):
            pupil = imh_cameras.pupilextraction(dlc_face.iloc[facetriggersum[idx]-triggers:facetriggersum[idx]], params['cameras']['pupil_conf_thresh'], round(params['cameras']['pupil_smooth_window'] * params['cameras']['FaceCam_FPS']), round(params['cameras']['pupil_interpolation_limit'] * params['cameras']['FaceCam_FPS']), params['cameras']['pupil_na_limit'])
            pupil.to_hdf(params['paths']['Results_Behaviour'], key='w{:02d}/pupil/'.format(idx+startsweep))
            eyeliddist, eyeblinks = imh_cameras.eyeblink(dlc_face.iloc[facetriggersum[idx]-triggers:facetriggersum[idx]], params['cameras']['eyelid_conf_thresh'],
                                              round(params['cameras']['eyelid_smooth_window'] * params['cameras'][
                                                  'FaceCam_FPS']),
                                              round(params['cameras']['eyelid_interpolation_limit'] * params['cameras'][
                                                  'FaceCam_FPS']))
            eyeliddist.to_hdf(params['paths']['Results_Behaviour'], key='w{:02d}/eyeliddist/'.format(idx+startsweep))
            eyeblinks.to_hdf(params['paths']['Results_Behaviour'], key='w{:02d}/eyeblinks/'.format(idx+startsweep))
    else:
        face = pd.DataFrame()
        pupil = imh_cameras.pupilextraction(dlc_face, params['cameras']['pupil_conf_thresh'], round(params['cameras']['pupil_smooth_window'] * params['cameras']['FaceCam_FPS']), round(params['cameras']['pupil_interpolation_limit'] * params['cameras']['FaceCam_FPS']), params['cameras']['pupil_na_limit'])
        pupil.to_hdf(params['paths']['Results_Behaviour'], key='pupil')
        face['PupilDiam'] = pupil['diam_z']
        face['PupilX'] = pupil['x_raw']
        face['PupilY'] = pupil['y_raw']
        face['EyeLidDist'], face['EyeBlinks'] = imh_cameras.eyeblink(dlc_face, params['cameras']['eyelid_conf_thresh'], round(params['cameras']['eyelid_smooth_window'] * params['cameras']['FaceCam_FPS']), round(params['cameras']['eyelid_interpolation_limit'] * params['cameras']['FaceCam_FPS']))


# Face region motion
hfg = h5py.File(params['paths']['Results_Behaviour'], 'a')
if 1==1:#'facemasks' not in hfg.keys():
    facemasks, face_anchor = imh_cameras.faceregions(params, dlc_face)
    face_anchor.to_hdf(params['paths']['Results_Behaviour'], key='face_anchor')
    #hfg.create_dataset('facemasks', data=facemasks)
else:
    facemasks = [np.array(hfg['facemasks'])[i] for i in range(np.array(hfg['facemasks']).shape[0])]
hfg.close()

if 'camtriggers' in params['treadmill']:
    for idx, triggers in enumerate(params['treadmill']['facecamtriggers']):
        print("Sweep {}/{}...".format(idx+startsweep, int(params['xlsmeta']['Sweeps'].split('_')[-1])))
        frame_diff = imh_cameras.facemotion(params['paths']['Data_FaceCam'], facemasks, round(
            params['cameras']['faceregion_smooth_window'] * params['cameras']['FaceCam_FPS']),
                                            videoslice=np.arange(facetriggersum[idx] - triggers, facetriggersum[idx]).tolist())
        pupil = pd.read_hdf(params['paths']['Results_Behaviour'], key='w{:02d}/pupil'.format(idx+startsweep))
        face = pd.DataFrame()
        face['PupilDiam'] = pupil['diam_raw']
        face['PupilX'] = pupil['x_raw']
        face['PupilY'] = pupil['y_raw']
        face['EyeLidDist'], face['EyeBlinks'] = pd.read_hdf(params['paths']['Results_Behaviour'], key='w{:02d}/eyeliddist'.format(idx+startsweep))
        face = pd.concat([face, frame_diff], axis=1)[:triggers]
        face.to_hdf(params['paths']['Results_Behaviour'], key='w{:02d}/face'.format(idx+startsweep))
else:
    face_motion = imh_cameras.facemotion(params['paths']['Data_FaceCam'], facemasks)
    whisk_freq = imh_cameras.freq_analysis2(face_motion['OFang_Whiskerpad'], params['cameras']['FaceCam_FPS'], rollwin=params['cameras']['FaceCam_FPS'], min_periods=int(params['cameras']['FaceCam_FPS']*.67))
    sniff_freq = imh_cameras.freq_analysis2(face_motion['OFang_Nose'], params['cameras']['FaceCam_FPS'], rollwin=params['cameras']['FaceCam_FPS'], min_periods=int(params['cameras']['FaceCam_FPS']*.67))
    chewenv, chew = imh_cameras.hilbert_peaks(face_motion['OFang_Mouth'], params['cameras']['FaceCam_FPS'])
    face_freq = pd.DataFrame({'Whisking_freq': whisk_freq, 'Sniff_freq': sniff_freq, 'Chewing_Envelope': chewenv, 'Chew': chew})
    face = pd.concat([face, face_motion, face_freq], axis=1)[:params['cameras']['FaceCam_FrameTotal']]
    face.to_hdf(params['paths']['Results_Behaviour'], key='face')

# Create labeled video from face data
# if params['cameras']['face_output'] and not 'camtriggers' in params['treadmill']:
#     imh_cameras.create_labeled_video_face(params, dlc_face, pupil, face, facemasks, face_anchor)
