#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import deeplabcut
import matplotlib.pyplot as plt
import sys
import os
import glob
import pandas as pd
import fun.imh_cameras as imh_cameras
import h5py
import numpy as np
from deepgraphpose.models.fitdgp_util import get_snapshot_path
from deepgraphpose.models.eval import estimate_pose, plot_dgp

plt.interactive(False)

#  To evade cuDNN error message:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.logging.set_verbosity(tf.logging.ERROR)

#  scan video files
dir = sys.argv[1]
faceID = sys.argv[2]
bodyID = sys.argv[3]

# dir = '/mnt/ag-remy-2/Oliver/2018_03_proxSub-reward/207-12/training'
# faceID = '22632089'
# bodyID = '22611474'
dir_out = '/mnt/ag-remy-2/Oliver/2018_03_proxSub-reward/207-12/training/DLC'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
filetype = '.mp4'
DLC_FaceConfig = '/media/oliver/Oliver_SSD/MouseFace-Barnstedt-2019-08-21/config.yaml'
DLC_BodyConfig = '/media/oliver/Oliver_SSD/MouseBody-Barnstedt-2019-09-09/config.yaml'
facefiles = glob.glob(dir+'/*'+faceID+'*'+filetype)
bodyfiles = glob.glob(dir+'/*'+bodyID+'*'+filetype)
facevid_output = True
bodyvid_output = True


#  BODY DEEPGRAPHPOSE
snapshot_path, cfg_yaml = get_snapshot_path('snapshot-step2-final--0', os.path.dirname(DLC_BodyConfig), shuffle=1)

for bodyfile in bodyfiles:
    print("Applying ", DLC_BodyConfig, " on BODY video: ", bodyfile)
    estimate_pose(proj_cfg_file=DLC_BodyConfig,
                  dgp_model_file=str(snapshot_path),
                  video_file=bodyfile,
                  output_dir=dir_out,
                  shuffle=1,
                  save_pose=True,
                  save_str='',
                  new_size=None)
    print("DGP body labels saved in ", dir_out)

#  FACE DEEPGRAPHPOSE
snapshot_path, cfg_yaml = get_snapshot_path('snapshot-step2-final--0', os.path.dirname(DLC_FaceConfig), shuffle=1)

for facefile in facefiles:
    print("Applying ", DLC_FaceConfig, " on FACE video: ", facefile)
    estimate_pose(proj_cfg_file=DLC_FaceConfig,
                  dgp_model_file=str(snapshot_path),
                  video_file=facefile,
                  output_dir=dir_out,
                  shuffle=1,
                  save_pose=True,
                  save_str='',
                  new_size=None)
    print("DGP face labels saved in ", dir_out)


#  FACE ANALYSIS
pupil_conf_thresh = .99
pupil_smooth_window = 1
FaceCam_FPS = 75
pupil_interpolation_limit = 2
pupil_na_limit = .25
eyelid_conf_thresh = .999
eyelid_smooth_window = 1
eyelid_interpolation_limit = 2
faceregion_smooth_window = .333
params = {'cameras': {'faceregion_conf_thresh': .99,
                      'faceregion_size_whiskers': 1,
                      'faceregion_size_nose': 1.5,
                      'faceregion_size_mouth': 1,
                      'faceregion_size_cheek': 1},
          'paths': {'Results_Cam_Dir': dir_out}}

faceDLCs = glob.glob(dir_out + '/*' + faceID + '*labeled.h5')
for faceDLC in faceDLCs:
    if os.path.exists(faceDLC[:-4] + '_analysis.h5'):
        print('Analysis file already found. Skipping to next...')
    else:
        print('Processing DLC data from '+faceDLC)
        facefile = dir+'/'+os.path.basename(faceDLC).split('DeepCut_resnet')[0]+filetype
        params['paths']['Data_FaceCam'] = facefile
        dlc_face = pd.read_hdf(faceDLC, mode='r')
        FaceCam_FrameTotal = len(dlc_face)
        face = pd.DataFrame()
        pupil = imh_cameras.pupilextraction(dlc_face, pupil_conf_thresh, round(pupil_smooth_window * FaceCam_FPS), round(pupil_interpolation_limit * FaceCam_FPS), pupil_na_limit)
        pupil.to_hdf(faceDLC[:-4]+'_analysis.h5', key='pupil')
        face[['PupilDiam', 'PupilX', 'PupilY', 'Saccades']] = pupil[['diam_z', 'x_raw', 'y_raw', 'saccades']]
        face['EyeLidDist'], face['EyeBlinks'] = imh_cameras.eyeblink(dlc_face, eyelid_conf_thresh, round(eyelid_smooth_window * FaceCam_FPS), round(eyelid_interpolation_limit * FaceCam_FPS))

        # Face region motion
        facemasks, face_anchor = imh_cameras.faceregions(params, dlc_face)
        face_anchor.to_hdf(faceDLC[:-4]+'_analysis.h5', key='face_anchor')
        hfg = h5py.File(faceDLC[:-4]+'_analysis.h5', 'a')
        try:
            hfg.create_dataset('facemasks', data=facemasks)
        except:
            None
        hfg.close()

        frame_diff = imh_cameras.facemotion(facefile, facemasks, round(faceregion_smooth_window * FaceCam_FPS))
        face = pd.concat([face, frame_diff], axis=1)[:FaceCam_FrameTotal]
        face.to_hdf(faceDLC[:-4]+'_analysis.h5', key='face')


#  BODY ANALYSIS
bodyDLCs = glob.glob(dir_out + '/*' + bodyID + '*labeled.h5')
for bodyDLC in bodyDLCs:
    bodyfile = dir + '/' + os.path.basename(faceDLC).split('DeepCut_resnet')[0] + filetype
    dlc_body = pd.read_hdf(bodyDLC, mode='r')
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
    stride_freq = imh_cameras.freq_analysis(rightpaws_fbdiff, params['cameras']['FaceCam_FPS'], M=128)
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
    cylinder_motion_raw, cylinder_motion = imh_cameras.cylinder_motion(params['paths']['Data_BodyCam'], cylinder_mask)
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
    body.to_hdf(bodyDLC[:-4]+'_analysis.h5', key='body')
