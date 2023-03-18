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
DLC_FaceConfig = '/mnt/ag-remy-2/Imaging/AnalysisTools/DLC/MouseFace-Barnstedt-2019-08-21/config.yaml'
DLC_BodyConfig = '/mnt/ag-remy-2/Imaging/AnalysisTools/DLC/MouseBody-Barnstedt-2019-09-09/config.yaml'
facefiles = glob.glob(dir+'/*'+faceID+'*'+filetype)
bodyfiles = glob.glob(dir+'/*'+bodyID+'*'+filetype)
facevid_output = True
bodyvid_output = True


#  FACE DEEPLABCUT
for facefile in facefiles:
    if not glob.glob(dir_out + '/' + os.path.basename(facefile)[:-4] + '*.h5'):  # check if H5 already exists
        print("Applying ", DLC_FaceConfig, " on FACE video: ", facefile)
        deeplabcut.analyze_videos(config=DLC_FaceConfig, videos=[facefile], shuffle=1, videotype=filetype, destfolder=dir_out)
        print("DLC face labels saved in ", dir_out)
    else:
        print("DLC Face H5 file already exists. Skipping ahead.")

    #  creating face DLC plots
    if glob.glob(dir_out + '/' + os.path.basename(facefile)[:-4] + '*.h5'):
        print("Plotting face DLC label trajectories...")
        deeplabcut.plot_trajectories(config=DLC_FaceConfig, videos=[facefile], showfigures=False, destfolder=dir_out+'/'+os.path.basename(facefile)[:-4])
        plt.close('all')

    #  making labelled DLC face video
    if facevid_output and not glob.glob(dir_out + '/' + os.path.basename(facefile)[:-4] + '*labeled.mp4'):
        print("Generating labeled face video...")
        deeplabcut.create_labeled_video(config=DLC_FaceConfig, videos=[facefile], draw_skeleton=False, destfolder=dir_out)


#  BODY DEEPLABCUT
for bodyfile in bodyfiles:
    if not glob.glob(dir_out + '/' + os.path.basename(bodyfile)[:-4] + '*.h5'):  # check if H5 already exists
        print("Applying ", DLC_BodyConfig, " on BODY video: ", bodyfile)
        deeplabcut.analyze_videos(config=DLC_BodyConfig, videos=[bodyfile], shuffle=1, videotype=filetype, destfolder=dir_out)
        print("DLC body labels saved in ", dir_out)
    else:
        print("DLC Body H5 file already exists. Skipping ahead.")

    #  creating body DLC plots
    if glob.glob(dir_out + '/' + os.path.basename(bodyfile)[:-4] + '*.h5'):
        print("Plotting body DLC label trajectories...")
        deeplabcut.plot_trajectories(config=DLC_BodyConfig, videos=[bodyfile], showfigures=False, destfolder=dir_out)
        plt.close('all')

    #  making labelled DLC body video
    if facevid_output and not glob.glob(dir_out + '/' + os.path.basename(bodyfile)[:-4] + '*labeled.mp4'):
        print("Generating labeled body video...")
        deeplabcut.create_labeled_video(config=DLC_BodyConfig, videos=[bodyfile], draw_skeleton=False, destfolder=dir_out)


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

faceDLCs = glob.glob(dir_out + '/*' + faceID + '*00.h5')
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
bodyDLCs = glob.glob(dir_out + '/*' + bodyID + '*.h5')
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
