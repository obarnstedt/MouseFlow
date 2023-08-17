#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os

# import gdown
import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore

import mouseflow.body_processing as body_processing
import mouseflow.face_processing as face_processing
from mouseflow import apply_models
from mouseflow.utils import config_tensorflow, is_installed, motion_processing, confidence_na, process_raw_data
from mouseflow.utils.preprocess_video import flip_vid, crop_vid

matplotlib.use('TKAgg')
plt.interactive(False)


def runDLC(models_dir, vid_dir=os.getcwd(), facekey='face', bodykey='body', dgp=True, batch='all', overwrite=False,
           filetype='.mp4', vid_output=1000, body_facing='right', face_facing='left', face_crop=[], body_crop=[],
           facemodel_name='MouseFace-Barnstedt-2019-08-21', bodymodel_name='MouseBody-Barnstedt-2019-09-09'):
    # vid_dir defines directory to detect face/body videos, standard: current working directory
    # facekey defines unique string that is contained in all face videos. If None, no face videos will be considered.
    # bodykey defines unique string that is contained in all body videos. If None, no body videos will be considered.
    # dgp defines whether to use DeepGraphPose (if True), otherwise resorts to DLC
    # batch defines how many videos to analyse ('all' for all, integer for the first n videos)
    # face/body_crop allows initial cropping of video in the form [x_start, x_end, y_start, y_end]

    #  To evade cuDNN error message:
    config_tensorflow(log_level='ERROR', allow_growth=True)

    # check if DGP is working, otherwise resort to DLC
    if dgp == True and not is_installed('deepgraphpose'):
        print('DGP import error; working with DLC...')
        dgp = False

    # check where marker models are located, download if not present
    dlc_faceyaml, dlc_bodyyaml = apply_models.download_models(
        models_dir, facemodel_name, bodymodel_name)

    # identify video files
    facefiles = []
    bodyfiles = []
    if os.path.isfile(vid_dir):
        if facekey in vid_dir:
            facefiles = [vid_dir]
        elif bodykey in vid_dir:
            bodyfiles = [vid_dir]
        else:
            print(
                f'Need to pass <facekey> or <bodykey> argument to classify video {vid_dir}.')
    if facekey == True:
        facefiles = [vid_dir]
    elif bodykey == True:
        bodyfiles = [vid_dir]
    elif facekey == '' or facekey == False or facekey == None:
        bodyfiles = glob.glob(os.path.join(vid_dir, '*'+bodykey+'*'+filetype))
    elif bodykey == '' or bodykey == False or bodykey == None:
        facefiles = glob.glob(os.path.join(vid_dir, '*'+facekey+'*'+filetype))
    else:
        facefiles = glob.glob(os.path.join(vid_dir, '*'+facekey+'*'+filetype))
        bodyfiles = glob.glob(os.path.join(vid_dir, '*'+bodykey+'*'+filetype))

    # cropping videos
    facefiles = [f for f in facefiles if '_cropped.*' not in f]  # sort out already cropped videos
    bodyfiles = [b for b in bodyfiles if '_cropped.*' not in b]  # sort out already cropped videos
    if face_crop:
        facefiles_cropped = []
        for vid in facefiles:
            facefiles_cropped.append(crop_vid(vid, face_crop))
        facefiles = facefiles_cropped
    if body_crop:
        bodyfiles_cropped = []
        for vid in bodyfiles:
            bodyfiles_cropped.append(crop_vid(vid, body_crop))
        bodyfiles = bodyfiles_cropped

    # flipping videos
    facefiles = [f for f in facefiles if '_flipped.*' not in f]  # sort out already flipped videos
    bodyfiles = [b for b in bodyfiles if '_flipped.*' not in b]  # sort out already flipped videos
    if face_facing != 'left':
        facefiles_flipped = []
        for vid in facefiles:
            facefiles_flipped.append(flip_vid(vid, horizontal=True))
        facefiles = facefiles_flipped
    if body_facing != 'right':
        bodyfiles_flipped = []
        for vid in bodyfiles:
            bodyfiles_flipped.append(flip_vid(vid, horizontal=True))
        bodyfiles = bodyfiles_flipped

    # batch mode (if user specifies a number n, it will only process the first n files)
    try:
        batch = int(batch)
        facefiles = facefiles[:batch]
        bodyfiles = bodyfiles[:batch]
        print(f'Only processing first {batch} face and body videos...')
    except ValueError:
        pass

    # set directories
    if os.path.isdir(vid_dir):
        dir_out = os.path.join(vid_dir, 'mouseflow')
    else:
        dir_out = os.path.join(os.path.dirname(vid_dir), 'mouseflow')
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # Apply DLC/DGP Model to each face video
    for facefile in facefiles:
        if glob.glob(os.path.join(dir_out, os.path.basename(facefile)[:-4]+'*.h5')) and not overwrite:
            print(
                f'Video {os.path.basename(facefile)} already labelled. Skipping ahead...')
        else:
            print("Applying ", dlc_faceyaml, " on FACE video: ", facefile)
            if dgp:
                apply_models.apply_dgp(
                    dlc_faceyaml, dir_out, facefile, vid_output)
            else:
                apply_models.apply_dlc(
                    filetype, vid_output, dlc_faceyaml, dir_out, facefile, overwrite)

    # Apply DLC/DGP Model to each body video
    for bodyfile in bodyfiles:
        if glob.glob(os.path.join(dir_out, os.path.basename(bodyfile)[:-4]+'*.h5')) and not overwrite:
            print(
                f'Video {os.path.basename(bodyfile)} already labelled. Skipping ahead...')
        else:
            print("Applying ", dlc_bodyyaml, " on BODY video: ", bodyfile)
            if dgp:
                apply_models.apply_dgp(dlc_bodyyaml, dir_out, bodyfile)
            else:
                apply_models.apply_dlc(
                    filetype, vid_output, dlc_bodyyaml, dir_out, bodyfile, overwrite)


def runMF(dlc_dir=os.getcwd(),
          overwrite=False,
          dgp=True,
          conf_thresh=None,
          interpolation_limits_sec={
              'pupil': 2,
              'eyelid': 1,
          },
          smoothing_windows_sec={
              'PupilDiam': 1,
              'PupilMotion': 0.25,
              'eyelid': 0.1,
              'MotionEnergy': 0.25,
          },
          na_limit=0.25,
          faceregions_sizes={
                'whiskers': 1,
                'nose': 1,
                'mouth': 1,
                'cheek': 1,
          }
    ):
    # dir defines directory to detect face/body videos, standard: current working directory
    # facekey defines unique string that is contained in all face videos. If none, no face videos will be considered.
    # bodykey defines unique string that is contained in all face videos. If none, no face videos will be considered.
    # dgp defines whether to use DeepGraphPose (if True), otherwise resorts to DLC
    # batch defines how many videos to analyse (True for all, integer for the first n videos)
    # of_type sets the optical flow algorithm

    # TODO: go through DGP files if requested, required beforehand: common naming convention!
    facefiles = glob.glob(os.path.join(dlc_dir, '*MouseFace*1030000.h5'))
    bodyfiles = glob.glob(os.path.join(dlc_dir, '*MouseBody*1030000.h5'))

    if (len(facefiles) + len(bodyfiles)) == 0:
        print(
            f'No marker files found in directory {dlc_dir}. Check directory.')
    else:
        print(
            f'Found the following marker files: \n {[str(f) for f in facefiles]} \n \
            {[str(f) for f in bodyfiles]}')

    #  FACE ANALYSIS
    for faceDLC in facefiles:
        if os.path.exists(mf_file) and not overwrite:
            print(mf_file + ' data already analysed. Skipping ahead...')
            continue

        print('Processing DLC data from '+faceDLC)
        mf_file = faceDLC[:-3] + '_mouseflow.h5'
        facefile = glob.glob(os.path.join(os.path.dirname(
            dlc_dir), os.path.basename(faceDLC).split('DLC')[0] + '*'))[0]
        facevidcap = cv2.VideoCapture(facefile)
        FaceCam_FPS = facevidcap.get(cv2.CAP_PROP_FPS)

        # Reading in DLC/DGP file
        markers_face = pd.read_hdf(faceDLC, mode='r')
        markers_face.columns = markers_face.columns.droplevel(0)

        # Filling low-confidence markers with NAN
        markers_face_conf = confidence_na(dgp, conf_thresh, markers_face)

        # Interpolating missing data up to <na_limits>
        interpolation_limits_frames = {x: int(k * FaceCam_FPS)
                                       for (x, k) in interpolation_limits_sec.items()}
        markers_face_conf.loc[:, ['pupil'+str(n+1) for n in range(6)]] = \
            markers_face_conf.loc[:, ['pupil'+str(n+1) for n in range(6)]].interpolate(
                method='linear', limit=interpolation_limits_frames['pupil'])

        # Extracting pupil and eyelid data
        pupil_raw = face_processing.pupilextraction(
            markers_face_conf[['pupil'+str(n+1) for n in range(6)]].values)
        eyelid_dist_raw = pd.Series(motion_processing.dlc_pointdistance2(
            markers_face_conf['eyelid1'], markers_face_conf['eyelid2']), name='EyeLidDist')

        # Define and save face regions
        facemasks, face_anchor = face_processing.define_faceregions(
            markers_face_conf, facefile, faceregions_sizes, faceDLC)
        face_anchor.to_hdf(mf_file, key='face_anchor')
        hfg = h5py.File(mf_file, 'a')
        hfg.create_dataset('facemasks', data=facemasks)
        hfg.close()

        # Extract motion in face regions
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            print("No CUDA support detected. Processing without optical flow...")
            face_motion = face_processing.facemotion_nocuda(
                facefile, facemasks)
            face_raw = pd.concat([pupil_raw, eyelid_dist_raw, face_motion], axis=1)
        else:
            face_motion = face_processing.facemotion(facefile, facemasks)
            whisk_freq = motion_processing.freq_analysis2(
                face_motion['OFang_Whiskerpad'], FaceCam_FPS, rollwin=FaceCam_FPS, min_periods=int(FaceCam_FPS*.67))
            sniff_freq = motion_processing.freq_analysis2(
                face_motion['OFang_Nose'],       FaceCam_FPS, rollwin=FaceCam_FPS, min_periods=int(FaceCam_FPS*.67))
            chewenv, chew = motion_processing.hilbert_peaks(
                face_motion['OFang_Mouth'],    FaceCam_FPS)
            face_freq = pd.DataFrame(
                {'Whisking_freq': whisk_freq, 'Sniff_freq': sniff_freq, 'Chewing_Envelope': chewenv, 'Chew': chew})
            face_raw = pd.concat([pupil_raw, eyelid_dist_raw, face_motion, face_freq], axis=1)

        # further process raw data and save
        face = process_raw_data(smoothing_windows_sec, na_limit, FaceCam_FPS, interpolation_limits_frames, face_raw)
        face.to_hdf(mf_file, key='face')

    #  BODY ANALYSIS
    for bodyDLC in bodyfiles:

        # Load Body Data
        markers_body = pd.read_hdf(bodyDLC, mode='r')
        markers_body.columns = markers_body.columns.droplevel(0)

        # Get body video info
        bodyfile = glob.glob(os.path.join(os.path.dirname(
            dlc_dir), os.path.basename(bodyDLC).split('DLC')[0] + '*'))[0]
        BodyCam_FPS = cv2.VideoCapture(bodyfile).get(cv2.CAP_PROP_FPS)
        BodyCam_width = cv2.VideoCapture(
            bodyfile).get(cv2.CAP_PROP_FRAME_WIDTH)
        BodyCam_height = cv2.VideoCapture(
            bodyfile).get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Paw motion
        motion_frontpaw = body_processing.dlc_pointmotion(
            markers_body['paw_front-right2', 'x'], markers_body['paw_front-right2', 'y'], markers_body['paw_front-right2', 'likelihood'])
        motion_backpaw = body_processing.dlc_pointmotion(
            markers_body['paw_back-right2', 'x'],  markers_body['paw_back-right2', 'y'],  markers_body['paw_back-right2', 'likelihood'])

        # Paw angles
        angle_paws_front = body_processing.dlc_angle(
            markers_body['paw_front-right1'], markers_body['paw_front-right2'], markers_body['paw_front-right3'])
        angle_paws_back = body_processing.dlc_angle(
            markers_body['paw_back-right1'],  markers_body['paw_back-right2'],  markers_body['paw_back-right3'])

        # Stride and gait information
        rightpaws_fbdiff = body_processing.dlc_pointdistance(
            markers_body['paw_front-right2'], markers_body['paw_back-right2'])
        stride_freq = body_processing.freq_analysis(
            rightpaws_fbdiff, BodyCam_FPS, M=128)

        # Mouth motion
        motion_mouth = body_processing.dlc_pointmotion(
            markers_body['mouth', 'x'], markers_body['mouth', 'y'], markers_body['mouth', 'likelihood'])

        # Tail information
        angle_tail = body_processing.dlc_angle(
            markers_body['tail1'], markers_body['tail2'], markers_body['tail3'])
        tailroot_level = -zscore(markers_body['tail1', 'y'])

        cylinder_mask = np.zeros([BodyCam_height, BodyCam_width])
        cylinder_mask[int(np.nanpercentile(
            markers_body['paw_back-right1', 'y'].values, 99) + 30):, :int(BodyCam_width/3)] = 1
        cylinder_motion = body_processing.cylinder_motion(
            bodyfile, cylinder_mask)

        body_raw = pd.DataFrame({
            'PointMotion_FrontPaw': motion_frontpaw.raw_distance,
            'AngleMotion_FrontPaw': motion_frontpaw.angles,
            'PointMotion_Mouth': motion_mouth.raw_distance,
            'AngleMotion_Mouth': motion_mouth.angles,
            'PointMotion_BackPaw': motion_backpaw.raw_distance,
            'AngleMotion_BackPaw': motion_backpaw.angles,
            'Angle_Tail_3': angle_tail.angle3,
            'Angle_Tail': angle_tail.slope,
            'Angle_Paws_Front_3': angle_paws_front.angle3,
            'Angle_Paws_Front': angle_paws_front.slope,
            'Angle_Paws_Back_3': angle_paws_back.angle3,
            'Angle_Paws_Back': angle_paws_back.slope,
            'Tailroot_Level': tailroot_level,
            'Cylinder_Motion': cylinder_motion.raw,
            'Stride_Frequency': stride_freq,
        })

        # further process raw data and save
        body = process_raw_data(smoothing_windows_sec, na_limit, FaceCam_FPS, interpolation_limits_frames, body_raw)
        body.to_hdf(mf_file, key='body')