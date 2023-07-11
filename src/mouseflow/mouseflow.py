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
from mouseflow.utils import config_tensorflow, is_installed, motion_processing, confidence_na
from mouseflow.utils.preprocess_video import flip_vid

matplotlib.use('TKAgg')
plt.interactive(False)


def runDLC(models_dir, vid_dir=os.getcwd(), facekey='face', bodykey='body', dgp=True, batch='all', overwrite=False,
           filetype='.mp4', vid_output=1000, bodyflip=False, faceflip=False,
           facemodel_name='MouseFace-Barnstedt-2019-08-21', bodymodel_name='MouseBody-Barnstedt-2019-09-09'):
    # vid_dir defines directory to detect face/body videos, standard: current working directory
    # facekey defines unique string that is contained in all face videos. If None, no face videos will be considered.
    # bodykey defines unique string that is contained in all body videos. If None, no body videos will be considered.
    # dgp defines whether to use DeepGraphPose (if True), otherwise resorts to DLC
    # batch defines how many videos to analyse ('all' for all, integer for the first n videos)

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

    # sort out already flipped videos
    facefiles = [f for f in facefiles if '_flipped.*' not in f]
    bodyfiles = [b for b in bodyfiles if '_flipped.*' not in b]

    # flipping videos
    if faceflip:
        facefiles_flipped = []
        for vid in facefiles:
            facefiles_flipped.append(flip_vid(vid, horizontal=True))
        facefiles = facefiles_flipped
    if bodyflip:
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

    # TODO: go through DGP files if requested
    facefiles = glob.glob(os.path.join(dlc_dir, '*MouseFace*1030000.h5'))
    bodyfiles = glob.glob(os.path.join(dlc_dir, '*MouseBody*1030000.h5'))

    if (len(facefiles) + len(bodyfiles)) == 0:
        print(
            f'No marker files found in directory {dlc_dir}. Check directory.')
    else:
        print(
            f'Found the following marker files: \n {[str(f) for f in facefiles]} \n {[str(f) for f in bodyfiles]}')

    for faceDLC in facefiles:

        #  FACE ANALYSIS
        if os.path.exists(faceDLC_analysis) and not overwrite:
            print(faceDLC_analysis + ' data already analysed. Skipping ahead...')
        else:
            print('Processing DLC data from '+faceDLC)
            faceDLC_analysis = faceDLC[:-3] + '_mouseflow.h5'
            facefile = glob.glob(os.path.join(os.path.dirname(
                dlc_dir), os.path.basename(faceDLC).split('DLC')[0] + '*'))[0]
            facevidcap = cv2.VideoCapture(facefile)
            FaceCam_FPS = facevidcap.get(cv2.CAP_PROP_FPS)
            FaceCam_FrameCount = int(facevidcap.get(cv2.CAP_PROP_FRAME_COUNT))

            # reading in DLC/DGP file
            markers_face = pd.read_hdf(faceDLC, mode='r')
            markers_face.columns = markers_face.columns.droplevel(0)

            # TODO: process unlikely markers before further processing
            # filling low-confidence markers with NAN
            markers_face_conf = confidence_na(dgp, conf_thresh, markers_face)

            # interpolating missing data up to <na_limits>
            interpolation_limits_frames = {x: int(k * FaceCam_FPS)
                                for (x, k) in interpolation_limits_sec.items()}
            markers_face_conf[['pupil'+str(n+1) for n in range(6)]] = \
                markers_face_conf[['pupil'+str(n+1) for n in range(6)]].interpolate(
                    method='linear', limit=interpolation_limits_frames['pupil'])

            # creating and populating 'face' dataframe
            face = pd.DataFrame()
            pupil = face_processing.pupilextraction(markers_face_conf[['pupil'+str(n+1) for n in range(6)]].values)

            face['EyeLidDist'], face['EyeBlinks'] = face_processing.eyeblink(
                markers_face,
                eyelid_conf_thresh,
                round(eyelid_smooth_window * FaceCam_FPS),
                round(eyelid_interpolation_limit * FaceCam_FPS)
            )

            # Face region motion
            facemasks, face_anchor = face_processing.faceregions(
                markers_face,
                facefile,
                faceregion_conf_thresh,
                faceregion_size_whiskers,
                faceregion_size_nose,
                faceregion_size_mouth,
                faceregion_size_cheek,
                dlc_file=faceDLC
            )
            face_anchor.to_hdf(faceDLC_analysis, key='face_anchor')
            hfg = h5py.File(faceDLC_analysis, 'a')
            try:
                hfg.create_dataset('facemasks', data=facemasks)
            except:
                pass
            hfg.close()

            if cv2.cuda.getCudaEnabledDeviceCount() == 0:
                print("No CUDA support detected. Processing without optical flow...")
                face_motion = face_processing.facemotion_nocuda(
                    facefile, facemasks)
                face = pd.concat([face, face_motion], axis=1)[
                    :FaceCam_FrameCount]
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
                face = pd.concat([face, face_motion, face_freq], axis=1)[
                    :FaceCam_FrameCount]
                
            # NA LIMITS
        #         if np.mean(pupil_diam_raw.isna()) > pupil_na_limit:  # if too many missing values present, fill everything with NANs
        # pupil = np.nan
            # SMOOTHING
            # ZSCORING

                # pupil_xydist_interp = pupil_xydist_raw.interpolate(method='linear', limit=pupil_interpolation_limit)  # linear interpolation
    # pupil_xydist_smooth = pd.Series(smooth(pupil_xydist_interp, window_len=round(pupil_smooth_window/10))).shift(
    #     periods=-int(pupil_smooth_window/10 / 2))
    # pupil_xydist_z = ((pupil_xydist_interp - pupil_xydist_interp.mean()) / pupil_xydist_interp.std(ddof=0))[:len(pupil_x_raw)]

        # pupil_saccades = pd.Series(pupil_x_smooth.diff().abs() > 1.5)


            face['PupilDiam'] = pupil.pupil_diam_raw
            face['PupilX'] = pupil.pupil_x_raw
            face['PupilY'] = pupil.pupil_y_raw
            face['PupilMotion'] = pupil.pupil_xydist_raw

            face.to_hdf(faceDLC_analysis, key='face')




    for bodyDLC in bodyfiles:

        #  BODY ANALYSIS

        # Load Body Data
        markers_body = pd.read_hdf(bodyDLC, mode='r')
        markers_body.columns = markers_body.columns.droplevel(0)

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

        body = pd.DataFrame({
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
        body = body[:len(markers_body)]
        body.to_hdf(bodyDLC, key='body')
