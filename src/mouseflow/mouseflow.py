#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use('TKAgg')
import glob
import os

# import gdown
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore

import mouseflow.body_processing as body_processing
import mouseflow.face_processing as face_processing
from mouseflow.utils import motion_processing
from mouseflow.utils.preprocess_video import flip_vid
from mouseflow import apply_models

plt.interactive(False)



def runDLC(vid_dir=os.getcwd(), facekey='', bodykey='', dgp=True, batch=True, overwrite=False, 
           filetype='.mp4', vid_output=1000, bodyflip=False, faceflip=False, models_dir='', 
           facemodel_name='MouseFace-Barnstedt-2019-08-21', bodymodel_name='MouseBody-Barnstedt-2019-09-09'):
    # vid_dir defines directory to detect face/body videos, standard: current working directory
    # facekey defines unique string that is contained in all face videos. If none, no face videos will be considered.
    # bodykey defines unique string that is contained in all body videos. If none, no body videos will be considered.
    # dgp defines whether to use DeepGraphPose (if True), otherwise resorts to DLC
    # batch defines how many videos to analyse (True for all, integer for the first n videos)

    import tensorflow as tf

    #  To evade cuDNN error message:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.logging.set_verbosity(tf.logging.ERROR)

    # check if DGP is working, otherwise resort to DLC
    if dgp:
        try:
            import deepgraphpose
        except ImportError as e:
            print('DGP import error; working with DLC...')
            dgp = False

    # set directories
    if os.path.isdir(vid_dir):
        dir_out = os.path.join(vid_dir, 'mouseflow')
    else:
        dir_out = os.path.join(os.path.dirname(vid_dir), 'mouseflow')
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # check where marker models are located, download if not present 
    dlc_faceyaml, dlc_bodyyaml = apply_models.download_models(models_dir, facemodel_name, bodymodel_name)

    # identify video files
    if facekey==True:
        facefiles = [vid_dir]
        bodyfiles = []
    elif bodykey==True:
        facefiles = []
        bodyfiles = [vid_dir]
    elif facekey=='' or facekey==False or facekey==None:
        facefiles = []
        bodyfiles = glob.glob(os.path.join(vid_dir,'*'+bodykey+'*'+filetype))
    elif bodykey=='' or bodykey==False or bodykey==None:
        bodyfiles = []
        facefiles = glob.glob(os.path.join(vid_dir,'*'+facekey+'*'+filetype))
    else:
        facefiles = glob.glob(os.path.join(vid_dir,'*'+facekey+'*'+filetype))
        bodyfiles = glob.glob(os.path.join(vid_dir,'*'+bodykey+'*'+filetype))

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

    # Apply DLC/DGP Model to each face video
    for facefile in facefiles:
        if glob.glob(os.path.join(dir_out, os.path.basename(facefile)[:-4]+'*.h5')) and not overwrite:
            print(f'Video {os.path.basename(facefile)} already labelled. Skipping ahead...')
        else:
            print("Applying ", dlc_faceyaml, " on FACE video: ", facefile)
            if dgp:
                apply_models.apply_dgp(dlc_faceyaml, dir_out, facefile, vid_output)
            else:
                print(f"Generating labeled face video...")
                apply_models.apply_dlc(filetype, vid_output, dlc_faceyaml, dir_out, facefile)
                print("DLC face labels saved in ", dir_out)

    # Apply DLC/DGP Model to each body video
    for bodyfile in bodyfiles:
        if glob.glob(os.path.join(dir_out, os.path.basename(bodyfile)[:-4]+'*.h5')) and not overwrite:
            print(f'Video {os.path.basename(bodyfile)} already labelled. Skipping ahead...')
        else:
            print("Applying ", dlc_bodyyaml, " on BODY video: ", bodyfile)          
            if dgp:
                apply_models.apply_dgp(dlc_bodyyaml, dir_out, bodyfile)
                print("DGP body labels saved in ", dir_out)
            else:
                print(f"Generating labeled face video...")
                apply_models.apply_dlc(filetype, vid_output, dlc_bodyyaml, dir_out, bodyfile)
                print("DLC body labels saved in ", dir_out)


def runMF(dlc_dir=os.getcwd(), overwrite=False, dgp=True,
             pupil_conf_thresh = .99, pupil_smooth_window = 1, 
              pupil_interpolation_limit = 2, pupil_na_limit = .25, eyelid_conf_thresh = .999, 
              eyelid_smooth_window = 1, eyelid_interpolation_limit = 2, 
              faceregion_conf_thresh = .99, faceregion_size_whiskers=1, faceregion_size_nose=1, 
              faceregion_size_mouth=1, faceregion_size_cheek=1, conf_thresh=None):
    # dir defines directory to detect face/body videos, standard: current working directory
    # facekey defines unique string that is contained in all face videos. If none, no face videos will be considered.
    # bodykey defines unique string that is contained in all face videos. If none, no face videos will be considered.
    # dgp defines whether to use DeepGraphPose (if True), otherwise resorts to DLC
    # batch defines how many videos to analyse (True for all, integer for the first n videos)
    # of_type sets the optical flow algorithm

    # TODO: go through DGP files if requested
    facefiles = glob.glob(os.path.join(dlc_dir,'*MouseFace*1030000.h5'))
    bodyfiles = glob.glob(os.path.join(dlc_dir,'*MouseBody*1030000.h5'))

    for faceDLC in facefiles:
        
        #  FACE ANALYSIS
        if os.path.exists(faceDLC_analysis) and not overwrite:
            print(faceDLC_analysis + ' data already analysed. Skipping ahead...')
        else:
            print('Processing DLC data from '+faceDLC)
            faceDLC_analysis = faceDLC[:-3] + '_mouseflow.h5'
            facefile = glob.glob(os.path.join(os.path.dirname(dlc_dir), os.path.basename(faceDLC).split('DLC')[0] + '*'))[0]
            facevidcap = cv2.VideoCapture(facefile)
            FaceCam_FPS = facevidcap.get(cv2.CAP_PROP_FPS)
            FaceCam_FrameCount = int(facevidcap.get(cv2.CAP_PROP_FRAME_COUNT))

            # reading in DLC/DGP file
            facemarkers = pd.read_hdf(faceDLC, mode='r')

            #TODO: process unlikely markers before further processing
            if not conf_thresh:
                conf_thresh = 0.5 if dgp else 0.99

            face = pd.DataFrame()
            pupil = face_processing.pupilextraction(
                facemarkers, 
                pupil_conf_thresh, 
                round(pupil_smooth_window * FaceCam_FPS), 
                round(pupil_interpolation_limit * FaceCam_FPS), 
                pupil_na_limit,
            )
            if not type(pupil)==float:
                pupil.to_hdf(faceDLC_analysis, key='pupil')
                face[['PupilDiam', 'PupilX', 'PupilY', 'Saccades']] = pupil[['diam_z', 'x_raw', 'y_raw', 'saccades']]
            face['EyeLidDist'], face['EyeBlinks'] = face_processing.eyeblink(
                facemarkers, 
                eyelid_conf_thresh, 
                round(eyelid_smooth_window * FaceCam_FPS), 
                round(eyelid_interpolation_limit * FaceCam_FPS)
            )

            # Face region motion
            facemasks, face_anchor = face_processing.faceregions(
                facemarkers, 
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
                face_motion = face_processing.facemotion_nocuda(facefile, facemasks)
                face = pd.concat([face, face_motion], axis=1)[:FaceCam_FrameCount]
            else:
                face_motion = face_processing.facemotion(facefile, facemasks)
                whisk_freq = motion_processing.freq_analysis2(face_motion['OFang_Whiskerpad'], FaceCam_FPS, rollwin=FaceCam_FPS, min_periods=int(FaceCam_FPS*.67))
                sniff_freq = motion_processing.freq_analysis2(face_motion['OFang_Nose'],       FaceCam_FPS, rollwin=FaceCam_FPS, min_periods=int(FaceCam_FPS*.67))
                chewenv, chew = motion_processing.hilbert_peaks(face_motion['OFang_Mouth'],    FaceCam_FPS)
                face_freq = pd.DataFrame({'Whisking_freq': whisk_freq, 'Sniff_freq': sniff_freq, 'Chewing_Envelope': chewenv, 'Chew': chew})
                face = pd.concat([face, face_motion, face_freq], axis=1)[:FaceCam_FrameCount]
            face.to_hdf(faceDLC_analysis, key='face')

    for bodyDLC in bodyfiles:

        #  BODY ANALYSIS

        # Load Body Data
        body_markers = pd.read_hdf(bodyDLC, mode='r')
        body_markers.columns = body_markers.columns.droplevel(0)

        bodyfile = glob.glob(os.path.join(os.path.dirname(dlc_dir), os.path.basename(bodyDLC).split('DLC')[0] + '*'))[0]
        BodyCam_FPS = cv2.VideoCapture(bodyfile).get(cv2.CAP_PROP_FPS)
        BodyCam_width = cv2.VideoCapture(bodyfile).get(cv2.CAP_PROP_FRAME_WIDTH)
        BodyCam_height = cv2.VideoCapture(bodyfile).get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Paw motion
        motion_frontpaw = body_processing.dlc_pointmotion(body_markers['paw_front-right2', 'x'], body_markers['paw_front-right2', 'y'], body_markers['paw_front-right2', 'likelihood'])
        motion_backpaw =  body_processing.dlc_pointmotion(body_markers['paw_back-right2', 'x'],  body_markers['paw_back-right2', 'y'],  body_markers['paw_back-right2', 'likelihood'])
        
        # Paw angles
        angle_paws_front = body_processing.dlc_angle(body_markers['paw_front-right1'], body_markers['paw_front-right2'], body_markers['paw_front-right3'])
        angle_paws_back =  body_processing.dlc_angle(body_markers['paw_back-right1'],  body_markers['paw_back-right2'],  body_markers['paw_back-right3'] )

        # Stride and gait information
        rightpaws_fbdiff = body_processing.dlc_pointdistance(body_markers['paw_front-right2'], body_markers['paw_back-right2'])
        stride_freq = body_processing.freq_analysis(rightpaws_fbdiff, BodyCam_FPS, M=128)

        # Mouth motion
        motion_mouth = body_processing.dlc_pointmotion(body_markers['mouth', 'x'], body_markers['mouth', 'y'], body_markers['mouth', 'likelihood'])
        
        # Tail information
        angle_tail =       body_processing.dlc_angle(body_markers['tail1'], body_markers['tail2'], body_markers['tail3'])
        tailroot_level = -zscore(body_markers['tail1', 'y'])

        cylinder_mask = np.zeros([BodyCam_height, BodyCam_width])
        cylinder_mask[int(np.nanpercentile(body_markers['paw_back-right1', 'y'].values, 99) + 30):, :int(BodyCam_width/3)] = 1
        cylinder_motion = body_processing.cylinder_motion(bodyfile, cylinder_mask)

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
        body = body[:len(body_markers)]
        body.to_hdf(bodyDLC, key='body')
