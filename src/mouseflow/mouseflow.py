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

import mouseflow.body_processing as body_processing
import mouseflow.face_processing as face_processing
from mouseflow.utils import motion_processing
from mouseflow.utils.preprocess_video import flip_vid

plt.interactive(False)



def runDLC(vid_dir=os.getcwd(), facekey='', bodykey='', dgp=True, batch=True, overwrite=False, 
           filetype='.mp4', vid_output=1000, bodyflip=False, faceflip=False, dlc_faceyaml='', dlc_bodyyaml=''):
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

    # download_dlc()

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
            apply_facemodel(dgp, filetype, vid_output, dlc_faceyaml, dir_out, facefile)

    # Apply DLC/DGP Model to each body video
    for bodyfile in bodyfiles:
        if glob.glob(os.path.join(dir_out, os.path.basename(bodyfile)[:-4]+'*.h5')) and not overwrite:
            print(f'Video {os.path.basename(bodyfile)} already labelled. Skipping ahead...')
        else:
            print("Applying ", dlc_bodyyaml, " on BODY video: ", bodyfile)          
            apply_bodymodel(dgp, filetype, vid_output, dlc_bodyyaml, dir_out, bodyfile)


def runMF(dlc_dir=os.getcwd(),
            overwrite=False, of_type='farneback', vid_output=1000,
             pupil_conf_thresh = .99, pupil_smooth_window = 1, 
              pupil_interpolation_limit = 2, pupil_na_limit = .25, eyelid_conf_thresh = .999, 
              eyelid_smooth_window = 1, eyelid_interpolation_limit = 2, faceregion_smooth_window = .333, 
              faceregion_conf_thresh = .99, faceregion_size_whiskers=1, faceregion_size_nose=1, 
              faceregion_size_mouth=1, faceregion_size_cheek=1):
    # dir defines directory to detect face/body videos, standard: current working directory
    # facekey defines unique string that is contained in all face videos. If none, no face videos will be considered.
    # bodykey defines unique string that is contained in all face videos. If none, no face videos will be considered.
    # dgp defines whether to use DeepGraphPose (if True), otherwise resorts to DLC
    # batch defines how many videos to analyse (True for all, integer for the first n videos)
    # of_type sets the optical flow algorithm

    facefiles = glob.glob(os.path.join(dlc_dir,'*MouseFace*1030000.h5'))
    bodyfiles = glob.glob(os.path.join(dlc_dir,'*MouseBody*1030000.h5'))

    for faceDLC in facefiles:
        
        #  FACE ANALYSIS
        faceDLC_analysis = faceDLC[:-3] + '_analysis.h5'
        print('Processing DLC data from '+faceDLC)
        facefile = glob.glob(os.path.join(os.path.dirname(dlc_dir), os.path.basename(faceDLC).split('DLC')[0]+'*'))[0]
        facevidcap = cv2.VideoCapture(facefile)
        FaceCam_FPS = facevidcap.get(cv2.CAP_PROP_FPS)
        FaceCam_FrameCount = int(facevidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        dlc_face = pd.read_hdf(faceDLC, mode='r')
        FaceCam_FrameTotal = len(dlc_face)
        face = pd.DataFrame()
        pupil = face_processing.pupilextraction(dlc_face, pupil_conf_thresh, round(pupil_smooth_window * FaceCam_FPS), round(pupil_interpolation_limit * FaceCam_FPS), pupil_na_limit)
        if not type(pupil)==float:
            pupil.to_hdf(faceDLC_analysis, key='pupil')
            face[['PupilDiam', 'PupilX', 'PupilY', 'Saccades']] = pupil[['diam_z', 'x_raw', 'y_raw', 'saccades']]
        face['EyeLidDist'], face['EyeBlinks'] = face_processing.eyeblink(dlc_face, eyelid_conf_thresh, round(eyelid_smooth_window * FaceCam_FPS), round(eyelid_interpolation_limit * FaceCam_FPS))

        # Face region motion
        facemasks, face_anchor = face_processing.faceregions(dlc_face, facefile, faceregion_conf_thresh,
              faceregion_size_whiskers, faceregion_size_nose, faceregion_size_mouth, faceregion_size_cheek, dlc_file=faceDLC)
        face_anchor.to_hdf(faceDLC_analysis, key='face_anchor')
        hfg = h5py.File(faceDLC_analysis, 'a')
        try:
            hfg.create_dataset('facemasks', data=facemasks)
        except:
            None
        hfg.close()

        if cv2.cuda.getCudaEnabledDeviceCount()==0:
            print("No CUDA support detected. Processing without optical flow...")
            face_motion = face_processing.facemotion_nocuda(facefile, facemasks)
            face = pd.concat([face, face_motion], axis=1)[:FaceCam_FrameCount]
        else:
            face_motion = face_processing.facemotion(facefile, facemasks)
            whisk_freq = motion_processing.freq_analysis2(face_motion['OFang_Whiskerpad'], FaceCam_FPS, rollwin=FaceCam_FPS, min_periods=int(FaceCam_FPS*.67))
            sniff_freq = motion_processing.freq_analysis2(face_motion['OFang_Nose'], FaceCam_FPS, rollwin=FaceCam_FPS, min_periods=int(FaceCam_FPS*.67))
            chewenv, chew = motion_processing.hilbert_peaks(face_motion['OFang_Mouth'], FaceCam_FPS)
            face_freq = pd.DataFrame({'Whisking_freq': whisk_freq, 'Sniff_freq': sniff_freq, 'Chewing_Envelope': chewenv, 'Chew': chew})
            face = pd.concat([face, face_motion, face_freq], axis=1)[:FaceCam_FrameCount]
        face.to_hdf(faceDLC_analysis, key='face')

    for bodyDLC in bodyfiles:

        #  BODY ANALYSIS
        bodyDLC_analysis = faceDLC[:-3] + '_analysis.h5'
        dlc_body = pd.read_hdf(bodyDLC, mode='r')
        bodyfile = glob.glob(os.path.join(os.path.dirname(dlc_dir), os.path.basename(bodyDLC).split('DLC')[0]+'*'))[0]
        bodyvidcap = cv2.VideoCapture(bodyfile)
        BodyCam_FPS = bodyvidcap.get(cv2.CAP_PROP_FPS)
        BodyCam_FrameTotal = len(dlc_body)
        motion_frontpaw_raw, motion_frontpaw, motionangle_frontpaw = body_processing.dlc_pointmotion(dlc_body.values[:, 12],
                                                                                                dlc_body.values[:, 13],
                                                                                                dlc_body.values[:, 14])
        motion_backpaw_raw, motion_backpaw, motionangle_backpaw = body_processing.dlc_pointmotion(dlc_body.values[:, 24],
                                                                                            dlc_body.values[:, 25],
                                                                                            dlc_body.values[:, 26])
        frontpaw_lrdiff = body_processing.dlc_pointdistance(dlc_body.values[:, [9, 10, 11]], dlc_body.values[:, [18, 19, 20]])
        backpaw_lrdiff = body_processing.dlc_pointdistance(dlc_body.values[:, [21, 22, 23]], dlc_body.values[:, [30, 31, 32]])
        rightpaws_fbdiff = body_processing.dlc_pointdistance(dlc_body.values[:, [12, 13, 14]], dlc_body.values[:, [24, 25, 26]])
        stride_freq = body_processing.freq_analysis(rightpaws_fbdiff, BodyCam_FPS, M=128)
        motion_mouth_raw, motion_mouth, motionangle_mouth = body_processing.dlc_pointmotion(dlc_body.values[:, 3],
                                                                                        dlc_body.values[:, 4],
                                                                                        dlc_body.values[:, 5])
        angle_tail_3, angle_tail = body_processing.dlc_angle(dlc_body.values[:, [33, 34, 35]], dlc_body.values[:, [36, 37, 38]],
                                                        dlc_body.values[:, [39, 40, 41]])
        angle_paws_front_3, angle_paws_front = body_processing.dlc_angle(dlc_body.values[:, [9, 10, 11]],
                                                                    dlc_body.values[:, [12, 13, 14]],
                                                                    dlc_body.values[:, [15, 16, 17]])
        angle_paws_back_3, angle_paws_back = body_processing.dlc_angle(dlc_body.values[:, [21, 22, 23]],
                                                                dlc_body.values[:, [24, 25, 26]],
                                                                dlc_body.values[:, [27, 28, 29]])
        tailroot_level = -pd.Series(
            (dlc_body.values[:, 34] - np.nanmean(dlc_body.values[:, 34])) / np.nanstd(dlc_body.values[:, 34]))
        cylinder_mask = np.zeros([582, 782])
        cylinder_mask[int(np.nanpercentile(dlc_body.values[:, 22], 99) + 30):, :250] = 1
        cylinder_motion_raw, cylinder_motion = body_processing.cylinder_motion(bodyfile, cylinder_mask)
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



## New functions
def download_dlc():   
    raise NotImplementedError("") 
    # download / set DLC network directories
    # dlc_faceyaml = '/media/oliver/Oliver_SSD1/MouseFace-Barnstedt-2019-08-21/config.yaml'
    # dlc_bodyyaml = '/media/oliver/Oliver_SSD1/MouseBody-Barnstedt-2019-09-09/config.yaml'

    # if not dlc_faceyaml:
    #     dlc_face_url = 'https://drive.google.com/drive/folders/1_XPPyzaxMjQ901vJCwtv1g_h5DYWHM8j?usp=sharing'
    #     if dlc_dir:
    #         gdown.download_folder(dlc_face_url, dlc_dir, quiet=True, use_cookies=False)
    #     else:
    #         os.makedirs(os.path.join(dir, 'DLC_MouseFace'))
    #         gdown.download_folder(dlc_face_url, os.path.join(dir, 'DLC_MouseFace'), quiet=True, use_cookies=False)

    # if not dlc_bodyyaml:
    #     dlc_body_url = 'https://drive.google.com/drive/folders/1_XPPyzaxMjQ901vJCwtv1g_h5DYWHM8j?usp=sharing'
    #     if dlc_dir:
    #         gdown.download_folder(dlc_body_url, dlc_dir, quiet=True, use_cookies=False)
    #     else:
    #         os.makedirs(os.path.join(dir, 'DLC_MouseBody'))
    #         gdown.download_folder(dlc_body_url, os.path.join(dir, 'DLC_MouseBody'), quiet=True, use_cookies=False)


def apply_dlc_body(filetype, vid_output, dlc_bodyyaml, dir_out, bodyfile):
    import deeplabcut
    deeplabcut.analyze_videos(config=dlc_bodyyaml, videos=[bodyfile], shuffle=1, videotype=filetype, destfolder=dir_out)
    print("DLC body labels saved in ", dir_out)
    if vid_output and not glob.glob(dir_out + '/' + os.path.basename(bodyfile)[:-4] + '*labeled.mp4'):
        print("Generating labeled body video...")
        if vid_output > 1:
            plottingframes = np.arange(vid_output)
        else:
            plottingframes = None
        deeplabcut.create_labeled_video(config=dlc_bodyyaml, videos=[bodyfile], draw_skeleton=False,
                                                        destfolder=dir_out, Frames2plot=plottingframes)
                                                

def apply_dgp_body(dlc_bodyyaml, dir_out, bodyfile):
    snapshot_path, _ = get_snapshot_path('snapshot-step2-final--0', os.path.dirname(dlc_bodyyaml), shuffle=1)
    estimate_pose(proj_cfg_file=dlc_bodyyaml,
                            dgp_model_file=str(snapshot_path),
                            video_file=bodyfile,
                            output_dir=dir_out,
                            shuffle=1,
                            save_pose=True,
                            save_str='',
                            new_size=None)
    
def apply_bodymodel(dgp, filetype, vid_output, dlc_bodyyaml, dir_out, bodyfile):
    if dgp:
        apply_dgp_body(dlc_bodyyaml, dir_out, bodyfile)
        print("DGP body labels saved in ", dir_out)
    else:
        apply_dlc_body(filetype, vid_output, dlc_bodyyaml, dir_out, bodyfile)

def apply_facemodel(dgp, filetype, vid_output, dlc_faceyaml, dir_out, facefile):
    if dgp:
        apply_dgp_face(dlc_faceyaml, dir_out, facefile)
    else:
        apply_dlc_face(filetype, vid_output, dlc_faceyaml, dir_out, facefile)

def apply_dlc_face(filetype, vid_output, dlc_faceyaml, dir_out, facefile):
    import deeplabcut
    deeplabcut.analyze_videos(config=dlc_faceyaml, videos=[facefile], shuffle=1, 
                                        videotype=filetype, destfolder=dir_out)
    print("DLC face labels saved in ", dir_out)
    if vid_output and not glob.glob(dir_out + '/' + os.path.basename(facefile)[:-4] + '*labeled.mp4'):
        print("Generating labeled face video...")
        if vid_output > 1:
            plottingframes = np.arange(vid_output)
        else:
            plottingframes = None
        deeplabcut.create_labeled_video(config=dlc_faceyaml, videos=[facefile], draw_skeleton=False,
                                                     destfolder=dir_out, Frames2plot=plottingframes)

def apply_dgp_face(dlc_faceyaml, dir_out, facefile):
    from deepgraphpose.models.eval import estimate_pose
    from deepgraphpose.models.fitdgp_util import get_snapshot_path
    snapshot_path, _ = get_snapshot_path('snapshot-step2-final--0', os.path.dirname(dlc_faceyaml), shuffle=1)
    estimate_pose(proj_cfg_file=dlc_faceyaml,
                            dgp_model_file=str(snapshot_path),
                            video_file=facefile,
                            output_dir=dir_out,
                            shuffle=1,
                            save_pose=True,
                            save_str='',
                            new_size=None)
    print("DGP face labels saved in ", dir_out)
                                             
