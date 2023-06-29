from tqdm import tqdm
import cv2  # Tested on OpenCV-contrib 4.2 compiled with CUDA support on Ubuntu 20.04 w/ GeForce RTX 2080 Ti
import time
import flow_vis  # visualisation from https://github.com/tomrunia/OpticalFlow_Visualization
import os
import numpy as np
import pandas as pd

# some info on parameters: https://docs.opencv.org/4.2.0/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
timings = {}  # place holder for time keeping

videopath = '/media/oliver/Oliver_SSD1/Duerschmid/2023-05-11_12-48-11.mp4'#'/media/oliver/Oliver_SSD/facecam-training/sample.mp4'
resultsdir = '/media/oliver/Oliver_SSD1/Duerschmid'
videoslice = [0, 1000]  # use only slice of video
save_vid = True
blend_gray_optflow = 0.5  # factor for blending in gray frame with optical flow color output video
me_cmap = cv2.COLORMAP_TURBO  # colormap for motion energy video
use_masks = False#'gpu'  # to extract avg mag/ang from masked video areas with either 'cpu' or 'gpu'
if use_masks:
    # import h5py
    # hfg = h5py.File('/mnt/ag-remy-2/Imaging/OB/Results/269/behaviour.h5')
    # masks = (np.array(hfg['facemasks'])[:2]).astype('float32')
    masks = np.array(np.zeros((480, 640)))
    masks[240:, :] = 1
    masks = masks[np.newaxis, ...].astype('float32')

    mask_mag = np.empty((videoslice[1]-videoslice[0], masks.shape[0]))
    mask_ang = np.empty((videoslice[1]-videoslice[0], masks.shape[0]))
    mask_x = np.empty((videoslice[1]-videoslice[0], masks.shape[0]))
    mask_y = np.empty((videoslice[1]-videoslice[0], masks.shape[0]))
    gpu_masks = []
    for m in range(masks.shape[0]):
        gpu_mask = cv2.cuda_GpuMat()
        gpu_mask.upload(masks[m, :, :])
        gpu_masks.append(gpu_mask)
    masks[masks==0] = np.nan

# SELECT OPTICAL FLOW ALGORITHM
flownames = ['cuda_NvidiaOpticalFlow_1_0',
            #  'cuda_NvidiaOpticalFlow_2_0',  # not yet working
             'cuda_FarnebackOpticalFlow',
             'cuda_BroxOpticalFlow',
             'motion_energy',
             'cuda_DensePyrLKOpticalFlow',
             'cuda_OpticalFlowDual_TVL1']

for flowname in flownames:
    facemp4 = cv2.VideoCapture(videopath)
    if videoslice:
        facemp4.set(1, videoslice[0])
        framelength = len(range(videoslice[0], videoslice[1]))
    else:
        framelength = int(facemp4.get(7))
    ret, current_frame = facemp4.read()
    previous_frame = current_frame

    if flowname == 'cuda_FarnebackOpticalFlow':
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(numLevels=5, pyrScale=.5, fastPyramids=True, winSize=25,
                                                        numIters=3, polyN=5, polySigma=1.2, flags=0)
        # STANDARD: numLevels = 5, pyrScale = 0.5, fastPyramids=False, winSize=15, numIters=3, polyN=5, polySigma=1.2, flags=0

    elif flowname == 'cuda_NvidiaOpticalFlow_1_0':
        gpu_flow = cv2.cuda_NvidiaOpticalFlow_1_0.create((current_frame.shape[1], current_frame.shape[0]),
                                                         5, False, False, False, 0)
    elif flowname == 'cuda_NvidiaOpticalFlow_2_0':
        gpu_flow = cv2.cuda_NvidiaOpticalFlow_2_0.create((current_frame.shape[1], current_frame.shape[0]),
                                                         5, 4, 4, False, False, False, 0)
    elif flowname == 'cuda_BroxOpticalFlow':
        gpu_flow = cv2.cuda_BroxOpticalFlow.create(alpha=0.197, gamma=1.0, scale_factor=0.25, inner_iterations=5,
                                                   outer_iterations=150, solver_iterations=10)
    elif flowname == 'cuda_DensePyrLKOpticalFlow':
        gpu_flow = cv2.cuda_DensePyrLKOpticalFlow.create()
    elif flowname == 'cuda_OpticalFlowDual_TVL1':
        gpu_flow = cv2.cuda_OpticalFlowDual_TVL1.create(.25, .15, .3, 5, 5, .01, 300, .8, 0.0, False)
        # tau = 0.25, lambda = 0.15, theta = 0.3, nscales = 5, warps = 5, epsilon = 0.01, iterations = 300,
        # scaleStep = 0.8, gamma = 0.0, useInitialFlow = false

    if save_vid and not os.path.exists(os.path.join(resultsdir, flowname)):
        os.makedirs(os.path.join(resultsdir, flowname))

    start_time = time.time()
    i = videoslice[0]
    with tqdm(total=framelength) as pbar:
        while facemp4.isOpened():
            # PREPARE FRAMES
            if flowname == 'cuda_BroxOpticalFlow':  # Brox expects 32 bit images
                current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY).astype('float32')
                previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY).astype('float32')
            else:
                current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

            #  push frames to GPU:
            if flowname != 'motion_energy':  # motion energy is sometimes faster without GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(current_frame_gray)
                gpu_previous = cv2.cuda_GpuMat()
                gpu_previous.upload(previous_frame_gray)

            # CALCULATE OPTICAL FLOW
            if flowname == 'cuda_NvidiaOpticalFlow_1_0':
                flow = gpu_flow.calc(gpu_frame, gpu_previous, None)[0]
                flow = gpu_flow.upSampler(flow, (current_frame.shape[1], current_frame.shape[0]), gpu_flow.getGridSize(),
                                          None)
            elif flowname == 'cuda_NvidiaOpticalFlow_2_0':
                flow = gpu_flow.calc(gpu_frame, gpu_previous, None)[0]
            elif flowname == 'motion_energy':
                flow = cv2.absdiff(current_frame_gray, previous_frame_gray)  # CUDA: cv2.cuda.absdiff()
            else:
                flow = gpu_flow.calc(gpu_frame, gpu_previous, None)

            # SAVE VIDEO
            if save_vid:
                if flowname == 'motion_energy':
                    flow_color = cv2.applyColorMap(flow.astype('uint8'), me_cmap)
                elif flowname == 'cuda_NvidiaOpticalFlow_2_0':
                    flow_color = flow_vis.flow_to_color(np.array([flow[0].download(), flow[1].download()]), convert_to_bgr=True)
                else:
                    flow_color = flow_vis.flow_to_color(flow.download(), convert_to_bgr=True)
                if blend_gray_optflow:
                    dst = cv2.addWeighted(current_frame, blend_gray_optflow, flow_color, 1-blend_gray_optflow, 0)
                    cv2.imwrite(os.path.join(resultsdir, flowname, f'{i:03d}.png'), dst)
                else:
                    cv2.imwrite(os.path.join(resultsdir, flowname, f'{i:03d}.png'), flow_color)

            # calculate average activity in masks with either GPU / CPU
            if use_masks=='gpu':
                gpu_flow_x = cv2.cuda_GpuMat(flow.size(), cv2.CV_32FC1)
                gpu_flow_y = cv2.cuda_GpuMat(flow.size(), cv2.CV_32FC1)
                cv2.cuda.split(flow, [gpu_flow_x, gpu_flow_y])
                gpu_mag, gpu_ang = cv2.cuda.cartToPolar(gpu_flow_x, gpu_flow_y, angleInDegrees=True)
                for j, mask in enumerate(gpu_masks):
                    x_mask = cv2.cuda.multiply(gpu_flow_x, mask)
                    y_mask = cv2.cuda.multiply(gpu_flow_y, mask)
                    mag_mask = cv2.cuda.multiply(gpu_mag, mask)
                    ang_mask = cv2.cuda.multiply(gpu_ang, mask)
                    mask_mag[i, j] = np.nansum(mag_mask.download()) / np.nansum(masks[j])
                    mask_ang[i, j] = np.nansum(ang_mask.download()) / np.nansum(masks[j])
                    mask_x[i, j] = np.nansum(x_mask.download()) / np.nansum(masks[j])
                    mask_y[i, j] = np.nansum(y_mask.download()) / np.nansum(masks[j])
            elif use_masks:
                for j, mask in enumerate(masks):
                    if flowname == 'motion_energy':
                        mask_mag[i, j] = np.nanmean(flow * mask)
                    else:
                        mask_mag[i, j] = np.nanmean(flow.download()[..., 0] * mask)
                        mask_ang[i, j] = np.nanmean(flow.download()[..., 1] * mask)

            pbar.update(1)
            i += 1
            previous_frame = current_frame.copy()
            ret, current_frame = facemp4.read()
            if i > videoslice[-1]-1:
                break
    facemp4.release()
    print(round(time.time() - start_time, 4))

    if not save_vid and not use_masks:
        timings[flowname] = time.time() - start_time

    if use_masks:
        oflow = pd.DataFrame(data={'OFmag': mask_mag[:,0], 'OFang': mask_ang[:,0], 'OFx': mask_x[:,0], 'OFy': mask_y[:,0]})



#
#
# # PLOTTING OF TRACES
# import pandas as pd
# import matplotlib.pyplot as plt
# optflow = pd.DataFrame()
# optflow['whiskmag'] = mask_mag[:, 1]
# optflow['whiskang'] = mask_ang[:, 1]
# optflow['whiskme'] = mask_mag[:, 1]
#
# # plotting of frames
# visibleframes = 150
# from scipy.stats import zscore
# optflow = optflow.apply(zscore)
# optflow.index = optflow.index - int(visibleframes/2)
#
#
# plt.style.use('dark_background')
# xticks = np.arange(-60, 75, 15)
# xticklabels = [round(l, 1) for l in np.arange(-.8, 1, .2)]
# dpi=96
# fig, ax = plt.subplots(figsize=(782/dpi, 200/dpi), dpi=dpi, facecolor='black')
# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.spines["left"].set_visible(False)
# ax.get_yaxis().set_visible(False)
#
# for k in range(500):
#     optflow_shifted = optflow.copy().shift(-k+int(visibleframes/2))
#     ax.plot(optflow_shifted.whiskme+3, label='Motion energy')
#     ax.plot(optflow_shifted.whiskmag, label='Optical flow magnitude')
#     ax.plot(optflow_shifted.whiskang-3, label='Optical flow angle')
#     ax.legend(loc=2)
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticklabels)
#     ax.set_title('Whiskerpad activity')
#     ax.set_ylim(-7, 8)
#     ax.set_xlabel('Time [s]')
#     ax.axvline(0, ls='--')
#     ax.set_xlim(-visibleframes / 2, visibleframes / 2)
#     ax.set_ylabel('Time [s]')
#     plt.savefig(f'/media/oliver/Oliver_SSD/optical_flow/plotting_gpu/{k:02d}')
#     ax.cla()




### Example CUDA compilation on Ubuntu 20.04 LTS:
# CONDA_ENV_PATH=/home/oliver/anaconda3/envs/imhotep
# CONDA_PYTHON_PATH=/home/oliver/anaconda3/envs/imhotep/bin/python
# WHERE_OPENCV=/home/oliver/Git/opencv-4.2.0
# WHERE_OPENCV_CONTRIB=/home/oliver/Git/opencv_contrib-4.2.0
#
# cmake -D CMAKE_BUILD_TYPE=RELEASE \
# -D CMAKE_C_COMPILER=/usr/bin/gcc-7 \
# -D CMAKE_INSTALL_PREFIX=/usr/local \
# -D INSTALL_PYTHON_EXAMPLES=ON \
# -D INSTALL_C_EXAMPLES=OFF \
# -D WITH_TBB=ON \
# -D BUILD_opencv_cudacodec=ON \
# -D ENABLE_FAST_MATH=1 \
# -D CUDA_FAST_MATH=1 \
# -D WITH_CUDA=ON \
# -D WITH_CUBLAS=1 \
# -D WITH_V4L=ON \
# -D WITH_QT=OFF \
# -D WITH_OPENGL=ON \
# -D WITH_GSTREAMER=ON \
# -D OPENCV_GENERATE_PKGCONFIG=ON \
# -D OPENCV_PC_FILE_NAME=opencv.pc \
# -D OPENCV_ENABLE_NONFREE=ON \
# -D OPENCV_PYTHON_INSTALL_PATH=~/anaconda3/envs/imhotep/lib/python3.9/site-packages \
# -D PYTHON_PACKAGES_PATH=~/anaconda3/envs/imhotep/lib/python3.9/site-packages \
# -D OPENCV_EXTRA_MODULES_PATH=~/Git/opencv_contrib-4.2.0/modules \
# -D PYTHON_EXECUTABLE=/home/oliver/anaconda3/envs/imhotep/bin/python \
# -D PYTHON3_LIBRARY=/home/oliver/anaconda3/envs/imhotep/lib/libpython3.so \
# -D PYTHON_INCLUDE_DIR=/home/oliver/anaconda3/envs/imhotep/include/python3.9 \
# -D PYTHON_NUMPY_INCLUDE_DIRS=/home/oliver/anaconda3/envs/imhotep/lib/python3.9/site-packages/numpy/core/include \
# -D PYTHON_DEFAULT_EXECUTABLE=/home/oliver/anaconda3/envs/imhotep/bin/python \
# -D BUILD_NEW_PYTHON_SUPPORT=ON \
# -D BUILD_PYTHON_SUPPORT=ON \
# -D BUILD_opencv_python3=ON \
# -D HAVE_opencv_python3=ON \
# -D BUILD_EXAMPLES=ON \
# -D WITH_CUDNN=ON \
# -D OPENCV_DNN_CUDA=ON \
# -D BUILD_opencv_python3=yes \
# -D CUDA_ARCH_BIN=7.5 ..


# https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7

# OPENCV 4.6 with Ubuntu 22.04, CUDA 11.7
# cmake \
# -D CMAKE_BUILD_TYPE=RELEASE \
# -D CMAKE_INSTALL_PREFIX=/usr/local \
# -D WITH_CUDA=ON \
# -D WITH_CUDNN=ON \
# -D WITH_CUBLAS=ON \
# -D WITH_TBB=ON \
# -D OPENCV_DNN_CUDA=ON \
# -D OPENCV_ENABLE_NONFREE=ON \
# -D CUDA_ARCH_BIN=7.5 \
# -D OPENCV_EXTRA_MODULES_PATH=$HOME/Git/opencv_contrib/modules \
# -D BUILD_EXAMPLES=OFF \
# -D HAVE_opencv_python3=ON \
# -D OPENCV_PYTHON_INSTALL_PATH=~/anaconda3/envs/mouseflow/lib/python3.9/site-packages \
# -D PYTHON_PACKAGES_PATH=~/anaconda3/envs/mouseflow/lib/python3.9/site-packages \
# -D PYTHON_EXECUTABLE=/home/oliver/anaconda3/envs/mouseflow/bin/python \
# -D PYTHON3_LIBRARY=/home/oliver/anaconda3/envs/mouseflow/lib/libpython3.so \
# -D PYTHON_INCLUDE_DIR=/home/oliver/anaconda3/envs/mouseflow/include/python3.9 \
# -D PYTHON_NUMPY_INCLUDE_DIRS=/home/oliver/anaconda3/envs/mouseflow/lib/python3.9/site-packages/numpy/core/include \
# -D PYTHON_DEFAULT_EXECUTABLE=/home/oliver/anaconda3/envs/mouseflow/bin/python \
# ..