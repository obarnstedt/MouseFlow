import cv2
from tqdm import tqdm

def flip_vid(vid_orig, horizontal=False, vertical=False):
    cap = cv2.VideoCapture(vid_orig)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    vidlength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_flipped = cv2.VideoWriter(vid_orig[:-4]+'_flipped.mp4', cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (frame_width, frame_height))
    
    print('Flipping video...')
    with tqdm(total=vidlength) as pbar:
        while cap.isOpened():
            ret, current_frame = cap.read()
            if current_frame is None:
                break
            if horizontal:
                current_frame = cv2.flip(current_frame, 1)
            if vertical:
                current_frame = cv2.flip(current_frame, 0)
            vid_flipped.write(current_frame)
            pbar.update(1)

    return vid_orig[:-4]+'_flipped.mp4'

def crop_vid(vid_orig, crop):
    crop = [int(c) for c in crop]  # ensuring only integers in list
    cap = cv2.VideoCapture(vid_orig)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    vidlength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_cropped = cv2.VideoWriter(vid_orig[:-4]+'_cropped.mp4', cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (frame_width, frame_height))
    
    print('Cropping video...')
    with tqdm(total=vidlength) as pbar:
        while cap.isOpened():
            _, current_frame = cap.read()
            if current_frame is None:
                break
            cropped_frame = current_frame[crop[0]:crop[1], crop[2]:crop[3]]
            vid_cropped.write(cropped_frame)
            pbar.update(1)

    return vid_orig[:-4]+'_cropped.mp4'