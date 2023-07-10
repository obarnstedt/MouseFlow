import glob
import os
import numpy as np
import gdown

def download_models(models_dir, facemodel_name, bodymodel_name):   
    if not os.path.exists(os.path.join(models_dir, facemodel_name)):
        dlc_face_url = 'https://drive.google.com/drive/folders/1_XPPyzaxMjQ901vJCwtv1g_h5DYWHM8j?usp=sharing'
        gdown.download_folder(dlc_face_url, models_dir, quiet=True, use_cookies=False)

    if not os.path.exists(os.path.join(models_dir, bodymodel_name)):
        dlc_body_url = 'https://drive.google.com/drive/folders/1_XPPyzaxMjQ901vJCwtv1g_h5DYWHM8j?usp=sharing'
        gdown.download_folder(dlc_body_url, models_dir, quiet=True, use_cookies=False)
    
    dlc_faceyaml = os.path.join(models_dir, facemodel_name, 'config.yaml')
    dlc_bodyyaml = os.path.join(models_dir, bodymodel_name, 'config.yaml')

    return dlc_faceyaml, dlc_bodyyaml


def apply_dgp(dlc_yaml, dir_out, vid_file, vid_output):
    from deepgraphpose.models.eval import estimate_pose, plot_dgp
    from deepgraphpose.models.fitdgp_util import get_snapshot_path
    snapshot_path, _ = get_snapshot_path('snapshot-0-step2-final--0', os.path.dirname(dlc_yaml), shuffle=1)
    if vid_output > 1:
        plot_dgp(vid_file,
                dir_out,
                proj_cfg_file=dlc_yaml,
                dgp_model_file=str(snapshot_path),
                shuffle=1,
                dotsize=8)
        print("DGP labels and labeled video saved in ", dir_out)
    else:
        estimate_pose(proj_cfg_file=dlc_yaml,
                            dgp_model_file=str(snapshot_path),
                            video_file=vid_file,
                            output_dir=dir_out,
                            shuffle=1,
                            save_pose=True,
                            save_str='',
                            new_size=None)
        print("DGP labels saved in ", dir_out)
    
def apply_dlc(filetype, vid_output, dlc_yaml, dir_out, vid_file, overwrite):

    import deeplabcut

    if overwrite:  # if overwrite desired, identify and delete previously processed marker and video file
        from deeplabcut.utils.auxiliaryfunctions import find_analyzed_data
        analysisfile, _, _ = find_analyzed_data(dir_out, os.path.splitext(vid_file)[0], scorer='DLC_resnet50_Mouse')
        if analysisfile:
            os.remove(analysisfile)
        labeled_vid = glob.glob(dir_out + '/' + os.path.basename(vid_file)[:-4] + '*labeled.mp4')
        if labeled_vid:
            [os.remove(f) for f in labeled_vid]

    deeplabcut.analyze_videos(config=dlc_yaml, videos=[vid_file], shuffle=1, videotype=filetype, destfolder=dir_out)
    print("DLC labels saved in ", dir_out)

    if vid_output and not glob.glob(dir_out + '/' + os.path.basename(vid_file)[:-4] + '*labeled.mp4'):
        print("Generating DLC labeled video...")
        deeplabcut.create_labeled_video(
            config=dlc_yaml, 
            videos=[vid_file], 
            draw_skeleton=False,
            destfolder=dir_out, 
            Frames2plot=np.arange(vid_output) if vid_output > 1 else None
        )
    print("DLC labeled video saved in ", dir_out)