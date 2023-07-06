import glob
import os
import numpy as np


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


def apply_dgp(dlc_yaml, dir_out, vid_file, vid_output):
    from deepgraphpose.models.eval import estimate_pose, plot_dgp
    from deepgraphpose.models.fitdgp_util import get_snapshot_path
    snapshot_path, _ = get_snapshot_path('snapshot-step2-final--0', os.path.dirname(dlc_yaml), shuffle=1)
    if vid_output > 1:
        plot_dgp(vid_file,
                dir_out,
                proj_cfg_file=dlc_yaml,
                dgp_model_file=str(snapshot_path),
                shuffle=1,
                dotsize=8)
        print("DGP face labels and labeled video saved in ", dir_out)
    else:
        estimate_pose(proj_cfg_file=dlc_yaml,
                            dgp_model_file=str(snapshot_path),
                            video_file=vid_file,
                            output_dir=dir_out,
                            shuffle=1,
                            save_pose=True,
                            save_str='',
                            new_size=None)
    
def apply_dlc(filetype, vid_output, dlc_yaml, dir_out, vid_file):
    import deeplabcut
    deeplabcut.analyze_videos(config=dlc_yaml, videos=[vid_file], shuffle=1, videotype=filetype, destfolder=dir_out)
    if vid_output and not glob.glob(dir_out + '/' + os.path.basename(vid_file)[:-4] + '*labeled.mp4'):
        deeplabcut.create_labeled_video(
            config=dlc_yaml, 
            videos=[vid_file], 
            draw_skeleton=False,
            destfolder=dir_out, 
            Frames2plot=np.arange(vid_output) if vid_output > 1 else None
        )