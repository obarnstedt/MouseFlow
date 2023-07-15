import numpy as np

def confidence_na(dgp, conf_thresh, markers_face):
    if not conf_thresh:
        conf_thresh = 0.5 if dgp else 0.99
    facemarker_names = markers_face.columns.get_level_values(0).to_list()
    markers_face_conf = markers_face.copy().drop('likelihood', axis=1, level=1)
    for facemarker_name in facemarker_names:
        markers_face_conf.loc[(markers_face.loc[:, (facemarker_name, 'likelihood')]<conf_thresh), (facemarker_name, ['x', 'y'])] = np.nan
    return markers_face_conf