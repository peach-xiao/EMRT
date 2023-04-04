import cv2
import numpy as np
from PIL import Image

ISPRS_color = {'Impervious surfaces': [255, 255, 255],
               'Building': [0, 0, 255],
               'Low vegetation': [0, 255, 255],
               'Tree': [0, 255, 0],
               'Car': [255, 255, 0],
               'Clutter/background': [255, 0, 0]}

dataset_color = {'ISPRS': ISPRS_color}


def color2label(img, dataset='ISPRS'):
    colormap = []
    for key in dataset_color[dataset]:
        colormap.append(dataset_color[dataset][key])
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

    data = np.array(img, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')
