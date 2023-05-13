import os

import cv2
import numpy as np

GENERAL_PATH = "/Users/edouardkoehn/Documents/GitHub/iar_project"
DATA_PATH = GENERAL_PATH + "/data_project"
SEGMENTATION_OUTPUT_PATH = DATA_PATH + "/segmentation_results"


def import_train():
    """Method for loading the train image and return them as a list"""

    train_path = DATA_PATH + "/train"
    img_list = []
    for i in range(15):
        img_name = (
            f"train_{str(i).zfill(2)}.png"  # zfill() pads the number with leading zeros
        )
        img_path = os.path.join(train_path, img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)
        else:
            print(f"Image {img_name} not found.")
    return img_list


def check_output_segmentation_folder():
    path_out = SEGMENTATION_OUTPUT_PATH
    if not (os.path.exists(path_out)):
        os.mkdir(path_out)
    return path_out
