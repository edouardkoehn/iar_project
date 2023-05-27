import glob
import os

import cv2
import numpy as np

GENERAL_PATH = (
    "/Users/begue/OneDrive/Documents/EPFL/Master/Cours/MA2/IAPR/projet/iar_project"
)
DATA_PATH = GENERAL_PATH + "/data_project"
DATA_PATH2 = GENERAL_PATH + "/data_project2"
SEGMENTATION_OUTPUT_PATH = DATA_PATH + "/segmentation_results"
SEGMENTATION_OUTPUT_PATH2 = DATA_PATH2 + "/segmentation_results"
SOL_OUTPUT_PATH2 = DATA_PATH2 + "/train2_solutions"
CLUSTERING_OUTPUT_PATH2 = DATA_PATH2 + "/clustering_results"


def import_train():
    """Method for loading the train image and return them as a list  (dataset2)"""

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


def import_train2():
    """Method for loading the second train image and return them as a list (dataset2)"""
    train_path = DATA_PATH2 + "/train2"
    img_list = []
    for i in range(12):
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
            print(img_path)
    return img_list


def import_seg_results(k, nb_im):
    """Method for loading the segmentation results"""
    img_list = []
    for i in range(k, nb_im):
        seg_path = SEGMENTATION_OUTPUT_PATH2 + "/" + str(i) + "_" + "*.png"
        tmp = []
        for filename in glob.glob(seg_path):
            if os.path.exists(filename):
                img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tmp.append(img)
            else:
                print(f"Image {filename} not found.")
                print(filename)
        img_list.append(tmp)
    return img_list


def import_solution():
    """Method for loading the segmentation results"""
    img_list = []
    seg_path = SOL_OUTPUT_PATH2 + "/solution" + "*.png"
    for filename in glob.glob(seg_path):
        if os.path.exists(filename):
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)
        else:
            print(f"Image {filename} not found.")
            print(filename)
    return img_list


def check_output_segmentation_folder(dataset_used=2):
    """Method for creating the segmentation output folder"""
    if dataset_used == 2:
        path_out = SEGMENTATION_OUTPUT_PATH2
    else:
        path_out = SEGMENTATION_OUTPUT_PATH2
    if not (os.path.exists(path_out)):
        os.mkdir(path_out)
    return path_out


def check_output_clustering_folder(dataset_used=2):
    """Method for creating the segmentation output folder"""
    if dataset_used == 2:
        path_out = CLUSTERING_OUTPUT_PATH2
    else:
        path_out = CLUSTERING_OUTPUT_PATH2
    if not (os.path.exists(path_out)):
        os.mkdir(path_out)
    return path_out
