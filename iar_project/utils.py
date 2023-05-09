import os

import cv2
import numpy as np

GENERAL_PATH = "/Users/edouardkoehn/Documents/GitHub/iar_project"
DATA_PATH = GENERAL_PATH + "/data_project"


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
            img = cv2.imread(img_path)
            img_list.append(img)
        else:
            print(f"Image {img_name} not found.")
    return img_list


def draw_contours(img, contours):
    """Method for drawing the contour found on an image
    Args:   img(np.array): the raw image
            conoutours(np.array): the list of countours"""
    # Make a copy of the original image to draw on
    img_with_contours = np.zeros_like(img)
    # Draw the contours on the image
    cv2.drawContours(
        image=img_with_contours,
        contours=contours,
        contourIdx=-1,
        color=(255, 255, 255),
        thickness=2,
    )
    cv2.fillPoly(img_with_contours, pts=contours, color=(255, 255, 255))
    return img_with_contours


def clean_mask(mask, min_area=1000, max_area=1600):
    """Method for cleaning the mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area
    filtered_contours = [
        cnt
        for cnt in contours
        if (cv2.contourArea(cnt) > min_area and cv2.contourArea(cnt) < max_area)
    ]

    rects = [cv2.minAreaRect(cnt) for cnt in filtered_contours]
    boxes = [cv2.boxPoints(rect) for rect in rects]
    boxes = [np.intp(box) for box in boxes]
    return boxes
