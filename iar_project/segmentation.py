import cv2
import numpy as np

from iar_project.utils import DATA_PATH


def draw_contours(img, contours):
    """Method for drawing the contour found on an image
    Args:   img(np.array): the raw image
            conoutours(np.array): the list of countours
    return  The binary image
    """
    # Make a copy of the original image to draw on
    img_with_contours = np.zeros_like(img)
    # Draw the contours on the image
    cv2.drawContours(
        image=img_with_contours,
        contours=contours,
        contourIdx=-1,
        color=(255, 255, 255),
        thickness=1,
    )
    cv2.fillPoly(img_with_contours, pts=contours, color=(255, 255, 255))
    return img_with_contours


def clean_mask(mask, min_area=1000, max_area=1600):
    """Method for cleaning the mask
    Args:   mask(np.array())= the thresholded image
            min_area=minimal size of the area
            max_area=maximal size of the area
    return  the cleaned countours(np.array)
    """
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


def extract_segemented_object(src_img, mask, src_image_number):
    """Method for extracting the objects and saving them under segementation_results
    Args:   src_img: source image in RGB
            mask: Binary mask of the image
            src_image_number: index of the images
    return  None
    """
    path_out = DATA_PATH + "/segmentation_results"
    # Split the source image
    src_b, src_g, src_r = cv2.split(src_img)
    # Find the contour in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ind = 0
    for c in contours:
        # Create the rectrangle
        rectangleArea = cv2.minAreaRect(c)
        angle = int(rectangleArea[2])
        center = (int(rectangleArea[0][0]), int(rectangleArea[0][1]))
        # Produce the rotated channel
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        r_rotated = cv2.warpAffine(
            src_r, M, dsize=src_r.shape[1::-1], flags=cv2.INTER_LINEAR
        )
        g_rotated = cv2.warpAffine(
            src_g, M, dsize=src_g.shape[1::-1], flags=cv2.INTER_LINEAR
        )
        b_rotated = cv2.warpAffine(
            src_b, M, dsize=src_b.shape[1::-1], flags=cv2.INTER_LINEAR
        )
        # Define where to crop
        min_y = np.min(c[:, :, 0])
        max_y = np.max(c[:, :, 0])
        min_x = np.min(c[:, :, 1])
        max_x = np.max(c[:, :, 1])
        # Crop each channel
        segement_r = r_rotated[min_x : max_x + 1, min_y : max_y + 1]
        segement_g = g_rotated[min_x : max_x + 1, min_y : max_y + 1]
        segement_b = b_rotated[min_x : max_x + 1, min_y : max_y + 1]
        # Fuse the channel
        segemented = cv2.merge([segement_r, segement_g, segement_b])
        cv2.imwrite(f"{path_out}/{src_image_number}_{ind}.png", segemented)
        ind += 1
    return


def filter_1(src_imgages):
    """Takes the list of images and return the filtered version in an array"""
    train_data_hue = [
        cv2.cvtColor(data, cv2.COLOR_BGR2HSV)[:, :, 0] for data in src_imgages
    ]
    threshold_h = [
        cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1] for img in train_data_hue
    ]
    mask_clean = [
        draw_contours(img, clean_mask(img, min_area=15000, max_area=16000))
        for img in threshold_h
    ]
    return mask_clean


def filter_2(src_images):
    """Takes the list of images and return the filtered version in an array"""
    train_data_v = [
        cv2.cvtColor(data, cv2.COLOR_BGR2HSV)[:, :, 2] for data in src_images
    ]
    threshold_v_100 = [
        cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] for img in train_data_v
    ]
    kernel = np.ones((5, 5), np.uint8)
    threshold_v_100 = [
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
        for img in threshold_v_100
    ]
    mask_clean = [
        draw_contours(img, clean_mask(img, min_area=15000, max_area=16000))
        for img in threshold_v_100
    ]
    return mask_clean


def filter_3(src_images):
    train_data_hue = [
        cv2.cvtColor(data, cv2.COLOR_BGR2HSV)[:, :, 0] for data in src_images
    ]
    threshold_h_28 = [
        cv2.threshold(img, 28, 255, cv2.THRESH_BINARY)[1] for img in train_data_hue
    ]
    kernel = np.ones((5, 5), np.uint8)
    threshold_h_28 = [
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
        for img in threshold_h_28
    ]
    mask_clean = [
        draw_contours(img, clean_mask(img, min_area=15000, max_area=16000))
        for img in threshold_h_28
    ]
    return mask_clean
