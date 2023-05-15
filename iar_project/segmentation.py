import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import iar_project.utils as utils


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
    rects = [rect for rect in rects if (abs(rect[1][0] - rect[1][1]) < 3)]
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
    path_out = utils.check_output_segmentation_folder()
    # Split the source image
    src_b, src_g, src_r = cv2.split(src_img)
    # Find the contour in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ind = 0
    for c in contours:
        if len(c) > 100:
            # Create the rectrangle
            rectangleArea = cv2.minAreaRect(c)
            angle = int(rectangleArea[2])
            center = (int(rectangleArea[0][0]), int(rectangleArea[0][1]))
            box = cv2.boxPoints(rectangleArea)
            box = np.intp(box)
            # Rotate the image
            M = cv2.getRotationMatrix2D(center, angle, 1)
            r_rotated = cv2.warpAffine(src_r, M, src_r.shape[:2], cv2.WARP_INVERSE_MAP)
            g_rotated = cv2.warpAffine(src_g, M, src_g.shape[:2], cv2.WARP_INVERSE_MAP)
            b_rotated = cv2.warpAffine(src_b, M, src_b.shape[:2], cv2.WARP_INVERSE_MAP)

            # Extract the rotated object
            miny = np.min(box[:, 1]) if np.min(box[:, 1]) >= 0 else 0
            maxy = np.max(box[:, 1]) if np.max(box[:, 1]) <= 2000 else 2000

            minx = np.min(box[:, 0]) if np.min(box[:, 0]) >= 0 else 0
            maxx = np.max(box[:, 0]) if np.max(box[:, 0]) <= 2000 else 2000

            cropped_r = r_rotated[miny:maxy, minx:maxx]
            cropped_g = g_rotated[miny:maxy, minx:maxx]
            cropped_b = b_rotated[miny:maxy, minx:maxx]
            # Refine the extraction
            if (
                cropped_b.shape[0]
                >= 128 & cropped_b.shape[1]
                >= 128 & cropped_r.shape[1]
                >= 128
            ):
                rows, cols = cropped_r.shape
                center = [int((cols - 1) / 2.0), int((rows - 1) / 2.0)]
                cropped_r = cropped_r[
                    center[0] - 64 : center[0] + 64, center[1] - 64 : center[1] + 64
                ]
                cropped_g = cropped_g[
                    center[0] - 64 : center[0] + 64, center[1] - 64 : center[1] + 64
                ]
                cropped_b = cropped_b[
                    center[0] - 64 : center[0] + 64, center[1] - 64 : center[1] + 64
                ]
            # Fuse the channel

            segemented = cv2.merge([cropped_r, cropped_g, cropped_b])

            if segemented.shape[0] == 128:
                export_file = f"{path_out}/{src_image_number}_{ind}.png"
                cv2.imwrite(export_file, segemented)
                ind += 1
    return


def filter_1(src_imgages):
    """Takes the list of images and return the filtered version in an array"""
    train_data_hue = [
        cv2.cvtColor(data, cv2.COLOR_BGR2HSV)[:, :, 0] for data in src_imgages
    ]
    threshold_h = [
        cv2.threshold(img, 0.1, 255, cv2.THRESH_BINARY_INV)[1] for img in train_data_hue
    ]

    mask_clean = [
        draw_contours(img, clean_mask(img, min_area=15000, max_area=16384))
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
        draw_contours(img, clean_mask(img, min_area=14000, max_area=16384))
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
        draw_contours(img, clean_mask(img, min_area=15000, max_area=16384))
        for img in threshold_h_28
    ]
    return mask_clean


def filter_4(src_images):
    train_data_hue = [
        cv2.cvtColor(data, cv2.COLOR_RGB2HSV)[:, :, 0] for data in src_images
    ]
    threshold_h = [
        cv2.threshold(img, 38, 255, cv2.THRESH_BINARY)[1] for img in train_data_hue
    ]
    kernel = np.ones((6, 6), np.uint8)
    thresholded_h = [
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
        for img in threshold_h
    ]
    mask_clean = [
        draw_contours(img, clean_mask(img, min_area=15000, max_area=16384))
        for img in thresholded_h
    ]
    return mask_clean


def filter_5(src_images):
    train_data_hue = [
        cv2.cvtColor(data, cv2.COLOR_RGB2HSV)[:, :, 0] for data in src_images
    ]
    threshold_h = [
        cv2.threshold(img, 0.1, 255, cv2.THRESH_BINARY)[1] for img in train_data_hue
    ]
    kernel = np.ones((2, 2), np.uint8)
    thresholded_h = [
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
        for img in threshold_h
    ]
    thresholded_h = [
        cv2.threshold(img, 0.0, 255, cv2.THRESH_BINARY_INV)[1] for img in thresholded_h
    ]

    mask_clean = [
        draw_contours(img, clean_mask(img, min_area=15000, max_area=16384))
        for img in thresholded_h
    ]
    return mask_clean


def filter_6(src_images):
    train_data_blue = [cv2.split(data)[2] for data in src_images]

    train_data_red = [cv2.split(data)[0] for data in src_images]

    threshold_blue = [
        cv2.threshold(img, 45, 255, cv2.THRESH_BINARY_INV)[1] for img in train_data_blue
    ]

    threshold_red = [
        cv2.threshold(img, 34, 255, cv2.THRESH_BINARY_INV)[1] for img in train_data_red
    ]
    threshold_comb = [b + r for r, b in zip(threshold_red, threshold_blue)]
    kernel = np.ones((4, 4), np.uint8)
    thresholded = [
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
        for img in threshold_comb
    ]
    kernel = np.ones((2, 2), np.uint8)
    thresholded = [
        cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
        for img in thresholded
    ]
    mask_clean = [
        draw_contours(img, clean_mask(img, min_area=15000, max_area=20000))
        for img in thresholded
    ]
    return mask_clean


def filter_7(src_images):
    train_data_blue = [cv2.split(data)[2] for data in src_images]
    threshold_blue = [
        cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1] for img in train_data_blue
    ]
    kernel = np.ones((3, 3), np.uint8)
    thresholded = [
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
        for img in threshold_blue
    ]
    mask_clean = [
        draw_contours(img, clean_mask(img, min_area=15000, max_area=20000))
        for img in thresholded
    ]
    return mask_clean


def plot_mask(images, masks, title="Test"):
    plt.figure(figsize=(20, 7))
    plt.suptitle(title)

    for img, mask, ind in zip(images, masks, np.arange(15)):
        b, g, r = cv2.split(img)
        r = r * mask
        g = g * mask
        b = b * mask
        color_img = cv2.merge([r, g, b])
        plt.title(f"img {ind-1}")
        plt.subplot(2, 8, ind + 1)
        plt.imshow(color_img, cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_mask2(images, masks, title="Test"):
    plt.figure(figsize=(20, 7))
    plt.suptitle(title)

    for img, mask, ind in zip(images, masks, np.arange(15)):
        b, g, r = cv2.split(img)
        r = r * mask
        g = g * mask
        b = b * mask
        color_img = cv2.merge([r, g, b])
        plt.title(f"img {ind-1}")
        plt.subplot(2, 8, ind + 1)
        plt.imshow(color_img, cmap="gray")
        plt.axis("off")
    plt.tight_layout()


# import iar_project.utils as utils
# train_data=utils.import_train2()
# plt.figure()
# mask_none=[np.ones_like(train_data[0][:,:,1]) for i in range(11)]
# plot_mask2(train_data,mask_none)

# M1=filter_1(train_data)
# M2=filter_2(train_data)
# M3=filter_3(train_data)
# M4=filter_4(train_data)
# M5=filter_5(train_data)
# M6=filter_6(train_data)
# M7=filter_7(train_data)

# mask=[]
# for a,b,c,d,e,f,g in zip(M1,M2,M3,M4,M5,M6,M7):
#      mask.append(a+b+c+d+e+f+g)
# plt.figure('seg')
# plot_mask(train_data,mask)
# plt.show()
# for i in range(0,15):
#     extract_segemented_object(train_data[i],mask[i],i)
