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


def extract_segemented_object(src_img, mask, src_image_number, dataset_use=2):
    """Method for extracting the objects and saving them under segementation_results
    Args:   src_img: source image in RGB
            mask: Binary mask of the image
            src_image_number: index of the images
    return  None
    """
    if dataset_use == 2:
        path_out = utils.check_output_segmentation_folder(dataset_used=2)
    else:
        path_out = utils.check_output_segmentation_folder(dataset_used=1)
    # Split the source image
    src_b, src_g, src_r = cv2.split(src_img)
    # Find the contour in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ind = 0
    for c in contours:
        if len(c) > 1:
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
        else:
            print()
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
    """Takes the list of images and return the filtered version in an array"""
    train_data_b = [cv2.split(data)[2] for data in src_images]
    canny = [cv2.Canny(data, 10, 50) for data in train_data_b]
    kernel = np.ones((5, 5), np.uint8)
    close = [
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3) for img in canny
    ]
    mask_clean = [
        draw_contours(img, clean_mask(img, min_area=14000, max_area=17000))
        for img in close
    ]
    return mask_clean


def filter_4(src_images):
    """Takes the list of images and return the filtered version in an array"""
    train_data_hue = [
        cv2.cvtColor(data, cv2.COLOR_RGB2HSV)[:, :, 0] for data in src_images
    ]
    threshold_h = [
        cv2.threshold(img, 38, 255, cv2.THRESH_BINARY)[1] for img in train_data_hue
    ]
    kernel = np.ones((3, 3), np.uint8)
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
    """Takes the list of images and return the filtered version in an array"""
    train_data_value = [
        cv2.cvtColor(data, cv2.COLOR_RGB2HSV)[:, :, 2] for data in src_images
    ]
    threshold = [
        cv2.threshold(img, 114, 255, cv2.THRESH_BINARY)[1] for img in train_data_value
    ]
    mask_clean = [
        draw_contours(img, clean_mask(img, min_area=15000, max_area=16384))
        for img in threshold
    ]
    return mask_clean


def filter_6(src_images):
    """Takes the list of images and return the filtered version in an array"""
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
    """Takes the list of images and return the filtered version in an array"""
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


def filter_8(src_images):
    """Takes the list of images and return the filtered version in an array"""
    train_data_gray = [cv2.cvtColor(data, cv2.COLOR_RGB2GRAY) for data in src_images]
    canny = [cv2.Canny(data, 20, 70) for data in train_data_gray]
    kernel = np.ones((5, 5), np.uint8)
    close = [
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3) for img in canny
    ]
    mask_clean = [
        draw_contours(img, clean_mask(img, min_area=15000, max_area=20000))
        for img in close
    ]
    return mask_clean


def filter_9(src_images):
    """Takes the list of images and return the filtered version in an array"""
    train_data_r = [cv2.split(data)[0] for data in src_images]
    # fig, axs=plt.subplots(1,2, figsize=(8,3),num='Red channel' )
    # axs[1].hist(train_data_r[2].ravel(), bins=256, density=True)
    # axs[1].set_title('Histogram red channel')
    # axs[1].set_ylabel('%')
    # axs[1].vlines(113,0,0.07,color='red')
    # axs[1].vlines(230,0,0.07,color='red')
    # axs[0].imshow(train_data_r[2], cmap='gray')
    # axs[0].set_title('Red channel')
    # axs[0].axis('off')
    # plt.tight_layout()
    # plt.savefig('Red.png')

    threshold_low = [
        cv2.threshold(img, 113, 255, cv2.THRESH_BINARY)[1] for img in train_data_r
    ]
    threshold_high = [
        cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)[1] for img in train_data_r
    ]
    threshold = [low - high for low, high in zip(threshold_low, threshold_high)]
    threshold = [
        cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)[1] for img in threshold
    ]

    # fig, axs=plt.subplots(1,2, figsize=(8,3),num='Thresholded channel' )
    # axs[1].hist(threshold[2].ravel(), bins=256, density=True)
    # axs[1].set_title('Histogram thresholded')
    # axs[1].set_ylabel('%')
    # axs[0].imshow(threshold[2], cmap='gray')
    # axs[0].set_title('Thresholded')
    # axs[0].axis('off')
    # plt.tight_layout()
    # plt.savefig('Thresholded.png')

    canny = [cv2.Canny(data, 10, 20) for data in threshold]

    kernel = np.ones((5, 5), np.uint8)
    close = [
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3) for img in canny
    ]

    # fig, axs=plt.subplots(1,2, figsize=(8,3),num='Canny +Closing' )
    # axs[0].imshow(canny[2],cmap='gray')
    # axs[0].set_title('Canny')
    # axs[0].axis('off')
    # axs[1].imshow(threshold[2], cmap='gray')
    # axs[1].set_title('Closing')
    # axs[1].axis('off')
    # plt.tight_layout()
    # plt.savefig('Canny_Closing.png')

    mask_clean = [
        draw_contours(img, clean_mask(img, min_area=13000, max_area=20000))
        for img in close
    ]
    # fig, axs=plt.subplots(1,1, figsize=(8,3),num='Final mask' )
    # axs.imshow(mask_clean[2],cmap='gray')
    # axs.set_title('Mask cleaned')
    # axs.axis('off')
    # plt.tight_layout()
    # plt.savefig('Mask_clean.png')
    return mask_clean


def compute_segementation_filters(src_images):
    """Method for computing all the segmentation masks"""
    mask = []
    M1 = filter_1(src_images)
    M2 = filter_2(src_images)
    M3 = filter_3(src_images)
    M4 = filter_4(src_images)
    M5 = filter_5(src_images)
    M6 = filter_6(src_images)
    M7 = filter_7(src_images)
    M8 = filter_8(src_images)
    M9 = filter_9(src_images)
    for f1, f2, f3, f4, f5, f6, f7, f8, f9 in zip(M1, M2, M3, M4, M5, M6, M7, M8, M9):
        mask.append(f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9)
    return mask


def plot_mask(images, masks, title="Test"):
    """Method for plotting the masks on the raw imagess"""
    plt.figure(figsize=(20, 7))
    plt.suptitle(title)

    for img, mask, ind in zip(images, masks, np.arange(15)):
        b, g, r = cv2.split(img)
        r = r * mask
        g = g * mask
        b = b * mask
        color_img = cv2.merge([r, g, b])
        plt.subplot(2, 8, ind + 1)
        plt.title(f"img {ind}")
        plt.imshow(color_img, cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_mask2(images, masks, title="Test"):
    """Method for plotting the masks on the raw imagess"""
    plt.figure(figsize=(20, 8))
    plt.suptitle(title)
    for img, mask, ind in zip(images, masks, np.arange(12)):
        b, g, r = cv2.split(img)
        r = r * mask
        g = g * mask
        b = b * mask
        color_img = cv2.merge([r, g, b])
        plt.subplot(2, 6, ind + 1)
        plt.title(f"img {ind}")
        plt.imshow(color_img, cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_comparison(images, masks, title="Comparison"):
    """Method for plotting the mask against the raw image"""
    fig, axs = plt.subplots(nrows=2, ncols=12, figsize=(20, 3))
    for img, mask, ind in zip(images, masks, np.arange(15)):
        b, g, r = cv2.split(img)
        r = r * mask
        g = g * mask
        b = b * mask
        color_img = cv2.merge([g, r, b])

        axs[0, ind].set_title(f"img {ind}")
        axs[0, ind].imshow(color_img, cmap="gray")
        axs[0, ind].axis("off")

        axs[1, ind].imshow(img)
        axs[1, ind].axis("off")
    plt.tight_layout()
    plt.show()


# import iar_project.utils as utils
# import matplotlib.pyplot as plt


# train_data=utils.import_train2()
# mask=compute_segementation_filters(train_data)
# fig,axs=plt.subplots(2,3)
# for i in range(3):
#     axs[0,i].imshow(train_data[i], cmap='gray')
#     axs[0,i].axis('off')

# for i in range(3):
#     axs[1,i].imshow(mask[i],cmap='gray')
#     axs[1,i].axis('off')
# plt.savefig('Seg_res')

# M1=filter_8(train_data)
# M4=filter_4(train_data)
# M9=filter_9(train_data)

# plt.figure('M1')
# plt.imshow(M1[5], cmap='gray')
# plt.axis('off')
# plt.savefig('M1')

# plt.figure('M4')
# plt.imshow(M4[5], cmap='gray')
# plt.axis('off')
# plt.savefig('M4')

# plt.figure('M5')
# plt.imshow(M9[5], cmap='gray')
# plt.axis('off')
# plt.savefig('M5')

# plt.figure('Res5')
# plt.imshow(mask[1], cmap='gray')
# plt.axis('off')
# plt.savefig('res5')
