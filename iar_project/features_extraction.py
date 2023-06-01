import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk


def get_entropy_r(gray_img, r):
    entropy_img1 = entropy(gray_img, disk(r))
    return np.median(entropy_img1)


def get_median_V(hsv_img):
    return np.median(hsv_img[:, :, 2])


def get_median_Green(rgb_img):
    return np.median(rgb_img[:, :, 1])


def get_mean_Saturation(hsv_img):
    return np.mean(hsv_img[:, :, 1])


def get_std_Saturation(hsv_img):
    return np.std(hsv_img[:, :, 1])


def get_mean_power(gray_img):
    dft_img = np.fft.rfft2(gray_img)
    return np.mean(np.abs(dft_img))


def get_features_color(seg_img):
    """Method for extracting all the features"""
    img_features = []
    for img in seg_img:
        features = np.zeros((len(img), 7))
        for i, til in enumerate(img):
            # Get the image in the different color space
            tile = til[10:118, 10:118, :]
            gray_im = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
            hsv_im = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
            # Extract the features
            features[i, 0] = get_entropy_r(gray_im, 5)  # 5 7
            features[i, 3] = get_entropy_r(gray_im, 2)
            features[i, 2] = get_median_V(hsv_im)  # 0 1 4 6
            features[i, 1] = get_median_Green(tile)  # 0 1 2
            features[i, 4] = get_std_Saturation(hsv_im)
            features[i, 5] = get_mean_Saturation(hsv_im)
            features[i, 6] = get_mean_power(gray_im)
        img_features.append(features)

    return img_features
