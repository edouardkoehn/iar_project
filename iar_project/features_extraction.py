import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as sk
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import iar_project.utils as utils


def return_mean_std(channel):
    mean_cha = np.mean(channel)
    median_cha = np.median(channel)
    std_cha = np.std(channel)

    return mean_cha, median_cha, std_cha


def extract_hsv(im):
    hsv = np.zeros((9))
    hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    for i in range(0, 3):
        hsv[i], hsv[i + 3], hsv[i + 6] = return_mean_std(hsv_im[:, :, i])

    return hsv


def extract_rgb(im):
    rgb = np.zeros((9))
    rgb_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    for i in range(0, 3):
        rgb[i], rgb[i + 3], rgb[i + 6] = return_mean_std(rgb_im[:, :, i])

    return rgb


def extract_gray(im, ft_type):
    if ft_type == 1:
        gray = np.zeros((6))
    else:
        gray = np.zeros((4))
    gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    gray[0], gray[1], gray[2] = return_mean_std(gray_im)
    dft_img = np.fft.fft2(gray_im)
    gray[3] = np.abs(dft_img[0, 0])
    if ft_type == 1:

        gray[4] = np.abs(dft_img[1, 1])
        gray[5] = np.abs(dft_img[2, 2])

    return gray


def get_features(seg_img, ft_type):
    img_features = []
    for img in seg_img:
        if ft_type == 0:
            features = np.zeros((len(img), 19))
        elif ft_type == 1:
            features = np.zeros((len(img), 6))
        else:
            features = np.zeros((len(img), 22))
        for i, tile in enumerate(img):
            if ft_type == 0:
                hsv = extract_hsv(tile)
                rgb = extract_rgb(tile)
                features[i, :] = np.concatenate((hsv, rgb), axis=None)
            elif ft_type == 1:
                gray = extract_gray(tile, ft_type)
                features[i, :] = gray
            else:
                hsv = extract_hsv(tile)
                rgb = extract_rgb(tile)
                gray = extract_gray(tile, ft_type)
                features[i, :] = np.concatenate((hsv, rgb, gray), axis=None)
        img_features.append(features)

    return img_features


def ft_PCA(ft, n_components=2):
    mean_ft = np.mean(ft, axis=0)
    std_ft = np.std(ft, axis=0)
    ft_norm = (ft - mean_ft) / std_ft
    pca = PCA(n_components)
    ft_2D = pca.fit(ft_norm).transform(ft_norm)
    exp = pca.explained_variance_ratio_

    return ft_2D, exp


def ft_DBSCAN(ft_2D, epsilon, min_sample):
    clustering = DBSCAN(eps=epsilon, min_samples=min_sample).fit(ft_2D)
    return clustering.labels_
