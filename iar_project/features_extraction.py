import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as sk
from skimage import feature
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


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
    for i in range(0, 3):
        rgb[i], rgb[i + 3], rgb[i + 6] = return_mean_std(im[:, :, i])

    return rgb


def extract_gray(im, ft_type):
    if ft_type == 1 or ft_type == 2:
        gray = np.zeros((10))
    else:
        gray = np.zeros((7))
    gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    gray[0], gray[1], gray[2] = return_mean_std(gray_im)
    dft_img = np.fft.fft2(gray_im)
    gray[3] = np.abs(dft_img[0, 0])
    entropy_img2 = entropy(gray_im, disk(9))
    entropy_img3 = entropy(gray_im, disk(3))
    entropy_img1 = entropy(gray_im, disk(5))
    gray[0], gray[1], gray[2] = return_mean_std(entropy_img3)
    gray[3], gray[4], gray[5] = return_mean_std(entropy_img1)
    gray[6], gray[7], gray[8] = return_mean_std(entropy_img2)
    med_img = cv2.medianBlur(gray_im, 3)
    edges = feature.canny(med_img, sigma=1)
    edges.dtype = "uint8"
    gray[9] = np.sum(edges)
    if ft_type == 1 or ft_type == 0:

        gray[7] = np.abs(dft_img[1, 1])
        gray[8] = np.abs(dft_img[2, 2])

    return gray


def get_features(seg_img, ft_type):
    img_features = []
    for img in seg_img:
        if ft_type == 0:
            features = np.zeros((len(img), 22))
        elif ft_type == 1:
            features = np.zeros((len(img), 9))
        else:
            features = np.zeros((len(img), 28))
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


def get_features_gray(seg_img):
    img_features = []
    for img in seg_img:
        features = np.zeros((len(img), 4))
        for i, tile in enumerate(img):
            gray_im = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
            features[i, 0] = np.mean(gray_im)
            entropy_img = entropy(gray_im, disk(5))
            features[i, 1] = np.median(entropy_img)
            dft_img = np.fft.fft2(gray_im)
            features[i, 2] = np.abs(dft_img[0, 0])
            features[i, 3] = np.std(gray_im)

        img_features.append(features)

    return img_features


def get_features_color(seg_img):
    img_features = []
    for img in seg_img:
        features = np.zeros((len(img), 4))
        for i, til in enumerate(img):
            tile = til[10:118, 10:118, :]
            gray_im = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
            hsv_im = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
            # features[i,0] = np.median(hsv_im[:,:,0])
            entropy_img1 = entropy(gray_im, disk(5))
            # features[i,0] = np.std(entropy_img1)
            features[i, 0] = np.median(entropy_img1)  # 5 7
            # features[i,2] = np.std(entropy_img3)
            features[i, 2] = np.median(tile[:, :, 1])  # 0 1 2
            # features[i,1] = np.median(tile[:,:,2])
            # features[i,2] = np.median(tile[:,:,0])
            features[i, 1] = np.median(hsv_im[:, :, 2])  # 0 1 4 6
            dft_img = np.fft.rfft2(gray_im)
            features[i, 3] = np.mean(np.abs(dft_img))
            # features[i,7] = np.min(gray_im)
            # features[i,8] = np.max(gray_im)
            # features[i,0] = np.std(gray_im)
            # med_img = cv2.medianBlur(gray_im, 3)
            # edges = feature.canny(med_img, sigma=1)
            # edges.dtype='uint8'
            # features[i,10] = np.sum(edges)

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
    labels = clustering.labels_

    return labels
