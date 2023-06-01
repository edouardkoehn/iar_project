import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import iar_project.features_extraction as extraction


def ft_PCA(ft, n_components=2):
    """Method for extracting the first two component of the PCA on the
    features matrix"""
    ft_norm = norm_features(ft)
    pca = PCA(n_components)
    ft_2D = pca.fit(ft_norm).transform(ft_norm)
    exp = pca.explained_variance_ratio_
    return ft_2D[:, 0:2], exp


def norm_features(ft):
    """Method for normalizing the data (z-score)
    x_norm= (x-mean)/std"""
    scaler = StandardScaler().fit(ft)
    ft_norm = scaler.transform(ft)
    return ft_norm


def test_KMeans(STATE, img_test, ft, K=3):
    """Method for checking if the produced clustets are coherent"""
    clustered_puzzle = []
    STATE = True

    kmeans = KMeans(n_clusters=K, n_init=30, init="random").fit(ft)
    labels = kmeans.labels_
    unique_lab = np.unique(labels)
    # nb_tiles = []
    out = 0
    for l, lab in enumerate(unique_lab):
        # print(lab)
        idx = np.argwhere(labels == lab)
        puzzle = []
        for i, id in enumerate(idx):
            # print(id[0])
            puzzle.append(img_test[id[0]])
        if lab != -1.0:
            if (
                len(puzzle) != 9
                and len(puzzle) != 12
                and len(puzzle) != 16
                and len(puzzle) != 1
                and len(puzzle) != 2
                and len(puzzle) != 3
            ):
                STATE = False
                # print(lab)
            elif len(puzzle) < 9:
                out += 1
        else:
            if len(puzzle) > 3:
                print("Fail")
                STATE = False
        clustered_puzzle.append(puzzle)
    if out > 1:
        STATE = False
    return clustered_puzzle, unique_lab, STATE


def clustering(seg_img, im_i, nb_ft):
    """Pipeline for the feature selection and the clustering"""
    clu_img = []
    clu_labels = []
    for imID, img_test in enumerate(seg_img):
        print(f"Processing Image {im_i}")
        features = extraction.get_features_color([img_test])
        ft_raw = features[0]
        ft_norm = norm_features(features[0])
        STATE = False
        k = 4
        p = 0
        compo = 7
        while not STATE:
            if p <= nb_ft - 1:
                print(f"    Trying Kmeans 1 feature:    K={k}   f_{p}")
                ft = ft_raw[:, p : (p + 1)]
                clu, clu_lab, STATE = test_KMeans(STATE, img_test, ft, K=k)
                if not STATE and p <= nb_ft - 2:
                    print(f"    Trying Kmeans 2 features:   K={k}   f_{p,p+1}")
                    ft = ft_norm[:, p : (p + 2)]
                    clu, clu_lab, STATE = test_KMeans(STATE, img_test, ft, K=k)
                    if STATE:
                        break
                k -= 1
                if k < 3:
                    p += 1
                    k = 4
            elif compo == nb_ft and p > nb_ft - 1:
                print(f"    Trying PCA 7 features:      K={k}   ")
                ft, exp = ft_PCA(ft_raw, compo)
                clu, clu_lab, STATE = test_KMeans(STATE, img_test, ft, K=k)
                k -= 1
                if k < 3:
                    compo += 1
                    k = 4
            else:
                print("    Trying Kmeans on all features...")
                ft = ft_norm
                clu, clu_lab, STATE = test_KMeans(STATE, img_test, ft, K=4)
                if not STATE:
                    print(f"    No solution for image {im_i}")
                    break

        clu_img.append(clu)
        clu_labels.append(clu_lab)

    return clu_img, clu_labels, ft


def save_puzzles(img, path_out, i):
    """Method for saving the puzzles tiles"""
    IMG_SIZE = 128
    image1 = []
    outlier = []
    cluster = []
    nb = 0
    for p, puzzle in enumerate(img):
        nb_of_tile = len(puzzle)
        if nb_of_tile == 9:
            cluster.append(puzzle)
            puz = np.zeros((3 * IMG_SIZE, 3 * IMG_SIZE, 3))
            # print(puz.shape)
            a = 0
            b = 0
            for t, tile in enumerate(puzzle):
                puz[
                    a * IMG_SIZE : (a + 1) * IMG_SIZE,
                    b * IMG_SIZE : (b + 1) * IMG_SIZE,
                    :,
                ] = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                a += 1
                if a == 3:
                    a = 0
                    b += 1
            image1.append(puz)
            export_file = (
                f"{path_out}/solution_{str(i).zfill(2)}_{str(nb).zfill(2)}.png"
            )
            cv2.imwrite(export_file, puz)
            nb += 1
        elif nb_of_tile == 12:
            cluster.append(puzzle)
            puz = np.zeros((4 * IMG_SIZE, 3 * IMG_SIZE, 3))
            a = 0
            b = 0
            for t, tile in enumerate(puzzle):
                puz[
                    a * IMG_SIZE : (a + 1) * IMG_SIZE,
                    b * IMG_SIZE : (b + 1) * IMG_SIZE,
                    :,
                ] = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                a += 1
                if a == 4:
                    a = 0
                    b += 1
            image1.append(puz)
            export_file = (
                f"{path_out}/solution_{str(i).zfill(2)}_{str(nb).zfill(2)}.png"
            )
            cv2.imwrite(export_file, puz)
            nb += 1
        elif nb_of_tile == 16:
            cluster.append(puzzle)
            puz = np.zeros((4 * IMG_SIZE, 4 * IMG_SIZE, 3))
            a = 0
            b = 0
            for t, tile in enumerate(puzzle):
                puz[
                    a * IMG_SIZE : (a + 1) * IMG_SIZE,
                    b * IMG_SIZE : (b + 1) * IMG_SIZE,
                    :,
                ] = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                a += 1
                if a == 4:
                    a = 0
                    b += 1
            image1.append(puz)
            export_file = (
                f"{path_out}/solution_{str(i).zfill(2)}_{str(nb).zfill(2)}.png"
            )
            cv2.imwrite(export_file, puz)
            nb += 1
    for p, puzzle in enumerate(img):
        nb_of_tile = len(puzzle)
        if nb_of_tile != 9 and nb_of_tile != 12 and nb_of_tile != 16:
            cluster.append(puzzle)
            outliers = []
            for t, tile in enumerate(puzzle):
                outliers.append(tile)
                export_file = (
                    f"{path_out}/outlier_{str(i).zfill(2)}_{str(t).zfill(2)}.png"
                )
                cv2.imwrite(export_file, cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
            outlier.append(outliers)

    return cluster, image1


def plot_clustering_results(clu_img, sizefig, lon):
    """Utils method for plotting the clustering"""
    for imID, img in enumerate(clu_img):
        fig, axs = plt.subplots(len(img), lon, figsize=sizefig)
        fig.suptitle(f"Image_{imID}")
        for p, puzzle in enumerate(img):
            for i, tile in enumerate(puzzle):
                if i > 15:
                    print("plus que 16 image")
                    break
                if len(img) == 1:
                    axs[i].imshow(tile)
                    axs[i].axis("off")
                else:
                    axs[p, i].imshow(tile)
                    axs[p, i].axis("off")

            plt.tight_layout()
    plt.show()
