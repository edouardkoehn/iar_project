import cv2
import features_extraction as extraction
import matplotlib.pyplot as plt
import numpy as np
import utils as utils
from sklearn.cluster import KMeans


def norm_features(ft):
    mean_ft = np.mean(ft, axis=0)
    std_ft = np.std(ft, axis=0)
    ft_norm = (ft - mean_ft) / std_ft

    return ft_norm


def pca_loop(ft, tresh):
    exp = 0
    nb_comp = 0
    while np.sum(exp) < tresh:
        nb_comp += 1
        ft_reduced, exp = extraction.ft_PCA(ft[0], nb_comp)

    return ft_reduced


def return_color_or_gray(tiles):
    idx_gray = []
    idx_color = []
    for t, tile in enumerate(tiles):
        if (tile[40:80, 40:80, 0] == tile[40:80, 40:80, 1]).all():
            idx_gray.append(t)
        else:
            idx_color.append(t)

    return idx_color, idx_gray


def test_KMeans(STATE, img_test, ft, K=3):
    clustered_puzzle = []
    STATE = True

    kmeans = KMeans(n_clusters=K, n_init="auto", init="random").fit(ft)
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
            elif len(puzzle) == 3 or len(puzzle) == 2 or len(puzzle) == 1:
                out += 1
        else:
            if len(puzzle) > 3:
                STATE = False
        clustered_puzzle.append(puzzle)
    if out > 1:
        STATE = False
    return clustered_puzzle, unique_lab, STATE


def clustering(seg_img, nb_ft):
    clu_img = []
    clu_labels = []
    for imID, img_test in enumerate(seg_img):
        # print(len(img_test))
        features = extraction.get_features_color([img_test])
        ft_raw = features[0]
        ft_norm = norm_features(features[0])
        STATE = False
        k = 4
        p = 0
        compo = 2
        while not STATE:
            if p <= nb_ft - 1:
                ft = ft_raw[:, p : p + 1]
                clu, clu_lab, STATE = test_KMeans(STATE, img_test, ft, K=k)
                k -= 1
                if k < 3:
                    p += 1
                    k = 4
            elif compo <= nb_ft and p > nb_ft - 1:
                ft, exp = extraction.ft_PCA(ft_raw, compo)
                clu, clu_lab, STATE = test_KMeans(STATE, img_test, ft, K=k)
                k -= 1
                if k < 3:
                    compo += 1
                    k = 4
            else:
                clu, clu_lab, STATE = test_KMeans(STATE, img_test, ft_norm[:, 0:2], K=4)
                print(f"No solution for image {imID}")
                break

        clu_img.append(clu)
        clu_labels.append(clu_lab)

    return clu_img, clu_labels


def save_puzzles(clu_img, path_out):
    IMG_SIZE = 128
    solution = []
    out = []
    for i, img in enumerate(clu_img):
        image1 = []
        outlier = []
        nb = 0
        for p, puzzle in enumerate(img):
            nb_of_tile = len(puzzle)
            if nb_of_tile == 9:
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

            else:
                outliers = []
                for t, tile in enumerate(puzzle):
                    outliers.append(tile)
                    export_file = (
                        f"{path_out}/outlier_{str(i).zfill(2)}_{str(t).zfill(2)}.png"
                    )
                    cv2.imwrite(export_file, cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
                outlier.append(outliers)

        solution.append(image1)
        out.append(outlier)


def plot_clustering_results(clu_img, sizefig, lon):
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


if __name__ == "__main__":

    # How many images ?
    nb_im = 12
    k = 0  # from which image
    seg_img = utils.import_seg_results(k, nb_im)
    sol_img = utils.import_solution()
    sol_img = [sol_img]
    plot = False
    IMG_SIZE = 128

    clu_img, clu_labels = clustering(seg_img, 4)

    path_out = utils.check_output_clustering_folder(dataset_used=2)

    save_puzzles(clu_img, path_out)

    if plot:
        plot_clustering_results(clu_img, sizefig=(14, 8), lon=16)
