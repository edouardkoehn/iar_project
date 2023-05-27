import features_extraction as extraction
import matplotlib.pyplot as plt
import numpy as np
import utils as utils
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk

nb_im = 12
k = 0
seg_img = utils.import_seg_results(k, nb_im)
sol_img = utils.import_solution()
sol_img = [sol_img[5:8]]

seg = []
for img in seg_img:
    img2 = []
    for tiles in img:
        tilescrop = tiles[10:118, 10:118, :]
        img2.append(tilescrop)
    seg.append(img2)


def visu():
    features = extraction.get_features([seg[2]], 2)
    """
    plt.figure(figsize=(20,10))
    for i in range(0,31):
        plt.subplot(3, 11, i + 1)
        plt.imshow(sol_img[0][i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    """
    features_name = [
        "Mean H",
        "Mean S",
        "Mean V",
        "Median H",
        "Median S",
        "Median V",
        "STD H",
        "STD S",
        "STD V",
        "Mean R",
        "Mean G",
        "Mean B",
        "Median R",
        "Median G",
        "Median B",
        "STD R",
        "STD G",
        "STD B",
        "Mean Entropy 3",
        "Median Entropy 3",
        "STD Entropy 3",
        "Mean Entropy 5",
        "Median Entropy 5",
        "STD Entropy 5",
        "Mean Entropy 9",
        "Median Entropy 9",
        "STD Entropy 9",
        "Edges",
    ]

    for ft in features:
        mean_ft = np.mean(ft, axis=0)
        std_ft = np.std(ft, axis=0)
        ft_norm = (ft - mean_ft) / std_ft
        fig, axs = plt.subplots(6, 1, figsize=(12, 12))
        nb_im = ft_norm.shape[0]
        for i in range(0, 6):
            for j in range(0, nb_im):
                axs[i].scatter(ft_norm[j, i], 1, label=j)
            axs[i].title.set_text(features_name[i])
        axs[i].legend(loc=(1.04, 0))
        plt.tight_layout()

        fig, axs = plt.subplots(6, 1, figsize=(12, 12))
        for i in range(6, 12):
            for j in range(0, nb_im):
                axs[i - 6].scatter(ft_norm[j, i], 1, label=j)
            axs[i - 6].title.set_text(features_name[i])
        axs[i - 6].legend(loc=(1.04, 0))
        plt.tight_layout()

        fig, axs = plt.subplots(6, 1, figsize=(12, 12))
        for i in range(12, 18):
            for j in range(0, nb_im):
                axs[i - 12].scatter(ft_norm[j, i], 1, label=j)
            axs[i - 12].title.set_text(features_name[i])
        axs[i - 12].legend(loc=(1.04, 0))
        plt.tight_layout()

        fig, axs = plt.subplots(6, 1, figsize=(12, 12))
        for i in range(18, 24):
            for j in range(0, nb_im):
                axs[i - 18].scatter(ft_norm[j, i], 1, label=j)
            axs[i - 18].title.set_text(features_name[i])
        axs[i - 18].legend(loc=(1.04, 0))
        plt.tight_layout()

        fig, axs = plt.subplots(4, 1, figsize=(12, 12))
        for i in range(24, 28):
            for j in range(0, nb_im):
                axs[i - 24].scatter(ft_norm[j, i], 1, label=j)
            axs[i - 24].title.set_text(features_name[i])
        axs[i - 24].legend(loc=(1.04, 0))
        plt.tight_layout()

        plt.figure(figsize=(12, 12))
        for i in range(0, 3):
            plt.subplot(4, 5, i + 1)
            plt.imshow(sol_img[0][i])
            plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    visu()
else:
    visu()
