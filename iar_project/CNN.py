import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model, save_model

GENERAL_PATH = (
    "/Users/begue/OneDrive/Documents/EPFL/Master/Cours/MA2/IAPR/projet/iar_project"
)
DATA_PATH = GENERAL_PATH + "/data_project2/"
MODEL_PATH = GENERAL_PATH + "/CNN_model/"


def extract_tiles(
    img_idx="0_", path=DATA_PATH, folder="segmentation_results", puzzle_size=128
):
    files = os.listdir(os.path.join(path, folder))
    solution_files = [
        file for file in files if file.startswith(img_idx)
    ]  #### en changeant le numéro ici on change d image train

    new_test_data = []

    for image_index, filename in enumerate(solution_files):
        path_solution = os.path.join(path, folder, filename)

        im = Image.open(path_solution).convert("RGB")

        # Resize the image to the desired size
        im = im.resize((puzzle_size, puzzle_size))

        # Convert the image to a NumPy array
        im_array = np.array(im)

        new_test_data.append(im_array)

    # Convert the list of image arrays to a NumPy array
    new_test_data = np.array(new_test_data)

    return new_test_data


def create_cluster(predicted_labels, new_test_data):
    # Find unique predicted labels
    unique_predicted_labels = np.unique(predicted_labels)

    # Create an empty dictionary to store the images
    image_dict = {label: [] for label in unique_predicted_labels}

    # Loop over each predicted label and corresponding image
    for label, image in zip(predicted_labels, new_test_data):
        image_dict[label].append(image)

    return image_dict


def merge_groups(groups):
    merged = np.concatenate(groups, axis=0)
    return merged


def merging_cluster(image_dict):
    filtered_groups = [
        (label, image_dict[label])
        for label in image_dict
        if (
            (len(image_dict[label]) % 3 != 0 and len(image_dict[label]) != 15)
            and len(image_dict[label]) < 16
        )
    ]

    # Print sizes of the groups before the merge operation
    print("Initial group sizes:")
    for label, group in filtered_groups:
        print(f"Group {label} size: {len(group)}")

    best_mse = float("inf")
    best_groups = None

    if len(filtered_groups) > 1:
        # Generate all combinations of 2 groups
        for (label1, group1), (label2, group2) in combinations(filtered_groups, 2):
            # Merge groups
            merged = merge_groups([group1, group2])
            # Flatten images
            flat_images = merged.reshape(len(merged), -1)
            # Calculate MSE
            mse = mean_squared_error(flat_images[:-1], flat_images[1:])
            # If MSE is better, update best MSE and groups
            if mse < best_mse:
                best_mse = mse
                best_groups = (label1, label2)

        # Merge the best groups
        merged_group1, merged_group2 = [image_dict[label] for label in best_groups]
        image_dict[best_groups[0]] = merge_groups([merged_group1, merged_group2])
        del image_dict[best_groups[1]]  # Remove the merged group from the dictionary

    return image_dict


def CNN_classification(i):
    idx = str(i) + "_"
    clusters = []
    new_test_data = extract_tiles(
        img_idx=idx, path=DATA_PATH, folder="segmentation_results", puzzle_size=128
    )
    # Load the model
    loaded_model = load_model(MODEL_PATH + "full_dataV2.h5")
    # Predictions
    predictions = loaded_model.predict(new_test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    image_dict = create_cluster(predicted_labels, new_test_data)

    image_dict = merging_cluster(image_dict)

    for label, images in image_dict.items():
        num_images = len(images)  # Get the number of images for the current group
        if num_images == 9 or num_images == 12 or num_images == 16:
            clusters.append(images)

    for label, images in image_dict.items():
        num_images = len(images)  # Get the number of images for the current group
        if num_images != 9 and num_images != 12 and num_images != 16:
            clusters.append(images)

    return clusters
