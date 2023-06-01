import numpy as np

import iar_project.CNN as cnn
import iar_project.save_evaluation_files as save_eval
import iar_project.segmentation as segmentation
import iar_project.test_clustering as clustering
import iar_project.utils as utils

GENERAL_PATH = (
    "/Users/begue/OneDrive/Documents/EPFL/Master/Cours/MA2/IAPR/projet/iar_project"
)
DATA_PATH = GENERAL_PATH + "/data_project2/test/"

group_ID = 19
nb_images = 1
DL = True

# Load data
data = utils.import_test(DATA_PATH, nb_images)
if not DL:
    for i, im in enumerate(data):
        solution = []
        # Segmentation
        mask = segmentation.compute_segementation_filters([im])
        solution.append(mask[0])
        segmentation.extract_segemented_object(im, mask[0], i, dataset_use=2)
        seg_img = utils.import_seg_results(i, i + 1)
        # Features extraction and clustering
        clu_img, clu_labels, features = clustering.clustering(seg_img, i, nb_ft=7)
        solution.append(features)
        # Save results (home made version) and create puzzles
        path_out = utils.check_output_clustering_folder(dataset_used=2)
        clusters, puzzles = clustering.save_puzzles(clu_img[0], path_out, i)

        solution.append(clusters)

        # Solve puzzle
        solution.append(puzzles)

        # Export solution
        save_eval.export_solutions(i, solution, path=DATA_PATH, group_id=group_ID)

else:
    for i, im in enumerate(data):
        solution = []
        mask = segmentation.compute_segementation_filters([im])
        solution.append(mask[0])
        segmentation.extract_segemented_object(im, mask[0], i, dataset_use=2)
        solution.append(np.ones((2, 2)))
        clusters = cnn.CNN_classification(i)
        solution.append(clusters)
        solution.append(np.ones((2, 2)))
        save_eval.export_solutions(
            i, solution, path=DATA_PATH, group_id=group_ID, CNN=DL
        )
