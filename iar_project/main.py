import matplotlib.pyplot as plt
import numpy as np

import iar_project.clustering as clustering
import iar_project.CNN as cnn
import iar_project.save_evaluation_files as save_eval
import iar_project.segmentation as segmentation
import iar_project.solve_puzzle as solving
import iar_project.utils as utils

# General config for the pipeline
DATA_PATH = utils.GENERAL_PATH + "/data_project2/test2/"
OUTPUT_PATH = utils.GENERAL_PATH + "/data_project2/"
group_ID = 19
nb_images = 1
DL = True

# Check if CNN model exists
utils.check_model_path()
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
        unsolved_puzzles = utils.load_unsolved_images(i, len(puzzles))
        solved_puzzles = []
        for unsolved_puzzle, ind in zip(
            unsolved_puzzles, np.arange(0, len(unsolved_puzzles))
        ):
            print(f"        Solving puzzle _{i}_{ind}")
            solved_puzzles.append(
                solving.solve_puzzle(
                    unsolved_puzzle,
                    piece_size=128,
                    population=100,
                    generations=50,
                    termination_threshold=30,
                    verbose=False,
                )
            )
        solution.append(solved_puzzles)

        # Export solution
        save_eval.export_solutions(i, solution, path=OUTPUT_PATH, group_id=group_ID)

else:
    for i, im in enumerate(data):
        solution = []
        # Segementation
        mask = segmentation.compute_segementation_filters([im])
        solution.append(mask[0])
        segmentation.extract_segemented_object(im, mask[0], i, dataset_use=2)
        solution.append(np.random.rand(2, 2))

        # Feature extraction and clustering
        clusters = cnn.CNN_classification(i)
        solution.append(clusters)
        path_out = utils.check_output_clustering_folder(dataset_used=2)
        _, puzzles = clustering.save_puzzles(clusters, path_out, i)

        # Solve puzzle
        unsolved_puzzles = utils.load_unsolved_images(i, len(puzzles))
        solved_puzzles = []
        for unsolved_puzzle, ind in zip(
            unsolved_puzzles, np.arange(0, len(unsolved_puzzles))
        ):
            print(f"        Solving puzzle _{i}_{ind}")
            solved_puzzles.append(
                solving.solve_puzzle(
                    unsolved_puzzle,
                    piece_size=128,
                    population=100,
                    generations=50,
                    termination_threshold=30,
                    verbose=False,
                )
            )

        solution.append(solved_puzzles)

        # Export
        save_eval.export_solutions(
            i, solution, path=OUTPUT_PATH, group_id=group_ID, CNN=DL
        )
