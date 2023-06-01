import iar_project.save_evaluation_files as save_eval
import iar_project.segmentation as segmentation
import iar_project.test_clustering as clustering
import iar_project.utils as utils

GENERAL_PATH = "/Users/edouardkoehn/Documents/GitHub/iar_project/"
DATA_PATH = GENERAL_PATH + "data_project2/test2/"

group_ID = 19
nb_images = 12


# Load data
data = utils.import_test(DATA_PATH, nb_images)

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
