# Project IAPR -Group 19
This repository contain all the source code for the IAPR-Project 23.
All the required documentation of the repository is contain in this README.
# Installation
1) Clone the repository

```bash
git clone https://github.com/edouardkoehn/iar_project.git
```
2) Create your virtual env
```bash
conda create -n iar_project python=3
conda activate iar_project
```
3) Install poetry (all the dependencies managnement was done using [Poetry](https://python-poetry.org/))
```bash
pip install poetry
```
4) Install the dependancies
```bash
poetry install
pip install tensorflow==2.12.0
```
5) Modifiy the project path in the file ```/iar_project/utils.py```

```bash
#Path to the repository
L:7 GENERAL_PATH = "/Users/jeanpaul/Documents/GitHub/iar_project"
```

6) Download the model for the CNN from https://drive.google.com/file/d/1-KgZ9ay6BIJjkRLsIsWJdCh0ZAHiwfpz/view, and place it under the folder CNN_model at the root of the repository.


# Running the pipeline
The complete pipelin run with a single script :```/iar_project/main.py```. This script run the following steps:
1) Import the data
2) Build the segementation
3) Compute the features
4) Produce the clustering based on the features
5) Solve the puzzle

First, follow the installation instructions. Then, verify that you modify the path to the repository (see Installation_part4).
We implemented two clustering algorithms (with a CNN or pure Image processing). You can select which one you want to use by setting the variable DL in main.py.
```bash
l16: DL = True # True = CNN algorithm, False = Standard clustering with Kmeans
```


To run the code, go to the root of the repository and then call the script:
```bash
#Move to the root
cd /MyPATH/iar_project
#Run the code
python iar_project/main.py
```
The pipeline display several log during the runs. And exemple of the console output would be the following:
```bash
Processing Image 0
    Trying Kmeans 1 feature:    K=4   f_0
    Trying Kmeans 2 features:   K=4   f_(0, 1)
    Saving solutions in folder:  /Users/Vetterli/code/iar_project/data_project2/solutions_group_19
```
The ouptut of the script would be saved under ```/data_project2/solutions_group_19``` using the naming convention given by the TA's.

# Work description
## 1) Segmentation
For the segmentation, our approach was to combine mutliple simple filter to get an accurate segementation. Each filter is a binary mask that has been produced using a combination of simple segmentation algorithm:
- [Thresholding specific channels (HSV, RGB, Gray scale)](https://en.wikipedia.org/wiki/Balanced_histogram_thresholding)
- [Canny edge detection](https://en.wikipedia.org/wiki/Canny_edge_detector)
- [Morphological operation (Opening, Closing)](https://fr.wikipedia.org/wiki/Morphologie_math%C3%A9matique)

We applied a cleaning step on each to those filter to ensure that they fit our requirements. The cleaning step created square object and check that the area of the object is in a specific range.

Once those filter have been produce, we combine them to get a good segmentation mask.
<p align="center">
<img src=figures/fig_segmentation_0.png width=50%>
</p>


From the segmentation mask, we extract the object and save them as single png file. To save the object as single file, we need to rotate the object to ensure that it is aligned with the image referential.

All the source code for the segementation is contained in the file  ```iar_project/segementation.py```

## 2) Feature extraction
We extract the following 7 features from the images:

Shape:
- Entropy median with two different radii
- Mean of the power spectrum of the gray scale DFT

Pixel intensity:
- Median Value in the HSV space
- Mean Saturation in the HSV space
- Std Saturation channel in the HSV space
- Median green channel in the RGB space

The goal was to keep the number of features at the minimum and hence focus more on the quality than the quantity. As it can be seen on the following figure, for some images, only one feature is already enough to separate the tiles in the feature space.

<p align="center">
<img src=figures/fig_features1.png width=75%>
</p>

For other combination of tiles, 2 or more features are needed.

All the source code for the feature extraction is contained in the file  ```iar_project/features_extraction.py```

## 3) Clustering
For the clustering, Kmeans is used and the following assumption are taken:

- There are either 2 or 3 clusters of puzzle + 1 cluster of outlier
- The puzzles are either 3x3, 3x4, 4x4
- There are either 1, 2 or 3 outliers

These assumptions define what is a "coherent result" of the clustering. Hence if the result of the clustering violates one of these assumption, the solution is discarded and a new clustering is done with other features. Our clustering strategy consists to test with K=4 and K=3 clusters while looping over the features first individually and then by pairs until finding a solution that is coherent. If no coherent solution is found, a PCA is applied on the 7 features and the 2 explaining the most variance are kept. If the result of the clustering after the PCA is still not coherent, a last iteration is done using all the features. Note that every time that the clustering is done with more than one feature, the features are first normalized.

All the source code for the clustering is contained in the file  ```iar_project/clustering.py```

We implement a second algo for clustering based on CNN and it consist of 6 different layers that separate all images in labels. Give a puzzle piece as input, and it gives the label associate as output. You can find the code to train it as a notebook in the file CNN_train.

## 4) Solving the puzzle

We used a genetic algorithm that was based on a code that we changed to use for our project. Here is the link: https://github.com/nemanja-m/gaps

This algorithm is based on how genetic mute and the goal is to find an arrangement of puzzle pieces that forms a complete image. 

To be quick, here are the main steps of the algo.

- Initialization: Start with a population of randomly shuffled puzzle pieces.
- Fitness Evaluation: Calculate a fitness score for each puzzle piece arrangement based on how well the pieces fit together. 
- Analyse: Choose the best puzzle piece arrangements from the population based on their fitness scores.
- Crossover: Combine the selected puzzle piece arrangements to create new offspring arrangements. 
- Mutation: Introduce random changes or alterations to the genetic material of the puzzle piece arrangements. 
- Termination: Repeat steps 2-5 for a certain number of generations or until a specific fitness threshold is reached. This determines when the algorithm should stop.
- Solution: Once the termination condition is met, the best puzzle piece arrangement found is returned as the solution to the jigsaw puzzle.

<p align="center">
<img src=figures/fig_solver_0.png width=50%>
</p>



