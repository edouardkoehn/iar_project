# Project IAPR -Group 19
This repository contain all the source code for the IAPR-Project 23.
All the required documentation of the repository is contain in this README.
# Installation
1) Clone the repository

```bash
git clone https://github.com/edouardkoehn/iar_project.git
```
- Create your virtual env
```bash
conda create -n iar_project python=3
conda activate iar_project
```
2) Install poetry (all the dependencies managnement was done using [Poetry](https://python-poetry.org/))
```bash
pip install poetry
```
3) Install the dependancies
```bash
poetry install
```
4) Modifiy the project path in the file ```/iar_project/utils.py```

```bash
#Path to the repository
L:7 GENERAL_PATH = "/Users/jeanpaul/Documents/GitHub/iar_project"
```
# Running the pipeline
The complete pipelin run with a single script :```/iar_project/main.py```. This script run the following steps:
1) Import the data
2) Build the segementation
3) Compute the features
4) Produce the clustering based on the features
5) Solve the puzzle

First, follow the installation instructions. Then, verify that you modify the path to the repository (see Installation_part4).

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
## 3) Clustering
## 4) Solving the puzzle
