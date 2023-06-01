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


# Segmentation
Make sure that you followed the instrucrtions before running the code.
All the methods use for the segmentation can be found in the file ```iar_project/segmentation.py```

To run the segementation, you juste need to run the noto ```notebooks/0_Segmentation.ipynb```. The results of the segmentation would be strored in the folder ```data_project/segmentation_results``` or ```data_project2/segmentation_results``` depending on the database used.

# Clustering
# Solving the puzzle
# Installation
- Clone the repo

```bash
git clone https://github.com/edouardkoehn/WM_Atlas.git
```
- Create your virtual env
```bash
conda create -n iar_project python=3
conda activate iar_project
```
- Install poetry
```bash
pip install poetry
```
- Install the modul and set up the precommit
```bash
poetry install
pre-commit install
poetry env info
```
- Modifiy the project path in the file ```iar_project.py```

```bash
#Path to the repository
L:6 GENERAL_PATH = "/Users/jeanpaul/Documents/GitHub/iar_project"
```
