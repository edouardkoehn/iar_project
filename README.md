# Project IAPR
Repository containing the code for the IAPR-project.
# Segmentation
Make sure that you followed the instrucrtions before running the code.
All the methods use for the segmentation can be found in the file ```iar_project/segmentation.py```

To run the segementation, you juste need to run the noto ```notebooks/0_Segmentation.ipynb```. The results of the segmentation would be strored in the folder ```data_project/segmentation_results``` or ```data_project2/segmentation_results``` depending on the database used.

# Clustering
# Solving the puzzle
## Installation
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
