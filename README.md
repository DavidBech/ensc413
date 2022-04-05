# Project branch for ensc413 course

David Bechert
Colton Koop

Jan - Apr 2022

# PROJECT MYM
This repsitory used this repository as a starting point:
https://github.com/andrewleeunderwood/project_MYM

# Files and directories
## deepLearningModel.py -- analizeAcuracy.py -- trainModel.py
These files are used to train and run some graph producing code on the data located in the "data" directory

## RunData
This directory stores infromation from running trainModel and analizeAcuracy. This information includes model weights, a confusion matrix, information about the run (batch size, epochs)

## Data
This directory contains all the data and data creations scripts. The `create_data_*` scripts crop the full images and store them into classified fiels based on the json files. `The setup_data.py` script copies a portion of these cropped images into a `train` and `test` directory in order to run the model on a subset of the data without the need to continuously crop the origional images again.
