# Commai.ai Speed Challenge
This is my take on the speed challenge proposed by comma.ai. You can check out the link to 
the original contenst page here https://github.com/commaai/speedchallenge.

# Instructions 
Download the dataset from the commaai repo and extract it.  

Prepare the data set by running `./preprocessor.py` - This will save the images in a folder called data_preprocessed as well as a file called `preprocessed.csv` including path to image, time stamp and speed. This file is used by both the training file and the predict file as well.  

Train the model by running `./training.by` - This will save the weights in the file model-weights.h5.  

Predict by running `./predict.py` - This will use the saved model weights and test on some test data from the preprocessed.py.  
