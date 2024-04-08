'''
Author: HUI, Macarious

This is the main training script for the speech error detection model. It loads the data, trains the model, and saves the model to disk.
Podcast audio files are used as training data, and they are manually labeled with the location of speech errors.
The model is trained to predict the location of speech errors in the audio files.

The training script has the following steps:
    1. Load the data using the DataLoader class.
    2. Preprocess the data.
    3. Train the model.
    4. Save the model to disk.
    
The training script uses the following classes and functions:
    - DataLoader: Class that loads the data from the data_path and returns the data.
    - Preprocessor: Class that preprocesses the data for training.
    - Model: Class that defines the speech error detection model.
    - train_model: Function that trains the model.
    - save_model: Function that saves the model to disk.
'''

# Local imports
from util.logger_setup import logger
from util.DataLoader import DataLoader

# Third party imports


DATA_PATH = './dataset/dataset_test.csv'

def main():    
    # Load the data
    data_loader = DataLoader(DATA_PATH)
    data = data_loader.get_data()
    logger.info(f"Data loaded from {DATA_PATH}")
    
    print(data)
    
if __name__ == '__main__':
    main()