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


from util.logger_setup import logger
from util.AudioDataLoader import AudioDataLoader


DATA_PATH = './dataset/dataset_test.csv'
RECORDING_DIR = './dataset/audio'


def main():
    # Load the data
    data_loader = AudioDataLoader(
        data_path=DATA_PATH,
        recording_dir=RECORDING_DIR,
        sampling_rate=16000,
        cut_duration=5.0
    )
    logger.info(f"Data loaded from {DATA_PATH}")

    # Print and check if the recordings/supervisions are loaded correctly
    print(data_loader.data, '\n')
    print(data_loader.recordings, '\n')
    print(data_loader.supervisions, '\n')

    cuts = data_loader._cut_sets
    print(cuts.describe(), '\n')
    
    # Cut with a 1 .0s supervision
    print(cuts[66], '\n')
    
    # Cut without a supervision
    print(cuts[67], '\n')


if __name__ == '__main__':
    main()
