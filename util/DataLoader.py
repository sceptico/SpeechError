'''
The DataLoader class is used to load the data from the data_path and return the data.

The DataLoader class has the following methods:
    - __init__(data_path): Constructor that initializes the data_path attribute.
    - load_data(): Method that loads the data from the data_path.
    - get_data(): Method that returns the loaded data.
'''

from .logger_setup import logger

import librosa
import pandas as pd
import numpy as np
import os
# from lhotse import CutSet
from typing import List, Dict, Any


class DataLoader:
    def __init__(
            self,
            data_path: str,
            recording_dir: str,
            sampling_rate: int = 16000
    ) -> None:
        '''
        Constructor that initializes the DataLoader object.

        Parameters:
            data_path: str, path to the data file
            recording_dir: str, path to the directory containing the audio files
            sampling_rate: int, sampling rate of the audio files

        Returns:
            None
        '''
        self.data_path = data_path
        self.recording_dir = recording_dir
        self.sampling_rate = sampling_rate

        self.data = self._load_data()
        self.files = self._unique_recording_files()
        self.recordings = self._load_recordings()
        self.supervisions = self._generate_supervisions()

    def _load_data(self) -> pd.DataFrame:
        '''
        Load data from the data_path

        Parameters:
            None

        Returns:
            data: DataFrame, loaded data
        '''
        try:
            data = pd.read_csv(self.data_path)

        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            data = None
            exit()

        return data

    def _load_recordings(self) -> List[Dict[str, Any]]:
        '''
        Load all audio files from the recording_dir.

        Parameters:
            None

        Returns:
            audio_data: List[Dict[str, Any]], list of audio data
        '''
        recording_files = self._unique_recording_files()
        audio_data = []

        for recording_file in recording_files:
            recording_id = self._extract_id_from_file(recording_file)
            recording_path = os.path.join(self.recording_dir, recording_file)
            sampling_rate = self.sampling_rate

            audio_data.append({
                'recording_id': recording_id,
                'recording_file': recording_file,
                'recording_path': recording_path,
                'sampling_rate': sampling_rate,
            })

        return audio_data

    def _load_recording(self, recording_file: str) -> np.array:
        '''
        Load audio file from the recording_dir using librosa
        If an audio file cannot be loaded, remove data related to the audio file from self.data

        Parameters:
            recording_file: str, name of the audio file

        Returns:
            recording: np.array, audio data
        '''
        recording_file_path = os.path.join(self.recording_dir, recording_file)

        try:
            recording, _ = librosa.load(
                recording_file_path, sr=self.sampling_rate)

        except Exception as e:
            logger.error(f"Error loading recording {recording_file}: {e}")
            recording = None

        # If audio is not loaded, skip all data related to the audio file
        if recording is None:
            self.data = self.data[self.data['file'] != recording_file]
            self.data.reset_index(drop=True, inplace=True)

        return recording

    def _unique_recording_files(self) -> List[str]:
        '''
        Find unique audio files in the data

        Parameters:
            None

        Returns:
            unique_recording_files: List[str], unique audio files in the data
        '''
        return self.data['file'].unique()

    def _generate_supervisions(self) -> List[Dict[str, Any]]:
        '''
        Generate supervision metadata for all speech errors.
        This is used to create CutSet objects in Lhotse.

        Parameters:
            None

        Returns:
            supervisions: List[Dict[str, Any]], list of supervision metadata
        '''
        supervisions = []
        for index, row in self.data.iterrows():
            recording_id = self._extract_id_from_file(row['file'])
            recording_file = row['file']
            label = row['label']
            start_time = row['start']
            end_time = row['end']

            supervision = self._generate_supervision(
                recording_id, recording_file, label, start_time, end_time)
            supervisions.append(supervision)

        return supervisions

    def _generate_supervision(
        self,
        recording_id: str,
        recording_file: str,
        label: str,
        start_time: float,
        end_time: float,
    ) -> Dict[str, Any]:
        '''
        Generate supervision metadata for each speech error

        Parameters:
            recording_id: str, unique id for the speech error
            recording_file: str, name of the audio file
            label: str, label for the speech error
            start_time: float, start time of the speech error
            end_time: float, end time of the speech error

        Returns:
            supervision: Dict[str, Any], supervision metadata
        '''
        supervision = {
            'recording_id': recording_id,
            'recording_file': recording_file,
            'label': label,
            'start_time': start_time,
            'end_time': end_time
        }

        return supervision

    def _extract_id_from_file(self, file: str) -> str:
        '''
        Extract the unique id from the file name
        ex. '/ac/ac083_2008-04-06.mp3' -> 'ac083'

        Parameters:
            file: str, file name

        Returns:
            id: str, unique id
        '''
        return file.split('/')[2].split('_')[0]

    def get_data(self) -> pd.DataFrame:
        '''
        Return the loaded data

        Parameters:
            None

        Returns:
            data: DataFrame, loaded data
        '''
        return self.data

    def get_recordings(self) -> List[Dict[str, Any]]:
        '''
        Return the loaded audio data

        Parameters:
            None

        Returns:
            recordings: List[Dict[str, Any]], loaded audio data
        '''
        return self.recordings

    def get_supervisions(self) -> List[Dict[str, Any]]:
        '''
        Return the supervision metadata

        Parameters:
            None

        Returns:
            supervisions: List[Dict[str, Any]], supervision metadata
        '''
        return self.supervisions
