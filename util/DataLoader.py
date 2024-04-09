'''
The DataLoader class is used to load the data from the data_path and return the data.
'''

from .logger_setup import logger

import pandas as pd
import numpy as np
from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    CutSet,
)
from typing import List


class DataLoader:
    '''
    The DataLoader class has the following methods:
        - __init__(data_path): Constructor that initializes the data_path attribute.
        - get_data(): Method that returns the loaded data.
        - get_recordings(): Method that returns the loaded audio data.
        - get_supervisions(): Method that returns the supervision metadata.
        - get_cut_sets(): Method that returns the cuts.

    The DataLoader class uses the following third-party libraries:
        - pandas: Library for data manipulation and analysis.
        - lhotse: Library for working with speech and audio datasets.
        - typing: Library for type hints.

    The DataLoader class uses the following classes and functions from the lhotse library:
        - RecordingSet: Class that represents a set of recordings.
        - SupervisionSegment: Class that represents a supervision segment.
        - SupervisionSet: Class that represents a set of supervision segments.
        - CutSet: Class that represents a set of cuts.
        - from_dir: Function that creates a RecordingSet from audio files in a directory.
        - from_segments: Function that creates a SupervisionSet from a list of supervision segments.
        - from_manifests: Function that creates a CutSet from recordings and supervisions.
        - cut_into_windows: Method that cuts the recordings into windows of a specified duration.
        - pad: Method that pads the cuts to a specified duration.
        - compute_supervisions_frame_mask: Function that generates a mask for the features of a cut.
    '''

    def __init__(
        self,
        data_path: str,
        recording_dir: str,
        sampling_rate: int = 16000,
        cut_duration: float = 5.0
    ) -> None:
        '''
        Constructor that initializes the DataLoader object.

        Parameters:
            data_path: str, path to the data file
            recording_dir: str, path to the directory containing the audio files
            sampling_rate: int, sampling rate of the audio files
            cut_duration: float, duration of each cut in seconds

        Returns:
            None
        '''
        self.data_path = data_path
        self.recording_dir = recording_dir
        self.sampling_rate = sampling_rate
        self.cut_duration = cut_duration

        self.data = self._load_data()
        self.files = self._unique_recording_files()
        self.recordings = self._load_recordings()
        self.supervisions = self._create_supervisions()
        self.cut_sets = self._generate_cut_sets()

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

    def _unique_recording_files(self) -> List[str]:
        '''
        Find unique audio files in the data

        Parameters:
            None

        Returns:
            unique_recording_files: List[str], unique audio files in the data
        '''
        return self.data['file'].unique()

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

    def _extract_id_from_path(self, path) -> str:
        '''
        Extract the unique id from the file path
        ex. 'dataset\\audio\\ac\\ac083_2008-04-06.mp3' -> 'ac083'

        Parameters:
            path: WindowsPath, file path

        Returns:
            id: str, unique id
        '''
        path = str(path)
        return path.split('\\')[-1].split('_')[0]

    def _load_recordings(self) -> RecordingSet:
        '''
        Load audio files from the recording_dir using librosa and create Recording objects

        Parameters:
            None

        Returns:
            recordings: RecordingSet, RecordingSet object containing the audio data
        '''
        recordings = RecordingSet.from_dir(
            path=self.recording_dir,
            pattern='*.mp3',
            num_jobs=1,
            force_opus_sampling_rate=self.sampling_rate,
            recording_id=self._extract_id_from_path,
        )

        return recordings

    def _create_supervisions(self) -> SupervisionSet:
        '''
        Create supervision manifests from the data

        Parameters:
            None

        Returns:
            supervisions: SupervisionSet, supervision metadata
        '''
        supervisions = SupervisionSet.from_segments(
            segments=[
                self._generate_supervision(
                    id=index,
                    file=row['file'],
                    label=row['label'],
                    start=row['start'],
                    end=row['end']
                )
                for index, row in self.data.iterrows()
            ]
        )

        return supervisions

    def _generate_supervision(
        self,
        id: str,
        file: str,
        label: str,
        start: float,
        end: float
    ) -> SupervisionSegment:
        '''
        Generate a supervision segment

        Parameters:
            file: str, audio file name
            label: str, label of the supervision segment
            start: float, start time of the supervision segment
            end: float, end time of the supervision segment

        Returns:
            supervision: SupervisionSegment, supervision segment
        '''
        supervision = SupervisionSegment(
            id=id,
            recording_id=self._extract_id_from_file(file),
            start=start,
            duration=end-start,
            channel=0,
            custom={"label": label}
        )

        return supervision

    def _generate_cut_sets(self) -> CutSet:
        '''
        Generate cut sets from the recordings and supervisions
        Each cut is 5 seconds long

        Parameters:
            None

        Returns:
            cut_sets: CutSet, CutSet object containing the cuts
        '''
        cut_sets = CutSet.from_manifests(
            recordings=self.recordings,
            supervisions=self.supervisions,
        )

        # Original cuts are too long, cut them into 5-second windows
        cut_sets = cut_sets.cut_into_windows(duration=self.cut_duration)

        # Last cut may be shorter than the desired duration
        cut_sets = cut_sets.pad(duration=self.cut_duration)

        return cut_sets

    def get_data(self) -> pd.DataFrame:
        '''
        Return the loaded data

        Parameters:
            None

        Returns:
            data: DataFrame, loaded data
        '''
        return self.data

    def get_recordings(self) -> RecordingSet:
        '''
        Return the loaded audio data

        Parameters:
            None

        Returns:
            recordings: RecordingSet, RecordingSet object containing the audio data
        '''
        return self.recordings

    def get_supervisions(self) -> SupervisionSet:
        '''
        Return the supervision metadata

        Parameters:
            None

        Returns:
            supervisions: SupervisionSet, supervision metadata
        '''
        return self.supervisions

    def get_cut_sets(self) -> CutSet:
        '''
        Return the cuts

        Parameters:
            None

        Returns:
            cut_sets: CutSet, CutSet object containing the cuts
        '''
        return self.cut_sets

    def get_feature_mask(self, cut) -> np.ndarray:
        '''
        Generate a mask for the features of a cut

        Parameters:
            cut: Cut, input cut

        Returns:
            feature_mask: np.ndarray, mask for the features of the cut
        '''
        feature_mask = cut.compute_supervisions_frame_mask(cut)

        return feature_mask
