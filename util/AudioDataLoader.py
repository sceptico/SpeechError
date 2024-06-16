"""
The AudioDataLoader class is used to load the data from the data_path and return the data.

To-do:
- Add voice detection to cut out audio at beginning and end of podcast
- Use whisperX transcription timestamps to cut out audio
"""

from .logger_setup import logger

import pandas as pd
import numpy as np
from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    CutSet,
)
from lhotse.features.mfcc import TorchaudioMfcc
from lhotse.dataset.vad import VadDataset
import torch
from typing import List, Dict


class AudioDataLoader:
    """
    AudioDataLoader Class
    =====================

    The AudioDataLoader class loads and manages audio data for speech error classification.

    Public Methods
    --------------
    - __init__(data_path): Constructor that initializes the data_path attribute.

    Public Properties
    -----------------
    - data: loaded data.
    - audio_files: unique audio files in the data.
    - recordings: loaded audio data.
    - supervisions: supervision metadata.
    - cut_sets: cuts.

    Dependencies
    ------------
    The AudioDataLoader class uses the following third-party libraries:

    - [pandas](https://pandas.pydata.org): Library for data manipulation and analysis.
    - [lhotse](https://lhotse.readthedocs.io): Library for working with speech and audio datasets.
    - [typing](https://docs.python.org/3/library/typing.html): Library for type hints.
    - [numpy](https://numpy.org): Library for numerical computing.
    - [torch](https://pytorch.org): Library for machine learning.

    Lhotse Classes and Functions
    ----------------------------
    The AudioDataLoader class uses the following classes and functions from the lhotse library:

    - [RecordingSet](https://lhotse.readthedocs.io/en/latest/api.html#lhotse.audio.RecordingSet): Class that represents a set of recordings.
    - [SupervisionSegment](https://lhotse.readthedocs.io/en/latest/api.html#lhotse.supervision.SupervisionSegment): Class that represents a supervision segment.
    - [SupervisionSet](https://lhotse.readthedocs.io/en/latest/api.html#lhotse.supervision.SupervisionSet): Class that represents a set of supervision segments.
    - [CutSet](https://lhotse.readthedocs.io/en/latest/api.html#lhotse.cut.CutSet): Class that represents a set of cuts.
    - [TorchaudioMfcc](https://lhotse.readthedocs.io/en/latest/api.html#lhotse.features.mfcc.TorchaudioMfcc): Class that extracts MFCC features from audio data.
    - [VadDataset](https://lhotse.readthedocs.io/en/latest/datasets.html#lhotse.dataset.vad.VadDataset): Class that represents a VAD dataset.
    """

    def __init__(
        self,
        data_path: str,
        recording_dir: str,
        sampling_rate: int = 16000,
        cut_duration: float = 5.0
    ) -> None:
        """
        Constructor that initializes the AudioDataLoader object.

        Parameters:
            data_path (str): Path to the data file.
            recording_dir (str): Path to the directory containing the audio files.
            sampling_rate (int, optional): target sampling rate of the audio files. Defaults to 16000.
            cut_duration (float, optional): Duration of each cut in seconds. Defaults to 5.0.

        Returns:
            None
        """
        self.data_path = data_path
        self.recording_dir = recording_dir
        self.sampling_rate = sampling_rate
        self.cut_duration = cut_duration

        self._data = self._load_data()
        self._audio_files = self._unique_recording_files()
        self._recordings = self._load_recordings()
        self._supervisions = self._create_supervisions()
        self._cut_sets = self._generate_cut_sets()

    @property
    def data(self) -> pd.DataFrame:
        """
        Return the loaded data

        Parameters:
            None

        Returns:
            DataFrame, loaded data
        """
        return self._data

    @property
    def audio_files(self) -> List[str]:
        """
        Return the unique audio files in the data

        Parameters:
            None

        Returns:
            List[str], unique audio files in the data
        """
        return self._audio_files

    @property
    def recordings(self) -> RecordingSet:
        """
        Return the loaded audio data

        Parameters:
            None

        Returns:
            RecordingSet, RecordingSet object containing the audio data
        """
        return self._recordings

    @property
    def supervisions(self) -> SupervisionSet:
        """
        Return the supervision metadata

        Parameters:
            None

        Returns:
            SupervisionSet, supervision metadata
        """
        return self._supervisions

    @property
    def cut_sets(self) -> CutSet:
        """
        Return the cuts

        Parameters:
            None

        Returns:
            CutSet, CutSet object containing the cuts
        """
        return self._cut_sets

    def _load_data(self) -> pd.DataFrame:
        """
        Load data from the data_path.

        Returns:
            DataFrame: Loaded data.
        """
        try:
            data = pd.read_csv(self.data_path)

        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            data = None
            exit()

        return data

    def _unique_recording_files(self) -> List[str]:
        """
        Find unique audio files in the data.

        Returns:
            List[str]: Unique audio file names.
        """
        return self._data['file'].unique()

    def _extract_id_from_file(self, file: str) -> str:
        """
        Extract the unique id from the file name
        ex. '/ac/ac083_2008-04-06.mp3' -> 'ac083'

        Parameters:
            file: str, file name

        Returns:
            str: Unique id extracted from the file name.
        """
        return file.split('/')[2].split('_')[0]

    def _extract_id_from_path(self, path) -> str:
        """
        Extract the unique id from the file path
        ex. 'dataset\\audio\\ac\\ac083_2008-04-06.mp3' -> 'ac083'

        Parameters:
            path: WindowsPath, file path

        Returns:
            str: Unique id extracted from the file name.
        """
        path = str(path)
        return path.split('\\')[-1].split('_')[0]

    def _load_recordings(self) -> RecordingSet:
        """
        Load audio files from the recording_dir using librosa and create Recording objects.

        Returns:
            RecordingSet: RecordingSet object containing the audio data.
        """
        recordings = RecordingSet.from_dir(
            path=self.recording_dir,
            pattern='*.mp3',
            num_jobs=1,
            recording_id=self._extract_id_from_path,
        )

        return recordings

    def _create_supervisions(self) -> SupervisionSet:
        """
        Create supervision manifests from the data.

        Returns:
            SupervisionSet: SupervisionSet object containing the supervision metadata.
        """
        supervisions = SupervisionSet.from_segments(
            segments=[
                self._generate_supervision(
                    id=index,
                    file=row['file'],
                    label=row['label'],
                    start=row['start'],
                    end=row['end']
                )
                for index, row in self._data.iterrows()
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
        """
        Generate a supervision segment.

        Parameters:
            id (str): Unique identifier for the supervision segment.
            file (str): Audio file name.
            label (str): Label of the supervision segment.
            start (float): Start time of the supervision segment.
            end (float): End time of the supervision segment.

        Returns:
            SupervisionSegment: Supervision segment object.
        """
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
        """
        Generate cut sets from the recordings and supervisions.

        Returns:
            CutSet: CutSet object containing the cuts.
        """
        cut_sets = CutSet.from_manifests(
            recordings=self._recordings,
            supervisions=self._supervisions,
        )

        # Original cuts are too long, cut them into 5-second windows
        cut_sets = cut_sets.cut_into_windows(duration=self.cut_duration)

        # Last cut may be shorter than the desired duration
        cut_sets = cut_sets.pad(duration=self.cut_duration)

        return cut_sets

    def _extract_features(
        self,
        feature_extractor=None,
        sampling_rate: int = None,
    ) -> torch.Tensor:
        """
        Extract features from the audio data using the feature extractor.

        Parameters:
            feature_extractor: Optional feature extractor object.
            sampling_rate (int, Optional): Sampling rate of the audio data. Defaults to None.

        Returns:
            torch.Tensor: Extracted features.
        """
        if feature_extractor is None:
            feature_extractor = TorchaudioMfcc()

        if sampling_rate == 0:
            sampling_rate = self.sampling_rate

        audio_samples = self._cut_sets.load_audio()

        features = feature_extractor.extract_batch(
            samples=audio_samples,
            sampling_rate=sampling_rate
        )

        return features

    def _generate_vad_dataset(self) -> Dict[str, torch.Tensor]:
        """
        Generate a VAD dataset from the cuts.

        Returns:
            Dict[str, torch.Tensor]: VAD dataset.
        """
        vad_dataset = VadDataset()
        vad_dataset = vad_dataset.__getitem__(self._cut_sets)

        return vad_dataset
