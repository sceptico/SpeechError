import os
import json
import re
import numpy as np
import pandas as pd
import configparser
import string


class TranscriptAnnotator:
    def __init__(self, audio_file, predicted_labels_dir, actual_labels_dir, json_dir, label_info_csv, feature_config_path):
        self.audio_file = audio_file
        self.predicted_labels_dir = predicted_labels_dir
        self.actual_labels_dir = actual_labels_dir
        self.json_dir = json_dir
        self.label_info_csv = label_info_csv
        self.feature_config_path = feature_config_path

        self.audio_base_name = os.path.splitext(
            os.path.basename(audio_file))[0]

        # Initialize feature extraction parameters
        self.hop_length_samples = None
        self.win_length_samples = None
        self.sr = None

        self.label_times = {}
        self.predicted_error_intervals = []
        self.actual_error_intervals = []
        self.predicted_errors = []
        self.actual_errors = []
        self.transcript_data = None

        self.load_feature_config()

    def load_feature_config(self):
        config = configparser.ConfigParser()
        config.read(self.feature_config_path)

        feature_config = config['feature']
        self.hop_length_seconds = float(feature_config.get('hop_length', 0.02))
        self.win_length_seconds = float(feature_config.get('win_length', 0.04))
        self.sr = int(feature_config.get('sr', 16000))

        self.hop_length_samples = int(self.hop_length_seconds * self.sr)
        self.win_length_samples = int(self.win_length_seconds * self.sr)

    def load_label_info(self):
        # Read the label_info_csv to create a mapping from label files to start and end times
        label_info = pd.read_csv(self.label_info_csv)
        for index, row in label_info.iterrows():
            feature_file = row['feature_file']
            feature_file_name = os.path.basename(feature_file)
            label_file_name = feature_file_name.replace('.npy', '_labels.npy')
            start_time = row['start_time']
            end_time = row['end_time']
            self.label_times[label_file_name] = (start_time, end_time)

    def find_json_file(self):
        # First, try to find the JSON file directly in json_dir
        json_file_path = os.path.join(
            self.json_dir, f"{self.audio_base_name}.json")

        if not os.path.exists(json_file_path):
            # Extract the podcast prefix using regular expression
            match = re.match(r'^([a-zA-Z]+)', self.audio_base_name)
            if match:
                podcast_prefix = match.group(1)
                sub_json_dir = os.path.join(self.json_dir, podcast_prefix)
                json_file_path = os.path.join(
                    sub_json_dir, f"{self.audio_base_name}.json")

                if not os.path.exists(json_file_path):
                    # Search recursively in json_dir
                    for root, dirs, files in os.walk(self.json_dir):
                        if f"{self.audio_base_name}.json" in files:
                            json_file_path = os.path.join(
                                root, f"{self.audio_base_name}.json")
                            break
                    else:
                        raise FileNotFoundError(
                            f"JSON transcription file for {self.audio_base_name} not found in {self.json_dir} or its subdirectories."
                        )
                else:
                    print(f"Found JSON file at: {json_file_path}")
            else:
                # Proceed to recursive search
                for root, dirs, files in os.walk(self.json_dir):
                    if f"{self.audio_base_name}.json" in files:
                        json_file_path = os.path.join(
                            root, f"{self.audio_base_name}.json")
                        print(f"Found JSON file at: {json_file_path}")
                        break
                else:
                    raise FileNotFoundError(
                        f"JSON transcription file for {self.audio_base_name} not found in {self.json_dir} or its subdirectories."
                    )
        else:
            print(f"Found JSON file at: {json_file_path}")
        return json_file_path

    def load_transcript_data(self):
        json_file_path = self.find_json_file()
        with open(json_file_path, 'r') as f:
            self.transcript_data = json.load(f)

    def collect_error_intervals(self):
        self.predicted_error_intervals = self._collect_error_intervals_from_dir(
            self.predicted_labels_dir)
        self.actual_error_intervals = self._collect_error_intervals_from_dir(
            self.actual_labels_dir)

    def _collect_error_intervals_from_dir(self, labels_dir):
        error_intervals = []
        # Construct the prefix for label files corresponding to the current audio file
        label_file_prefix = f"{self.audio_base_name}_"

        for label_file_name in os.listdir(labels_dir):
            if label_file_name.startswith(label_file_prefix) and label_file_name.endswith('_labels.npy'):
                label_file_path = os.path.join(labels_dir, label_file_name)
                labels = np.load(label_file_path)

                if label_file_name in self.label_times:
                    start_time, end_time = self.label_times[label_file_name]
                else:
                    print(
                        f"Start and end times for {label_file_name} not found. Skipping.")
                    continue

                num_frames = labels.shape[0]
                current_interval = None

                for i in range(num_frames):
                    label = labels[i]
                    if label == 1:
                        t_start = start_time + \
                            (i * self.hop_length_samples) / self.sr
                        t_end = t_start + self.win_length_seconds
                        t_end = min(t_end, end_time)

                        if current_interval is None:
                            current_interval = [t_start, t_end]
                        else:
                            # Extend the current interval
                            current_interval[1] = t_end
                    else:
                        if current_interval is not None:
                            error_intervals.append(tuple(current_interval))
                            current_interval = None

                # Append the last interval if it's still open
                if current_interval is not None:
                    error_intervals.append(tuple(current_interval))
            else:
                # This file does not correspond to the current audio file
                continue

        return error_intervals

    def map_errors_to_transcript(self):
        self.predicted_errors, self.predicted_text = self._annotate_transcript(
            self.predicted_error_intervals)
        self.actual_errors, self.actual_text = self._annotate_transcript(
            self.actual_error_intervals)

    def _annotate_transcript(self, error_intervals):
        errors = []
        annotated_text = ''
        in_error_phrase = False
        current_error = None

        for segment in self.transcript_data['segments']:
            for word_info in segment['words']:
                word = word_info['word']
                word_start = word_info['start']
                word_end = word_info['end']

                # Separate leading and trailing punctuation from word
                word_clean = word.strip()
                leading_punct = ''
                trailing_punct = ''
                while word_clean and word_clean[0] in string.punctuation:
                    leading_punct += word_clean[0]
                    word_clean = word_clean[1:]
                while word_clean and word_clean[-1] in string.punctuation:
                    trailing_punct = word_clean[-1] + trailing_punct
                    word_clean = word_clean[:-1]

                # If the entire word is punctuation, keep it as is
                if not word_clean:
                    word_clean = leading_punct + trailing_punct
                    leading_punct = ''
                    trailing_punct = ''

                # Check if the word overlaps with any error interval
                is_error = any(
                    error_start < word_end and word_start < error_end
                    for error_start, error_end in error_intervals
                )

                if is_error:
                    if not in_error_phrase:
                        # Start of error phrase
                        if annotated_text and not annotated_text.endswith(' '):
                            annotated_text += ' '
                        annotated_text += leading_punct + '[' + word_clean
                        in_error_phrase = True

                        # Start a new error
                        current_error = {
                            'start': word_start,
                            'end': word_end,
                            'text': word_clean
                        }
                    else:
                        # Continuing error phrase
                        annotated_text += ' ' + leading_punct + word_clean
                        # Update current error
                        current_error['end'] = word_end
                        current_error['text'] += ' ' + word_clean

                    # Add trailing punctuation inside brackets
                    annotated_text += trailing_punct
                else:
                    if in_error_phrase:
                        # End of error phrase
                        annotated_text += ']'  # Close brackets
                        in_error_phrase = False
                        # Save the current error
                        errors.append(current_error)
                        current_error = None

                    # Add non-error word with punctuation
                    if annotated_text and not annotated_text.endswith(' '):
                        annotated_text += ' '
                    annotated_text += leading_punct + word_clean + trailing_punct

        # Close any open error phrase at the end
        if in_error_phrase:
            annotated_text += ']'
            # Save the current error
            errors.append(current_error)

        annotated_text = annotated_text.strip()

        return errors, annotated_text

    def save_results_to_json(self, output_path):
        # Check if the output directory exists
        os.makedirs(output_path, exist_ok=True)

        result = {
            'predicted': {
                'errors': self.predicted_errors,
                'text': self.predicted_text
            },
            'actual': {
                'errors': self.actual_errors,
                'text': self.actual_text
            }
        }

        output_file = os.path.join(
            output_path, f"{self.audio_base_name}_with_errors.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        print(f"Results saved to {output_file}")

    def run(self, output_path):
        self.load_label_info()
        self.load_transcript_data()
        self.collect_error_intervals()
        self.map_errors_to_transcript()
        self.save_results_to_json(output_path)
