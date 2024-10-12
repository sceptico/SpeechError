'''
evaluate_utterance.py

This script evaluates an utterance by making predictions on the audio features,
comparing the predicted labels with the actual labels, and annotating the transcript
based on the errors in the prediction.

It requires the following arguments:
- model_path: Path to the model file (.keras)
- audio_file: Path to the audio file (.wav)
- features_dir: Directory containing .npy feature files
- labels_dir: Directory containing actual label files
- label_info_csv: Path to the label info CSV file
- output_labels_dir: Directory to save predicted label files
- json_dir: Directory containing JSON transcription files
- output_annotated_transcript: Path to save annotated transcript

Usage:
python evaluate_utterance.py \
    --model_path <model_path> \
    --audio_file <audio_file> \
    --features_dir <features_dir> \
    --labels_dir <labels_dir> \
    --label_info_csv <label_info_csv> \
    --output_labels_dir <output_labels_dir> \
    --json_dir <json_dir> \
    --output_annotated_transcript <output_annotated_transcript>
    
Example:
python evaluate_utterance.py \
    --model_path models/baseline.keras \
    --audio_file data/audio/ac003.wav \
    --features_dir data/features \
    --labels_dir data/labels \
    --label_info_csv data/metadata/label_info.csv \
    --output_labels_dir predictions/labels \
    --json_dir data/whisperX_word \
    --output_annotated_transcript predictions/annotated_transcript.txt
'''

import argparse
import os
import numpy as np
import pandas as pd

from src.evaluation.model_predictor import ModelPredictor
from src.evaluation.label_comparator import LabelComparator
from src.evaluation.transcript_annotator import TranscriptAnnotator


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate utterance and annotate transcript.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model file (.keras)')
    parser.add_argument('--audio_file', type=str, required=True,
                        help='Path to the audio file (.wav)')
    parser.add_argument('--feature_config', type=str, required=True,
                        help='Path to the feature extraction configuration file')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing .npy feature files')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Directory containing actual label files')
    parser.add_argument('--label_info_csv', type=str,
                        required=True, help='Path to the label info CSV file')
    parser.add_argument('--output_labels_dir', type=str, required=True,
                        help='Directory to save predicted label files')
    parser.add_argument('--json_dir', type=str, required=True,
                        help='Directory containing JSON transcription files')
    parser.add_argument('--output_annotated_transcript', type=str,
                        required=True, help='Path to save annotated transcript')

    args = parser.parse_args()

    # Append model name to the output_labels_dir
    args.output_labels_dir = args.output_labels_dir + \
        f"_{args.model_path.split('/')[-1].split('.')[0]}/"

    # 1. Make predictions and create label files
    predictor = ModelPredictor(
        model_path=args.model_path,
        features_dir=args.features_dir,
        output_labels_dir=args.output_labels_dir,
        audio_file=args.audio_file
    )
    predictor.run_predictions()

    # 2. Compare predicted labels with actual labels
    comparator = LabelComparator(
        predicted_labels_dir=args.output_labels_dir,
        actual_labels_dir=args.labels_dir
    )
    comparator.compare_labels()

    # 3. Annotate the transcript using the new TranscriptAnnotator class
    annotator = TranscriptAnnotator(
        audio_file=args.audio_file,
        predicted_labels_dir=args.output_labels_dir,
        actual_labels_dir=args.labels_dir,
        json_dir=args.json_dir,
        label_info_csv=args.label_info_csv,
        feature_config_path=args.feature_config,
    )
    annotator.run(output_path=args.output_annotated_transcript)


if __name__ == '__main__':
    main()
