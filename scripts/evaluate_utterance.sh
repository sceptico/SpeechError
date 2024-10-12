#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path/to/your_model.keras> <path/to/your_audio.wav>"
    exit 1
fi

MODEL_PATH=$1
AUDIO_FILE=$2

FEATURE_DIR=data/features
LABEL_DIR=data/labels
LABEL_INFO_CSV=data/metadata/label_info.csv
OUTPUT_LABELS_DIR=predictions/labels
JSON_DIR=data/whisperX_word
FEATURE_CONFIG=src/feature_extraction/feature.cfg

MODEL_FILENAME=$(basename "$MODEL_PATH")
MODEL_BASENAME="${MODEL_FILENAME%.*}"
OUTPUT_ANNOTATED_TRANSCRIPT="predictions/transcript_${MODEL_BASENAME}"

# Verify that the model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file '$MODEL_PATH' not found."
    exit 1
fi

# Verify that the audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file '$AUDIO_FILE' not found."
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$(dirname "$OUTPUT_ANNOTATED_TRANSCRIPT")" ]; then
    mkdir -p "$(dirname "$OUTPUT_ANNOTATED_TRANSCRIPT")"
fi

# Run evaluate_utterance.py
echo "Running evaluate_utterance.py"
python3 -m src.evaluation.evaluate_utterance \
    --model_path "$MODEL_PATH" \
    --audio_file "$AUDIO_FILE" \
    --features_dir "$FEATURE_DIR" \
    --labels_dir "$LABEL_DIR" \
    --label_info_csv "$LABEL_INFO_CSV" \
    --output_labels_dir "$OUTPUT_LABELS_DIR" \
    --json_dir "$JSON_DIR" \
    --output_annotated_transcript "$OUTPUT_ANNOTATED_TRANSCRIPT" \
    --feature_config "$FEATURE_CONFIG"
