AUDIO_DIR=data/audio
LIST_OUTPUT=data/metadata/wav_list.lst
SAMPLING_RATE=16000
PROCESS_NUM=4
WAVE_LIST=$LIST_OUTPUT
WAVE_DIR=data/audio
TRANSCRIPT_DIR=data/whisperX
FEATURE_DIR=data/features
LABEL_DIR=data/labels
FEATURE_CONFIG=src/feature_extraction/feature.cfg
ANNOTATIONS_PATH=data/metadata/dataset.csv
LABEL_INFO_DIR=data/metadata
LABEL_INFO_PATH="$LABEL_INFO_DIR/label_info.csv"
OUTPUT_DIR=$LABEL_INFO_DIR
EVAL_RATIO=0.1
TEST_RATIO=0.1
TRAIN_CSV_PATH="$LABEL_INFO_DIR/train.csv"
EVAL_CSV_PATH="$LABEL_INFO_DIR/eval.csv"
TEST_CSV_PATH="$LABEL_INFO_DIR/test.csv"
EPOCHS=50
BATCH_SIZE=64

# Convert mp3 to wav
echo "Converting mp3 to wav"
python src/audio_processing/convert_mp3_to_wav.py --audio_dir $AUDIO_DIR --output $AUDIO_DIR --sample_rate $SAMPLING_RATE

# Generate audio list
echo "Generating audio list"
python src/audio_processing/generate_audio_list.py --audio_dir $AUDIO_DIR --output $LIST_OUTPUT

# Generate features
echo "Extracting features"
if [ ! -d $FEATURE_DIR ]; then
    mkdir -p $FEATURE_DIR
fi
python3 src/feature_extraction/generate_features.py --wav_list $WAVE_LIST --wav_dir $WAVE_DIR --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --feature_config $FEATURE_CONFIG --n_process $PROCESS_NUM

# Generate labels
echo "Generating labels"
if [ ! -d $LABEL_DIR ]; then
    mkdir -p $LABEL_DIR
fi
python3 src/feature_extraction/generate_labels.py --annotations_path $ANNOTATIONS_PATH --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --label_dir $LABEL_DIR --label_info_dir $LABEL_INFO_DIR --feature_config $FEATURE_CONFIG --n_process $PROCESS_NUM

# Split data
echo "Splitting data into train, eval, and test sets"
python3 src/feature_extraction/split_data.py --label_info_path $LABEL_INFO_PATH --output_dir $OUTPUT_DIR --eval_ratio $EVAL_RATIO --test_ratio $TEST_RATIO

# Train model
echo "Training model"
python3 src/training/main.py --train_csv_path $TRAIN_CSV_PATH --eval_csv_path $EVAL_CSV_PATH --test_csv_path $TEST_CSV_PATH --epochs $EPOCHS --batch_size $BATCH_SIZE
