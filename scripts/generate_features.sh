PROCESS_NUM=20
WAVE_LIST=data/metadata/wav_list.lst
WAVE_DIR=data/audio
TRANSCRIPT_DIR=data/whisperX
FEATURE_DIR=data/features
LABEL_DIR=data/labels
FEATURE_CONFIG=src/feature_extraction/feature.cfg
ANNOTATION_PATH=data/metadata/dataset.csv
LABEL_INFO_DIR=data/metadata

# Generate features
if [ ! -d $FEATURE_DIR ]; then
    mkdir -p $FEATURE_DIR
fi

python3 src/feature_extraction/generate_features.py --wav_list $WAVE_LIST --wav_dir $WAVE_DIR --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --feature_config $FEATURE_CONFIG --n_process $PROCESS_NUM

# Generate labels
if [ ! -d $LABEL_DIR ]; then
    mkdir -p $LABEL_DIR
fi

python3 src/feature_extraction/generate_labels.py --wav_list $WAVE_LIST --annotation_path $ANNOTATION_PATH --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --label_dir $LABEL_DIR --label_info_dir $LABEL_INFO_DIR --feature_config $FEATURE_CONFIG --n_process $PROCESS_NUM
