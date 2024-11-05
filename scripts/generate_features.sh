# PROCESS_NUM=4
# WAVE_LIST=data/metadata/wav_list.lst
# WAVE_DIR=data/audio
# TRANSCRIPT_DIR=data/whisperX
# FEATURE_DIR=data/features
# LABEL_DIR=data/labels
# FEATURE_CONFIG=src/feature_extraction/feature.cfg
# ANNOTATIONS_PATH=data/metadata/dataset.csv
# LABEL_INFO_DIR=data/metadata

# # Generate features
# echo "Running generate_features.py"
# if [ ! -d $FEATURE_DIR ]; then
#     mkdir -p $FEATURE_DIR
# fi

# python3 src/feature_extraction/generate_features.py --wav_list $WAVE_LIST --wav_dir $WAVE_DIR --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --feature_config $FEATURE_CONFIG --n_process $PROCESS_NUM

# # Generate labels
# echo "Running generate_labels.py"
# if [ ! -d $LABEL_DIR ]; then
#     mkdir -p $LABEL_DIR
# fi

# python3 src/feature_extraction/generate_labels.py --annotations_path $ANNOTATIONS_PATH --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --label_dir $LABEL_DIR --label_info_dir $LABEL_INFO_DIR --feature_config $FEATURE_CONFIG --n_process $PROCESS_NUM

PROCESS_NUM=4
WAVE_LIST=data/metadata/wav_list.lst
WAVE_DIR=data/audio
TRANSCRIPT_DIR=data/whisperX
FEATURE_DIR=data/features
LABEL_DIR=data/labels
FEATURE_CONFIG=src/feature_extraction/feature.cfg
ANNOTATIONS_PATH=data/metadata/dataset.csv
LABEL_INFO_DIR=data/metadata

# Generate features
echo "Running generate_features.py"
if [ ! -d $FEATURE_DIR ]; then
    mkdir -p $FEATURE_DIR
fi

/opt/anaconda3/envs/speech-error/bin/python src/feature_extraction/generate_features.py --wav_list $WAVE_LIST --wav_dir $WAVE_DIR --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --feature_config $FEATURE_CONFIG --n_process $PROCESS_NUM

# Generate labels
echo "Running generate_labels.py"
if [ ! -d $LABEL_DIR ]; then
    mkdir -p $LABEL_DIR
fi

/opt/anaconda3/envs/speech-error/bin/python src/feature_extraction/generate_labels.py --annotations_path $ANNOTATIONS_PATH --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --label_dir $LABEL_DIR --label_info_dir $LABEL_INFO_DIR --feature_config $FEATURE_CONFIG --n_process $PROCESS_NUM
