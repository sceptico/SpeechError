PROCESS_NUM=20
WAVE_LIST=data/metadata/wav_list.lst
WAVE_DIR=data/audio
TRANSCRIPT_DIR=data/whisperX
FEATURE_DIR=data/feature
FEATURE_CONFIG=code/feature_extraction/feature.cfg

python code/feature_extraction/generate_features.py --wav_list $WAVE_LIST --wav_dir $WAVE_DIR --transcript_dir $TRANSCRIPT_DIR --feature_dir $FEATURE_DIR --feature_config $FEATURE_CONFIG --n_process $PROCESS_NUM
