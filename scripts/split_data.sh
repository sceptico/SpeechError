LABEL_INFO_PATH=data/metadata/label_info.csv
OUTPUT_DIR=data/metadata
EVAL_RATIO=0.1
TEST_RATIO=0.1

# Split data
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
echo "Running split_data.py"

python3 src/feature_extraction/split_data.py --label_info_path $LABEL_INFO_PATH --output_dir $OUTPUT_DIR --eval_ratio $EVAL_RATIO --test_ratio $TEST_RATIO
