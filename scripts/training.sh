TRAIN_CSV_PATH=data/metadata/train.csv
EVAL_CSV_PATH=data/metadata/eval.csv
TEST_CSV_PATH=data/metadata/test.csv
LOG_DIR=logs
EPOCHS=30
BATCH_SIZE=64
MODEL_NAME=baseline

# Train model
echo "Training model"
python3 src/training/training.py --model_name $MODEL_NAME --train_csv_path $TRAIN_CSV_PATH --eval_csv_path $EVAL_CSV_PATH --test_csv_path $TEST_CSV_PATH --log_dir $LOG_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE
