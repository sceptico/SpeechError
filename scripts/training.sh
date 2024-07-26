TRAIN_CSV_PATH=data/metadata/train.csv
EVAL_CSV_PATH=data/metadata/eval.csv
TEST_CSV_PATH=data/metadata/test.csv
EPOCHS=50
BATCH_SIZE=64

# Train model
echo "Training model"
python3 src/training/training.py --train_csv_path $TRAIN_CSV_PATH --eval_csv_path $EVAL_CSV_PATH --test_csv_path $TEST_CSV_PATH --epochs $EPOCHS --batch_size $BATCH_SIZE
