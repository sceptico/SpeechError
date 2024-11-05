# Define paths to the train and evaluation CSVs
TRAIN_DATA_PATH="data/metadata/label_train_resampled.csv"
EVAL_DATA_PATH="data/metadata/eval_context.csv"

# Define model configurations
MODEL_CONFIGS=("rbf:1.0:scale" "rbf:0.5:scale" "linear:1.0" "poly:1.0::3")

# Run the training and evaluation script with model configurations passed correctly
echo "Starting SVM training and evaluation..."
/opt/anaconda3/envs/speech-error/bin/python src/SVM/train_and_evaluate.py \
    --train_data_path "$TRAIN_DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --model_configs "${MODEL_CONFIGS[@]}"
