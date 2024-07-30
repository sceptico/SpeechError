EXPERIMENT_CONFIG_PATH=experiments/default.cfg

# Train model
echo "Training model"
python3 src/training/training.py $EXPERIMENT_CONFIG_PATH
