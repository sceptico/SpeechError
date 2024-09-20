EXPERIMENT_CONFIG_PATH=experiments/default.cfg

# Train model
echo "Training model"
python3 src/training/main.py $EXPERIMENT_CONFIG_PATH
