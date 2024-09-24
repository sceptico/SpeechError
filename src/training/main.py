"""
main.py

Main script for training models using k-fold cross-validation.

Functions:
- main: Main function for training models using k-fold cross-validation.
"""

import argparse
from model_trainer import ModelTrainer


def main(config_paths):
    """
    Main function for training models using k-fold cross-validation.
    """
    # Instantiate the ModelTrainer with the provided config paths
    trainer = ModelTrainer(config_paths)

    # Run the training and evaluation process
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model with cross-validation.")
    parser.add_argument(
        "config_paths",
        nargs='+',
        type=str,
        help="List of configuration files for different experiments."
    )
    args = parser.parse_args()
    main(args.config_paths)
