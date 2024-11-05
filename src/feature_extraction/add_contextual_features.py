"""
add_contextual_features.py

This script adds contextual features to each data sample by aggregating values within a specified time frame window and generates corresponding contextual labels. It also creates a `class` column derived from `label_count`, which is needed for consistency across training, evaluation, and test datasets.

Contextual features provide additional information about surrounding data points, which can enhance model performance by including temporal context. The script saves new feature and label files with context and updates the original metadata with paths to these files.

Usage:
    python add_contextual_features.py --label_info_path <label_info_path> --output_path <output_path> --contextual_feature_dir <contextual_feature_dir> --contextual_label_dir <contextual_label_dir> --window_size <window_size>

Arguments:
    - label_info_path (str): Path to the label_info.csv file containing metadata with paths to feature and label files and `label_count`.
    - output_path (str): Path where the updated label_info_context.csv with contextual features and labels metadata will be saved.
    - feature_dir (str): Directory containing the original feature (.npy) files.
    - contextual_feature_dir (str): Directory to save the new .npy files with contextual features.
    - contextual_label_dir (str): Directory to save the new .npy files with contextual labels.
    - window_size (int): Size of the contextual window (number of previous and following frames to include in aggregation). Default is 5.

Example:
    python add_contextual_features.py --label_info_path /data/metadata/label_info.csv --output_path /data/metadata/label_info_context.csv --feature_dir /data/features --contextual_feature_dir /data/contextual_features --contextual_label_dir /data/contextual_labels --window_size 5
"""

import os
import numpy as np
import pandas as pd
import argparse

def add_contextual_features_and_labels(df, feature_dir, contextual_feature_dir, contextual_label_dir, window_size=5):
    """
    Adds contextual features and generates contextual labels by calculating the mean, standard deviation,
    minimum, maximum, and median values within a specified time window around each sample.
    Saves the new contextual features and labels as .npy files and returns an updated DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing file paths and other metadata.
        feature_dir (str): Directory where the .npy feature files are located.
        contextual_feature_dir (str): Directory where the new contextual feature .npy files will be saved.
        contextual_label_dir (str): Directory where the new contextual label .npy files will be saved.
        window_size (int): The number of frames before and after to include in the contextual aggregation.

    Returns:
        pd.DataFrame: Updated DataFrame with paths to contextual .npy files, contextual labels, additional metadata, and a `class` column derived from `label_count`.
    """
    contextual_features = []  # List to store paths to generated contextual feature files
    contextual_labels = []    # List to store paths to generated contextual label files
    
    # Ensure the output directories for contextual features and labels exist
    os.makedirs(contextual_feature_dir, exist_ok=True)
    os.makedirs(contextual_label_dir, exist_ok=True)
    
    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        feature_file = row['feature_file']
        label_count = row['label_count']  # Original label (0 for non-error, 1 for error)
        
        # Load the feature data from the .npy file
        feature_path = os.path.join(feature_dir, os.path.basename(feature_file))
        feature_data = np.load(feature_path)

        # Contextual window aggregation for features
        context_feature = []  # Stores aggregated features for each time step
        context_label = []    # Stores aggregated labels for each time step
        for i in range(len(feature_data)):
            # Define the start and end indices for the contextual window
            start = max(0, i - window_size)
            end = min(len(feature_data), i + window_size + 1)
            context_window = feature_data[start:end]

            # Calculate aggregated statistics within the window
            context_feature.append([
                np.mean(context_window),
                np.std(context_window),
                np.min(context_window),
                np.max(context_window),
                np.median(context_window)
            ])
            
            # Generate contextual label based on the majority label within the window
            labels_in_window = df['label_count'][start:end].values
            majority_label = 1 if np.sum(labels_in_window) > (end - start) / 2 else 0
            context_label.append(majority_label)
        
        # Convert contextual features and labels to arrays and save them as .npy files
        context_feature = np.array(context_feature)
        context_label = np.array(context_label)
        
        # Define the paths to save the new contextual feature and label files
        output_feature_path = os.path.join(contextual_feature_dir, os.path.basename(feature_file))
        output_label_path = os.path.join(contextual_label_dir, os.path.basename(feature_file).replace('.npy', '_labels.npy'))
        
        # Save the aggregated feature and label data
        np.save(output_feature_path, context_feature)
        np.save(output_label_path, context_label)
        
        # Append paths of saved files to the lists
        contextual_features.append(output_feature_path)
        contextual_labels.append(output_label_path)

    # Add paths to the generated contextual files in the DataFrame
    df['contextual_feature_file'] = contextual_features
    df['contextual_label_file'] = contextual_labels
    
    # Add 'class' column for binary classification based on 'label_count' (1 for error, 0 for non-error)
    df['class'] = df['label_count'].apply(lambda x: 1 if x > 0 else 0)
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add contextual features and labels to dataset.")
    parser.add_argument('--label_info_path', type=str, required=True, help="Path to the label_info.csv.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the updated label_info_context.csv with contextual features and labels.")
    parser.add_argument('--feature_dir', type=str, required=True, help="Directory containing the original feature files.")
    parser.add_argument('--contextual_feature_dir', type=str, required=True, help="Directory to save new .npy files with contextual features.")
    parser.add_argument('--contextual_label_dir', type=str, required=True, help="Directory to save new .npy files with contextual labels.")
    parser.add_argument('--window_size', type=int, default=5, help="Size of the contextual window.")

    args = parser.parse_args()

    # Load the label information DataFrame
    df = pd.read_csv(args.label_info_path)

    # Add contextual features and labels, generate `class` column, and save new .npy files
    df_with_context = add_contextual_features_and_labels(
        df,
        args.feature_dir,
        args.contextual_feature_dir,
        args.contextual_label_dir,
        args.window_size
    )

    # Save the updated DataFrame with contextual metadata and `class` column
    df_with_context.to_csv(args.output_path, index=False)
    print(f"Contextual features and labels added and saved to {args.output_path}")
