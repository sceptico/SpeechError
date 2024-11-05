import os
import pandas as pd
import numpy as np
from collections import Counter
import shutil
import argparse

def duplicate_feature_file(original_path, resampled_dir, prefix="resampled_"):
    """
    Duplicates a feature file with a prefix to avoid overwriting and save in the specified directory.

    Args:
        original_path (str): Path to the original feature file.
        resampled_dir (str): Directory to save the duplicated file.
        prefix (str): Prefix for generating unique filenames.

    Returns:
        str: Path to the duplicated file.
    """
    resampled_filename = prefix + os.path.basename(original_path)
    resampled_path = os.path.join(resampled_dir, resampled_filename)
    shutil.copy(original_path, resampled_path)
    return resampled_path

def resample_data_downsample_upweight(df, contextual_feature_dir, label_dir, resampled_feature_dir, resampled_label_dir, downsample_factor=10, target_ratio=None):
    """
    Resamples data by downsampling the majority class and optionally duplicating the minority class to reach a target ratio.
    Also assigns an 'example_weight' column to help the model prioritize minority samples.

    Args:
        df (pd.DataFrame): DataFrame with label and feature metadata.
        contextual_feature_dir (str): Directory for original contextual feature files.
        label_dir (str): Directory for original label files.
        resampled_feature_dir (str): Directory to save resampled feature files.
        resampled_label_dir (str): Directory to save resampled label files.
        downsample_factor (int): Factor to downsample majority class (non-error samples).
        target_ratio (float, optional): Desired error-to-non-error ratio.

    Returns:
        pd.DataFrame: Resampled DataFrame with additional 'example_weight' column.
    """
    # Set binary class based on 'label_count'
    df['class'] = df['label_count'].apply(lambda x: 1 if x > 0 else 0)

    # Separate error and non-error samples
    error_samples = df[df['class'] == 1].copy()
    non_error_samples = df[df['class'] == 0].copy()
    print(f"Initial class distribution: {Counter(df['class'])}")

    # Downsample majority class (non-error samples)
    downsampled_non_errors = non_error_samples.sample(frac=1 / downsample_factor, random_state=0)
    downsample_weight = downsample_factor  # Weight for downsampled examples

    # Assign weights to each class
    error_samples['example_weight'] = 1  # Normal weight for minority class
    downsampled_non_errors['example_weight'] = downsample_weight  # Increased weight for downsampled majority class

    # Calculate target error count if a target ratio is provided
    if target_ratio:
        target_error_count = int(len(downsampled_non_errors) * target_ratio)
        if target_error_count > len(error_samples):
            # Duplicate error samples to reach the target ratio
            additional_errors = error_samples.sample(n=target_error_count - len(error_samples), replace=True, random_state=0)
            error_samples = pd.concat([error_samples, additional_errors])

    # Combine the resampled majority and expanded minority samples
    resampled_df = pd.concat([error_samples, downsampled_non_errors])
    print(f"New class distribution after downsampling and upweighting: {Counter(resampled_df['class'])}")

    # Shuffle the combined DataFrame to ensure a random distribution
    resampled_df = resampled_df.sample(frac=1, random_state=0).reset_index(drop=True)

    # Keep track of duplicated filenames
    resampled_data = []
    resampled_count = 1

    # Generate new files in resampled directories and retain all columns
    for _, row in resampled_df.iterrows():
        # Duplicate contextual features and labels for error samples if needed
        if row['class'] == 1 and target_ratio:
            resampled_feature_file = duplicate_feature_file(row['contextual_feature_file'], resampled_feature_dir)
            resampled_label_file = duplicate_feature_file(row['contextual_label_file'], resampled_label_dir)
        else:
            resampled_feature_file = row['contextual_feature_file']
            resampled_label_file = row['contextual_label_file']
        
        # Append the updated paths and data to the new resampled dataset
        row_data = row.to_dict()
        row_data['contextual_feature_file'] = resampled_feature_file
        row_data['contextual_label_file'] = resampled_label_file
        resampled_data.append(row_data)

    # Convert list of dictionaries to DataFrame
    resampled_df = pd.DataFrame(resampled_data)

    print("Resampling completed. Total entries in resampled dataset:", len(resampled_df))

    # Return DataFrame containing all columns and the 'example_weight' for balancing
    return resampled_df

def save_resampled_info(df_resampled, output_path):
    """
    Saves the resampled dataset with weights and updated paths to a CSV file.

    Args:
        df_resampled (pd.DataFrame): DataFrame with resampled information.
        output_path (str): Path to save the CSV file.
    """
    df_resampled.to_csv(output_path, index=False)
    print(f"Resampled label information saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply downsampling with upweighting to balance data.")
    parser.add_argument('--label_info_path', type=str, required=True, help="Path to the label_info_context.csv.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the resampled label_info.csv.")
    parser.add_argument('--contextual_feature_dir', type=str, required=True, help="Directory containing the contextual feature files.")
    parser.add_argument('--label_dir', type=str, required=True, help="Directory containing the original label files.")
    parser.add_argument('--resampled_feature_dir', type=str, required=True, help="Directory to save duplicated contextual feature files.")
    parser.add_argument('--resampled_label_dir', type=str, required=True, help="Directory to save duplicated label files.")
    parser.add_argument('--downsample_factor', type=int, default=10, help="Factor by which to downsample the majority class.")
    parser.add_argument('--target_ratio', type=float, help="Desired error-to-non-error ratio after resampling.")

    args = parser.parse_args()

    # Load the dataset
    df = pd.read_csv(args.label_info_path)

    # Ensure resampled directories exist
    os.makedirs(args.resampled_feature_dir, exist_ok=True)
    os.makedirs(args.resampled_label_dir, exist_ok=True)

    # Perform resampling
    df_resampled = resample_data_downsample_upweight(
        df,
        args.contextual_feature_dir,
        args.label_dir,
        args.resampled_feature_dir,
        args.resampled_label_dir,
        downsample_factor=args.downsample_factor,
        target_ratio=args.target_ratio
    )

    # Save resampled dataset
    save_resampled_info(df_resampled, args.output_path)



# import os
# import pandas as pd
# import numpy as np
# from collections import Counter
# import shutil
# import argparse
# import random

# def load_label_info(label_info_path):
#     # Load the label information CSV
#     df = pd.read_csv(label_info_path)
#     # Binary classification based on label_count (1 for error, 0 for non-error)
#     df['class'] = df['label_count'].apply(lambda x: 1 if x > 0 else 0)
#     return df

# def duplicate_feature_file(original_path, resampled_dir, resampled_count):
#     # Duplicate the original feature file with a unique name
#     resampled_filename = os.path.basename(original_path).replace('.npy', f'_dup_{resampled_count}.npy')
#     resampled_path = os.path.join(resampled_dir, resampled_filename)
#     shutil.copy(original_path, resampled_path)
#     return resampled_path

# def resample_data_downsample_upweight(df, contextual_feature_dir, label_dir, resampled_feature_dir, resampled_label_dir, downsample_factor=10, target_ratio=None):
#     # Separate the error and non-error samples, creating copies to avoid SettingWithCopyWarning
#     error_samples = df[df['class'] == 1].copy()
#     non_error_samples = df[df['class'] == 0].copy()

#     print(f"Initial class distribution: {Counter(df['class'])}")

#     # Downsample majority class (non-error samples)
#     downsampled_non_errors = non_error_samples.sample(frac=1/downsample_factor, random_state=0)
#     downsample_weight = downsample_factor

#     # Assign an 'example_weight' column for each class
#     error_samples['example_weight'] = 1  # Normal weight for minority class
#     downsampled_non_errors['example_weight'] = downsample_weight  # Upweight downsampled examples

#     # Calculate target error count based on the specified target ratio
#     if target_ratio:
#         target_error_count = int(len(downsampled_non_errors) * target_ratio)
#         if target_error_count > len(error_samples):
#             additional_errors = error_samples.sample(n=target_error_count - len(error_samples), replace=True, random_state=0)
#             error_samples = pd.concat([error_samples, additional_errors])

#     # Combine downsampled majority and expanded minority samples
#     resampled_df = pd.concat([error_samples, downsampled_non_errors])

#     print(f"New class distribution after downsampling and upweighting: {Counter(resampled_df['class'])}")

#     # Shuffle resampled_df to ensure random distribution
#     resampled_df = resampled_df.sample(frac=1, random_state=0).reset_index(drop=True)

#     # Keep track of duplicated file names
#     resampled_data = []
#     resampled_count = 1

#     for _, row in resampled_df.iterrows():
#         feature_file = row['feature_file']
#         label_file = row['label_file']
#         start_time = row['start_time']
#         end_time = row['end_time']
#         label_list = row['label_list']
#         label_count = row['label_count']
#         class_label = row['class']
#         example_weight = row['example_weight']  # Include the example weight

#         # Duplicate error samples if target count is set
#         if row['class'] == 1 and target_ratio:
#             feature_file = duplicate_feature_file(row['feature_file'], resampled_feature_dir, resampled_count)
#             label_file = duplicate_feature_file(row['label_file'], resampled_label_dir, resampled_count)
#             resampled_count += 1

#         # Append all data with example weights for rebalancing
#         resampled_data.append([feature_file, label_file, start_time, end_time, label_list, label_count, class_label, example_weight])

#     print("Resampling completed. Total entries in resampled dataset:", len(resampled_data))

#     # Return resampled DataFrame with all columns, including example weights
#     return pd.DataFrame(resampled_data, columns=['feature_file', 'label_file', 'start_time', 'end_time', 'label_list', 'label_count', 'class', 'example_weight'])


# def save_resampled_info(df_resampled, output_path):
#     # Save the resampled label info with example weights
#     df_resampled.to_csv(output_path, index=False)
#     print(f"Resampled label information saved to {output_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Apply downsampling with upweighting to balance data.")
#     parser.add_argument('--label_info_path', type=str, required=True, help="Path to the label_info_context.csv.")
#     parser.add_argument('--output_path', type=str, required=True, help="Path to save the resampled label_info.csv.")
#     parser.add_argument('--contextual_feature_dir', type=str, required=True, help="Directory containing the contextual feature files.")
#     parser.add_argument('--label_dir', type=str, required=True, help="Directory containing the original label files.")
#     parser.add_argument('--resampled_feature_dir', type=str, required=True, help="Directory to save duplicated contextual feature files.")
#     parser.add_argument('--resampled_label_dir', type=str, required=True, help="Directory to save duplicated label files.")
#     parser.add_argument('--downsample_factor', type=int, default=10, help="Factor by which to downsample the majority class.")
#     parser.add_argument('--target_ratio', type=float, help="Desired error-to-non-error ratio after resampling.")

#     args = parser.parse_args()

#     # Load data
#     df = load_label_info(args.label_info_path)

#     # Ensure resampled directories exist
#     os.makedirs(args.resampled_feature_dir, exist_ok=True)
#     os.makedirs(args.resampled_label_dir, exist_ok=True)

#     # Resample data using downsampling with upweighting and target ratio
#     df_resampled = resample_data_downsample_upweight(
#         df,
#         args.contextual_feature_dir,
#         args.label_dir,
#         args.resampled_feature_dir,
#         args.resampled_label_dir,
#         downsample_factor=args.downsample_factor,
#         target_ratio=args.target_ratio
#     )

#     # Save the resampled label info with example weights
#     save_resampled_info(df_resampled, args.output_path)



