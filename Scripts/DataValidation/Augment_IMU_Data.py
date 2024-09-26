"""
IMU Data Augmentation Pipeline for FM Classification

This script performs augmentation on IMU (Inertial Measurement Unit) data to create synthetic samples of negative (FM-) class instances. 
It primarily applies jittering, a technique where Gaussian noise is added to the data, to generate multiple variations of the original samples. 
The process includes validating the augmented data by calculating the difference and correlation between original and augmented samples.

Key Features:
- Reads binary classification data to identify negative samples for augmentation.
- Generates a specified number of augmented samples using jittering.
- Logs all steps, including augmentation details, and saves augmented samples to a new folder.
- Updates the classification file with the new augmented samples.

Dependencies:
- Augmentation techniques (imported from an external module).
- Configuration for file paths (from an external configuration file).
- Jittering is applied by default, while window warping has been commented out for potential future use.
- Validates the augmentation process by comparing original and augmented data in terms of statistical measures.

Techniques:
- **Jittering**: Gaussian noise is added to each sample to simulate variability in the data.
Future iterations may involve exploring or combining other augmentation techniques like window warping.
"""

import os
import sys
import numpy as np
import csv
from datetime import datetime

# Import augmentation techniques
from Augmentation_Techniques import jitter  # Removed window_warp import

# Add configuration path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from configuration file
from MasterThesis_Config import (
    SEGMENTED_IMU_DATA_FRAMES_FOLDER,
    COMBINED_DATA_FOLDER,
    LOGS_FOLDER,
    CLASSIFICATION_FOLDER
)

# Define the augmented data folder path
AUGMENTED_DATA_FOLDER = r'C:\GitRepositories\Master_Thesis\Data_IMU_Augmented'

# Ensure the augmented data folder exists
if not os.path.exists(AUGMENTED_DATA_FOLDER):
    os.makedirs(AUGMENTED_DATA_FOLDER)

# Paths to the necessary files
BINARY_CLASSIFICATION_FILE = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary.csv')
AUGMENTATION_LOG_FILE = os.path.join(AUGMENTED_DATA_FOLDER, 'IMU_Augmentation_Log.txt')
NEW_CLASSIFICATION_FILE = os.path.join(AUGMENTED_DATA_FOLDER, 'FM_QualityClassification_Binary_Updated.csv')

# IDs to be excluded from the augmentation process
EXCLUDED_IDS = ["042"]

# Function to log messages with timestamps
def log_message(message, log_file_path=AUGMENTATION_LOG_FILE):
    """Logs a message to both console and a log file."""
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)

# Function to read binary classification data from a CSV file
def read_binary_classification(file_path):
    """Reads binary classification data and returns a list of child IDs, their quality label, and an augmented flag."""
    classifications = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            if len(row) >= 2:  # Ensure valid row structure
                child_id = row[0].strip()
                quality = row[1].strip()
                classifications.append([child_id.zfill(3), quality, '0'])  # Mark original samples with '0'
    return classifications

# Function to adjust dimensions of a sample to a target shape
def adjust_dimensions(sample, target_shape):
    """Ensures that the sample has the specified target shape, filling with zeros if necessary."""
    current_shape = sample.shape
    if current_shape == target_shape:
        return sample  # Return the sample if no adjustment is required

    # Create a new array with the target shape filled with zeros
    adjusted_sample = np.zeros(target_shape, dtype=np.float32)

    # Determine the minimum shape along each dimension
    min_shape = tuple(min(cs, ts) for cs, ts in zip(current_shape, target_shape))

    # Adjust the dimensions by copying the data
    slices = tuple(slice(0, ms) for ms in min_shape)
    adjusted_sample[slices] = sample[slices]

    return adjusted_sample

# Function to validate the augmentation process
def validate_augmentation(original_sample, augmented_sample, original_index):
    """Compares the original sample with its augmented version and logs the differences."""
    differences = np.abs(original_sample - augmented_sample)
    mean_difference = np.mean(differences)
    std_difference = np.std(differences)

    # Calculate the correlation coefficient for each feature
    correlations = []
    for i in range(original_sample.shape[1]):
        corr = np.corrcoef(original_sample[:, i], augmented_sample[:, i])[0, 1]
        correlations.append(corr)
    mean_correlation = np.mean(correlations)
    std_correlation = np.std(correlations)

    # Log validation results
    log_message(f"Validation of augmentation for sample index {original_index}:")
    log_message(f"Mean absolute difference: {mean_difference}")
    log_message(f"Standard deviation of differences: {std_difference}")
    log_message(f"Mean correlation coefficient: {mean_correlation}")
    log_message(f"Standard deviation of correlations: {std_correlation}")

# Function to create augmented data files
def create_files_for_augmentation(combined_folder=COMBINED_DATA_FOLDER, num_augmentations_multiplier=8):
    """Creates augmented data by applying jittering on negative samples and saves new augmented files."""
    
    # Ensure the log folder exists
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)
        log_message(f"Created logs folder: {LOGS_FOLDER}")

    classifications = read_binary_classification(BINARY_CLASSIFICATION_FILE)
    negative_files = []  # List to store file paths of negative samples
    last_index = 0  # Track the last child index in the original dataset

    log_message("Starting to read files and select negative samples.")

    # Determine the last index among all existing files
    for file_name in sorted(os.listdir(combined_folder)):
        if file_name.startswith("IMU_CombineSegmentedData_") and file_name.endswith(".npy"):
            child_id = file_name.split('_')[-1].split('.')[0]
            if child_id.isdigit():
                last_index = max(last_index, int(child_id))

    log_message(f"Last index found among all files: {last_index}")

    # Identify negative files for augmentation
    for file_name in sorted(os.listdir(combined_folder)):
        if file_name.startswith("IMU_CombineSegmentedData_") and file_name.endswith(".npy"):
            child_id = file_name.split('_')[-1].split('.')[0]

            # Skip excluded IDs
            if child_id in EXCLUDED_IDS:
                log_message(f"Skipping child ID: {child_id} as it is in the excluded list.")
                continue

            file_path = os.path.join(combined_folder, file_name)

            # Check if the child ID is marked as '0' (negative sample)
            for entry in classifications:
                if entry[0] == child_id and entry[1] == '0':
                    negative_files.append(file_path)
                    log_message(f"File selected for augmentation: {file_path}")
                    break

    log_message("\n\n===== Files Selected for Augmentation =====")

    # Begin augmenting and saving new files
    current_index = last_index + 1
    new_classifications = list(classifications)  # Copy the original classifications

    for file_path in negative_files:
        try:
            child_data = np.load(file_path)
            target_shape = child_data.shape[1:]  # All samples should match this shape
            log_message(f"Augmenting data from {file_path} with shape {child_data.shape}")

            # Augment the data using jittering
            for i in range(num_augmentations_multiplier):
                sigma_jitter = np.random.uniform(0.025, 0.035)
                augmented_samples = []
                augmentation_info = []

                for original_index, sample in enumerate(child_data):
                    augmented_sample = sample.copy()

                    # Apply jittering augmentation
                    augmented_sample = jitter(augmented_sample, sigma=sigma_jitter)

                    # Adjust dimensions if necessary
                    augmented_sample = adjust_dimensions(augmented_sample, target_shape)

                    augmented_samples.append(augmented_sample.astype(np.float32))
                    augmentation_info.append({
                        'original_index': original_index,
                        'augmented_index': len(augmented_samples) - 1,
                        'original_shape': sample.shape,
                        'augmented_shape': augmented_sample.shape,
                        'techniques': 'Jittering',
                        'sigma_jitter': sigma_jitter,
                    })

                # Convert to numpy array
                augmented_samples_array = np.array(augmented_samples, dtype=np.float32)

                # Save augmented samples to file
                new_file_name = f"IMU_CombineSegmentedData_{str(current_index).zfill(3)}.npy"
                new_file_path = os.path.join(AUGMENTED_DATA_FOLDER, new_file_name)
                np.save(new_file_path, augmented_samples_array)
                log_message(f"Saved augmented file: {new_file_path} with {augmented_samples_array.shape[0]} samples")

                # Log augmentation info
                with open(AUGMENTATION_LOG_FILE, 'a') as log_file:
                    log_file.write("\n\n===== Augmentation Process Started =====\n")
                    for info in augmentation_info:
                        log_file.write(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Original Sample Index: {info['original_index']}, "
                            f"Augmented Sample Index: {info['augmented_index']}, "
                            f"Original Shape: {info['original_shape']}, Augmented Shape: {info['augmented_shape']}, "
                            f"Techniques: {info['techniques']}, Sigma Jitter: {info['sigma_jitter']}\n"
                        )
                    log_file.write(f"Total augmented samples created: {len(augmented_samples)}\n")
                    log_file.write("===== Augmentation Process Completed =====\n\n")

                # Validate the augmentation process
                for info in augmentation_info:
                    original_index = info['original_index']
                    augmented_index = info['augmented_index']
                    original_sample = child_data[original_index]
                    augmented_sample = augmented_samples_array[augmented_index]

                    validate_augmentation(original_sample, augmented_sample, original_index)

                # Update classification file with new data
                new_classifications.append([str(current_index).zfill(3), '0', '1'])  # FM- classification marked as augmented
                current_index += 1

        except Exception as e:
            log_message(f"Error augmenting data from {file_path}: {e}")

    # Save the updated classification file
    with open(NEW_CLASSIFICATION_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Index", "Classification", "Augmented"])  # Write the header
        csv_writer.writerows(new_classifications)

    log_message(f"Classification file created and saved to {NEW_CLASSIFICATION_FILE} with {len(new_classifications)} rows")
    log_message(f"\n\nAll files augmented and saved successfully in {AUGMENTED_DATA_FOLDER}. New files start from index {last_index + 1}.")

# Execute the augmentation function when the script is run
if __name__ == "__main__":
    create_files_for_augmentation()
