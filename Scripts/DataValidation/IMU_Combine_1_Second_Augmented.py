import os
import sys
import numpy as np
import csv
from datetime import datetime
from Augmentation_Techniques import jitter

# Add configuration path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from configuration file
from MasterThesis_Config import (SEGMENTED_IMU_DATA_FRAMES_FOLDER, COMBINED_DATA_FOLDER, LOGS_FOLDER, 
                                 CLASSIFICATION_FOLDER)

# Define the augmented data folder path
AUGMENTED_DATA_FOLDER = r'C:\GitRepositories\Master_Thesis\Data_IMU_Augmented'

# Ensure the augmented data folder exists
if not os.path.exists(AUGMENTED_DATA_FOLDER):
    os.makedirs(AUGMENTED_DATA_FOLDER)

# Paths to the necessary files
BINARY_CLASSIFICATION_FILE = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary.csv')
AUGMENTATION_LOG_FILE = os.path.join(AUGMENTED_DATA_FOLDER, 'IMU_Augmentation_Log.txt')
NEW_CLASSIFICATION_FILE = os.path.join(AUGMENTED_DATA_FOLDER, 'FM_QualityClassification_Binary_Updated.csv')

# IDs to be excluded
EXCLUDED_IDS = ["042"]

# Function to log messages
def log_message(message, log_file_path=AUGMENTATION_LOG_FILE):
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)

# Function to read binary classification
def read_binary_classification(file_path):
    classifications = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:  # Ensure the row has at least two columns
                child_id = row[0].strip()  # Extract the ID
                quality = row[1].strip()  # Extract the quality (0 or 1)
                classifications.append([child_id.zfill(3), quality, '0'])  # Add '0' for original samples
    return classifications

def adjust_dimensions(sample, target_shape):
    """Adjust the dimensions of the sample to match the target shape."""
    current_shape = sample.shape
    if current_shape == target_shape:
        return sample  # No adjustment needed

    # Create a new array with the target shape filled with zeros
    adjusted_sample = np.zeros(target_shape, dtype=np.float32)

    # Determine the minimum shape along each dimension
    min_shape = tuple(min(cs, ts) for cs, ts in zip(current_shape, target_shape))

    # Adjust the dimensions by copying the data
    slices = tuple(slice(0, ms) for ms in min_shape)
    adjusted_sample[slices] = sample[slices]

    return adjusted_sample

def augment_negative_samples(data, num_augmentations_multiplier, target_shape):
    """Augments negative samples by generating additional samples using only jittering with a constant sigma for each file."""
    augmented_data = []
    augmentation_info = []

    # Choose one random sigma value for all samples in this file
    sigma_jitter = np.random.uniform(0.01, 0.05)
    
    for original_index, sample in enumerate(data):
        for _ in range(num_augmentations_multiplier):
            # Apply jittering with the chosen constant sigma for all samples in this file
            jittered_sample = jitter(sample, sigma=sigma_jitter)
            jittered_sample = adjust_dimensions(jittered_sample, target_shape)
            augmented_data.append(jittered_sample.astype(np.float32))
            augmentation_info.append((original_index, sample.shape, jittered_sample.shape, 'Jittering', sigma_jitter))

    # Convert to numpy array
    augmented_data_array = np.array(augmented_data, dtype=np.float32)

    # Log the augmentation info with detailed information
    with open(AUGMENTATION_LOG_FILE, 'a') as log_file:
        log_file.write("\n\n===== Augmentation Process Started =====\n")
        for original_idx, original_shape, augmented_shape, technique, param in augmentation_info:
            log_file.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Original Sample Index: {original_idx}, Original Shape: {original_shape}, "
                f"Augmented Shape: {augmented_shape}, Technique: {technique}, Parameter: {param}\n"
            )
        log_file.write(f"Total augmented samples created: {len(augmented_data)}\n")
        log_file.write("===== Augmentation Process Completed =====\n\n")
    
    return augmented_data_array


def create_files_for_augmentation(combined_folder=COMBINED_DATA_FOLDER, num_augmentations_multiplier=5):
    # Ensure the log directory exists
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)
        log_message(f"Created logs folder: {LOGS_FOLDER}")

    classifications = read_binary_classification(BINARY_CLASSIFICATION_FILE)
    negative_files = []  # To store paths of negative samples
    last_index = 0  # Track the last child index in the original dataset

    log_message("Starting to read files and select negative samples.")

    # First, find the last index among all files
    for file_name in sorted(os.listdir(combined_folder)):
        if file_name.startswith("IMU_CombineSegmentedData_") and file_name.endswith(".npy"):
            child_id = file_name.split('_')[-1].split('.')[0]  # Extract child ID from the file name
            if child_id.isdigit():
                last_index = max(last_index, int(child_id))

    log_message(f"Last index found among all files: {last_index}")

    # Now, identify negative files for augmentation
    for file_name in sorted(os.listdir(combined_folder)):
        if file_name.startswith("IMU_CombineSegmentedData_") and file_name.endswith(".npy"):
            child_id = file_name.split('_')[-1].split('.')[0]  # Extract child ID from the file name

            # Skip selected IDs
            if child_id in EXCLUDED_IDS:
                log_message(f"Skipping child ID: {child_id} as it is in the excluded list.")
                continue

            file_path = os.path.join(combined_folder, file_name)

            # Check if the child ID is marked as '0' in classifications
            for entry in classifications:
                if entry[0] == child_id and entry[1] == '0':
                    negative_files.append(file_path)
                    log_message(f"File selected for augmentation: {file_path}")
                    break

    log_message("\n\n===== Files Selected for Augmentation =====")

    # Start augmenting and creating new files
    current_index = last_index + 1
    new_classifications = list(classifications)  # Copy existing classifications

    for file_path in negative_files:
        try:
            child_data = np.load(file_path)
            target_shape = child_data.shape[1:]  # All samples should match this shape
            log_message(f"Augmenting data from {file_path} with shape {child_data.shape}")

            # Augment FM- samples using only jittering
            for i in range(num_augmentations_multiplier):
                augmented_samples = augment_negative_samples(child_data, 1, target_shape)
                new_file_name = f"IMU_CombineSegmentedData_{str(current_index).zfill(3)}.npy"
                new_file_path = os.path.join(AUGMENTED_DATA_FOLDER, new_file_name)
                np.save(new_file_path, augmented_samples.astype(np.float32))
                log_message(f"Saved augmented file: {new_file_path} with {augmented_samples.shape[0]} samples")

                # Update classification data for the new files
                new_classifications.append([str(current_index).zfill(3), '0', '1'])  # FM- classification and marked as augmented
                current_index += 1

        except Exception as e:
            log_message(f"Error augmenting data from {file_path}: {e}")

    # Save the updated classification file
    with open(NEW_CLASSIFICATION_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Index", "Classification", "Augmented"])  # Write header
        csv_writer.writerows(new_classifications)

    log_message(f"Classification file created and saved to {NEW_CLASSIFICATION_FILE} with {len(new_classifications)} rows")
    log_message(f"\n\nAll files augmented and saved successfully in {AUGMENTED_DATA_FOLDER}. New files start from index {last_index + 1}.")

# Execute the function
create_files_for_augmentation()
