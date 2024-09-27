import os
import sys
import numpy as np
import csv
from datetime import datetime

# Add configuration path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from configuration file
from MasterThesis_Config import SEGMENTED_IMU_DATA_FRAMES_FOLDER, COMBINED_DATA_FOLDER, LOGS_FOLDER, CLASSIFICATION_FOLDER

# Paths to the necessary files
BIG_DATASET_FILE = os.path.join(COMBINED_DATA_FOLDER, 'IMU_Combine_1_Second_Augmented_Dataset.npy')
LOG_FILE_PATH = os.path.join(LOGS_FOLDER, 'IMU_Combine_1_Second_Augmented_Dataset_Log.txt')
BINARY_CLASSIFICATION_FILE = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary_Updated.csv')

# Update the output path for the classification CSV file
CLASSIFICATION_OUTPUT_FILE = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary_BigData_Augmented.csv')

# New output CSV file for ID, group, and range information
SEGMENT_INFO_OUTPUT_FILE = os.path.join(CLASSIFICATION_FOLDER, 'IMU_Segmented_Group_Info_Augmented.csv')

# IDs to be excluded
EXCLUDED_IDS = ["042"]

# Function to log messages
def log_message(message):
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)

# Function to read binary classification
def read_binary_classification(file_path):
    classifications = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3:  # Ensure the row has at least three columns (including augmented)
                child_id = row[0].strip()  # Extract the ID
                quality = row[1].strip()  # Extract the quality (0 or 1)
                augmented = row[2].strip()  # Extract the augmented flag (0 or 1)
                classifications[child_id.zfill(3)] = (quality, augmented)  # Store as tuple
    return classifications

# Function to create the large dataset
def create_big_dataset(combined_folder=COMBINED_DATA_FOLDER, output_file=BIG_DATASET_FILE):
    # Ensure the log directory exists
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)
        log_message(f"Created logs folder: {LOGS_FOLDER}")

    # Initialize the large dataset as a list
    large_dataset = []
    child_data_info = []  # Stores information about children and their data dimensions
    excluded_samples = []  # To store excluded samples
    total_rows_expected = 0  # Expected sum of the first dimensions
    log_message("Starting creation of the large dataset")

    # Iterate through each file in the folder
    for file_name in sorted(os.listdir(combined_folder)):
        if file_name.startswith("IMU_CombineSegmentedData_") and file_name.endswith(".npy"):
            child_id = file_name.split('_')[-1].split('.')[0]  # Extract child ID from the file name
            
            # Skip selected IDs
            if child_id in EXCLUDED_IDS:
                log_message(f"Skipping child ID: {child_id} as it is in the excluded list.")
                excluded_samples.append(child_id)
                continue
            
            file_path = os.path.join(combined_folder, file_name)
            try:
                child_data = np.load(file_path)
                log_message(f"Loaded data from {file_path} with shape {child_data.shape}")

                # Add child data to the large dataset
                if len(large_dataset) == 0:
                    large_dataset = child_data
                else:
                    large_dataset = np.concatenate((large_dataset, child_data), axis=0)
                log_message(f"Updated large dataset shape: {large_dataset.shape}")

                # Add information about the child and data dimensions
                child_data_info.append((child_id, child_data.shape))
                
                # Update the expected number of rows
                total_rows_expected += child_data.shape[0]

            except Exception as e:
                log_message(f"Error loading data from {file_path}: {e}")

    # Save the combined dataset to a new file
    np.save(output_file, large_dataset)
    log_message(f"Large dataset created and saved to {output_file} with shape {large_dataset.shape}\n")

    # Create log summary
    log_message("===== Summary =====")
    log_message(f"Total number of files included: {len(child_data_info)}")
    for child_id, shape in child_data_info:
        log_message(f"Child ID: {child_id}, Data Shape: {shape}")

    # Validation: check if the number of rows matches
    if large_dataset.shape[0] == total_rows_expected:
        log_message("Validation successful: The total number of rows in the big dataset matches the sum of all individual files.")
    else:
        log_message(f"Validation failed: Expected {total_rows_expected} rows, but got {large_dataset.shape[0]} rows in the big dataset.")
    
    # Log excluded samples
    log_message("===== Excluded Samples =====")
    log_message(f"Excluded child IDs: {', '.join(excluded_samples)}")
    
    return child_data_info, excluded_samples  # Return the collected child data information and excluded samples

def create_classification_file(child_data_info, classification_file_path, output_file_path, excluded_samples):
    # Load classification data from binary classification file
    classifications = read_binary_classification(classification_file_path)
    log_message(f"Loaded binary classification data from {classification_file_path}")

    # Initialize classification data to write
    classification_data = []

    # Initialize the index to 2 (to start from the first row after the header)
    current_index = 2

    # Initialize a list to store information for the new CSV file
    segment_info_data = []

    # Initialize counters for FM- and FM+ children
    fm_minus_count = 0
    fm_plus_count = 0

    # Iterate over child data info to generate classification
    for child_id, shape in child_data_info:
        num_rows = shape[0]
        classification_info = classifications.get(child_id)
        if classification_info is None:
            log_message(f"No classification found for child ID: {child_id}, skipping.")
            continue
        
        quality, augmented = classification_info

        # Increment counters based on classification
        if quality == '0':
            fm_minus_count += 1
        elif quality == '1':
            fm_plus_count += 1

        # Add classification for each row corresponding to this child
        start_index = current_index
        for i in range(num_rows):
            classification_data.append([current_index - 1, quality, augmented])
            current_index += 1

        end_index = current_index - 1
        # Log information about the current child data
        log_message(f"Child ID: {child_id} - Assigned classification '{quality}' (Augmented: {augmented}) for rows {start_index} to {end_index}.")
        
        # Add information to the new segment info list
        segment_info_data.append([child_id, quality, augmented, start_index, end_index])

    # Write the classification data to a new CSV file
    with open(output_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Index", "Classification", "Augmented"])
        csv_writer.writerows(classification_data)

    log_message(f"Classification file created and saved to {output_file_path} with {len(classification_data)} rows")
    log_message("Note: Row numbering starts from 1 due to the header row.")

    # Log excluded samples for the classification file
    log_message("===== Excluded Samples for Classification =====")
    log_message(f"Excluded child IDs: {', '.join(excluded_samples)}")

    # Write the segment info data to a new CSV file
    with open(SEGMENT_INFO_OUTPUT_FILE, 'w', newline='') as segment_csvfile:
        segment_csv_writer = csv.writer(segment_csvfile)
        segment_csv_writer.writerow(["Child ID", "Classification", "Augmented", "Start Row", "End Row"])
        segment_csv_writer.writerows(segment_info_data)

    log_message(f"Segment information file created and saved to {SEGMENT_INFO_OUTPUT_FILE} with {len(segment_info_data)} rows")

    # Log the counts of FM- and FM+ children
    log_message("===== Summary of FM- and FM+ Children =====")
    log_message(f"Number of children classified as FM- (0): {fm_minus_count}")
    log_message(f"Number of children classified as FM+ (1): {fm_plus_count}")

# Call the function to create big dataset
child_data_info, excluded_samples = create_big_dataset()

# Call the function to create classification file
create_classification_file(child_data_info, BINARY_CLASSIFICATION_FILE, CLASSIFICATION_OUTPUT_FILE, excluded_samples)
