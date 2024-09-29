import os
import sys
import numpy as np
import csv
from datetime import datetime

# Define the folder path for saving files
BASE_FOLDER_PATH = r'C:\GitRepositories\Master_Thesis\Data_IMU_Segmented_Combined'
SEGMENTED_60S_FOLDER = os.path.join(BASE_FOLDER_PATH, 'IMU_Segmented_60_Seconds_Augmented')

# Create the folder if it does not exist
if not os.path.exists(SEGMENTED_60S_FOLDER):
    os.makedirs(SEGMENTED_60S_FOLDER)

# Paths to the necessary files
BIG_DATASET_FILE = os.path.join(SEGMENTED_60S_FOLDER, 'IMU_Segmented_60s_BigDataset_Augmented.npy')
LOG_FILE_PATH = os.path.join(SEGMENTED_60S_FOLDER, 'IMU_Segmented_60s_BigDataset_Log_Augmented.txt')

# Update the output path for the classification CSV file
CLASSIFICATION_OUTPUT_FILE = os.path.join(SEGMENTED_60S_FOLDER, 'FM_QualityClassification_Binary_60s_BigData_Augmented.csv')

# New output CSV file for ID, group, range, and augmentation information
SEGMENT_INFO_OUTPUT_FILE = os.path.join(SEGMENTED_60S_FOLDER, 'IMU_Segmented_60s_Group_Info_Augmented.csv')

# IDs to be excluded
EXCLUDED_IDS = ["042", "100", "096", "091", "086", "082", "075", "050", "035", "009", "016", "020", "034", "103", "088", "067", "055", "057", "049", "046", "005", "008", "010", "025", "001", "002"]

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
            if len(row) >= 2:  # Ensure the row has at least two columns
                child_id = row[0].strip()  # Extract the ID
                quality = row[1].strip()  # Extract the quality (0 or 1)
                classifications[child_id.zfill(3)] = quality  # Ensure child_id has leading zeros
    return classifications

# Function to create the large dataset with 60-second segments
def create_big_dataset_60s(combined_folder=BASE_FOLDER_PATH, output_file=BIG_DATASET_FILE):
    # Initialize the large dataset as a list
    large_dataset = []
    child_data_info = []  # Stores information about children and their data dimensions
    excluded_samples = []  # To store excluded samples
    total_segments = 0  # Total number of valid segments
    log_message("Starting creation of the 60-second segmented dataset")

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

                # Check if the first dimension is divisible by 60
                num_full_segments = child_data.shape[0] // 60
                if num_full_segments == 0:
                    log_message(f"Child ID: {child_id} does not have enough data for a 60-second segment. Skipping.")
                    excluded_samples.append(child_id)
                    continue

                # Reshape the data into 60-second segments and discard remainder
                valid_data = child_data[:num_full_segments * 60, :, :].reshape(num_full_segments, 6000, 24)
                log_message(f"Child ID: {child_id} reshaped to {valid_data.shape[0]} segments of 60 seconds")

                # Add reshaped data to the large dataset
                if len(large_dataset) == 0:
                    large_dataset = valid_data
                else:
                    large_dataset = np.concatenate((large_dataset, valid_data), axis=0)
                log_message(f"Updated large dataset shape: {large_dataset.shape}")

                # Add information about the child and new segments (augmented flag is 0 for original)
                for segment_index in range(num_full_segments):
                    segment_id = f"{total_segments + 1}"  # Sequential ID starting from 1
                    child_data_info.append((segment_id, child_id, valid_data.shape, 0))  # Augmented flag: 0
                    total_segments += 1

            except Exception as e:
                log_message(f"Error loading data from {file_path}: {e}")

    # Save the combined dataset to a new file
    np.save(output_file, large_dataset)
    log_message(f"60-second segmented dataset created and saved to {output_file} with shape {large_dataset.shape}\n")

    # Create log summary
    log_message("===== Summary =====")
    log_message(f"Total number of files included: {len(child_data_info)}")
    for segment_id, child_id, shape, augmented in child_data_info:
        log_message(f"Child ID: {child_id}, Segment ID: {segment_id}, Data Shape: {shape}, Augmented: {augmented}")

    # Log excluded samples
    log_message("===== Excluded Samples =====")
    log_message(f"Excluded child IDs: {', '.join(excluded_samples)}")
    
    return child_data_info, excluded_samples  # Return the collected child data information and excluded samples

def create_classification_file_60s(child_data_info, classification_file_path=BINARY_CLASSIFICATION_FILE_CSV, output_file_path=CLASSIFICATION_OUTPUT_FILE, excluded_samples=[]):
    # Load classification data from binary classification file
    classifications = read_binary_classification(classification_file_path)
    log_message(f"Loaded binary classification data from {classification_file_path}")

    # Initialize classification data to write
    classification_data = []

    # Initialize a list to store information for the new CSV file
    segment_info_data = []

    # Initialize counters for FM- and FM+ children
    fm_minus_count = 0
    fm_plus_count = 0

    # Column names to match the new format
    csv_columns = ["Segment ID", "Child ID", "Classification", "Augmented"]

    # Iterate over child data info to generate classification
    for segment_id, child_id, shape, augmented in child_data_info:
        # Skip excluded samples
        if child_id in EXCLUDED_IDS:
            log_message(f"Skipping child ID: {child_id} for classification as it is in the excluded list.")
            continue

        classification = classifications.get(child_id)
        if classification is None:
            log_message(f"No classification found for child ID: {child_id}, skipping.")
            continue

        # Increment counters based on classification
        if classification == '0':
            fm_minus_count += 1
        elif classification == '1':
            fm_plus_count += 1

        # Add classification for each segment corresponding to this child
        classification_data.append([segment_id, child_id, classification, augmented])  # Augmented: 0 or 1

        # Log information about the current child data
        log_message(f"Segment ID: {segment_id} - Assigned classification '{classification}', Augmented: {augmented}.")

        # Add information to the new segment info list
        segment_info_data.append([segment_id, child_id, classification, augmented])

    # Write the classification data to a new CSV file
    with open(output_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_columns)  # Write header
        csv_writer.writerows(classification_data)

    log_message(f"Classification file created and saved to {output_file_path} with {len(classification_data)} rows")

    # Log excluded samples for the classification file
    log_message("===== Excluded Samples for Classification =====")
    log_message(f"Excluded child IDs: {', '.join(excluded_samples)}")

    # Write the segment info data to a new CSV file
    with open(SEGMENT_INFO_OUTPUT_FILE, 'w', newline='') as segment_csvfile:
        segment_csv_writer = csv.writer(segment_csvfile)
        segment_csv_writer.writerow(csv_columns)  # Write header
        segment_csv_writer.writerows(segment_info_data)

    log_message(f"Segment information file created and saved to {SEGMENT_INFO_OUTPUT_FILE} with {len(segment_info_data)} rows")

    # Log the counts of FM- and FM+ children
    log_message("===== Summary of FM- and FM+ Children =====")
    log_message(f"Number of segments classified as FM- (0): {fm_minus_count}")
    log_message(f"Number of segments classified as FM+ (1): {fm_plus_count}")

# Call the function to create the 60-second segmented dataset
child_data_info, excluded_samples = create_big_dataset_60s()

# Call the function to create classification file for 60-second segments
create_classification_file_60s(child_data_info)
