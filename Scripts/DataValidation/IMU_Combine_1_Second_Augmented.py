import os
import sys
import numpy as np
import csv
from datetime import datetime

# Add configuration path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from configuration file
from MasterThesis_Config import SEGMENTED_IMU_DATA_FRAMES_FOLDER, COMBINED_DATA_FOLDER, LOGS_FOLDER, CLASSIFICATION_FOLDER

"""
IMU Data Processing and Classification Script

This script processes segmented IMU (Inertial Measurement Unit) data files and combines them into 
a single large dataset. It also generates classification files with labels (FM+ or FM-) 
for each sample in the dataset. The script logs all operations, including excluded data 
and validation results.

Main Features:
1. **Dataset Creation**: Combines individual `.npy` files (IMU data) into one large dataset. 
   Information about each child and the shape of their data is logged.
2. **Classification File Generation**: Assigns binary classification labels to each sample 
   based on an external classification CSV file. Generates a classification file and logs the 
   classification details.
3. **Logging**: All steps, including errors, excluded samples, and summary information, 
   are logged to a log file.

Requirements:
- The paths to folders and files are defined in the `MasterThesis_Config` file, which must be correctly set up.
- This script relies on `.npy` files stored in the `COMBINED_DATA_FOLDER`.

"""

# Paths to the necessary files
BIG_DATASET_FILE = os.path.join(COMBINED_DATA_FOLDER, 'IMU_Combine_1_Second_Augmented_Dataset.npy')  
LOG_FILE_PATH = os.path.join(LOGS_FOLDER, 'IMU_Combine_1_Second_Augmented_Dataset_Log.txt')  
BINARY_CLASSIFICATION_FILE = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary_Updated.csv')  

# Output paths for new classification files
CLASSIFICATION_OUTPUT_FILE = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary_BigData_Augmented.csv')  # Path to save the new classification CSV file
SEGMENT_INFO_OUTPUT_FILE = os.path.join(CLASSIFICATION_FOLDER, 'IMU_Segmented_Group_Info_Augmented.csv')  # Path to save segment info (ID, group, range)

# IDs to be excluded from processing (for example, data that should not be included in the final dataset)
EXCLUDED_IDS = ["042"]

# Function to log messages
def log_message(message):
    """
    Logs a message with a timestamp to both the console and a log file.
    
    Parameters:
    - message (str): The message to log.
    """
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)

# Function to read binary classification data from a CSV file
def read_binary_classification(file_path):
    """
    Reads the binary classification CSV file and returns a dictionary mapping child IDs to 
    classification labels and augmentation flags.
    
    Parameters:
    - file_path (str): The path to the binary classification file.
    
    Returns:
    - classifications (dict): A dictionary mapping child IDs (as strings) to tuples (quality, augmented).
    """
    classifications = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3:  # Ensure the row has at least three columns (ID, quality, augmented)
                child_id = row[0].strip()  # Extract child ID
                quality = row[1].strip()  # Extract quality (0 or 1)
                augmented = row[2].strip()  # Extract augmented flag (0 or 1)
                classifications[child_id.zfill(3)] = (quality, augmented) 
    return classifications

# Function to create the large dataset by combining multiple `.npy` files
def create_big_dataset(combined_folder=COMBINED_DATA_FOLDER, output_file=BIG_DATASET_FILE):
    """
    Combines individual IMU segmented `.npy` files into a single large dataset.
    The function logs the process, handles exclusions, and validates the number of rows.

    Parameters:
    - combined_folder (str): The folder containing segmented IMU data files.
    - output_file (str): The path where the combined dataset will be saved.
    
    Returns:
    - child_data_info (list): A list of tuples containing child IDs and their data dimensions.
    - excluded_samples (list): A list of child IDs that were excluded from the dataset.
    """
    # Ensure the log directory exists
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)
        log_message(f"Created logs folder: {LOGS_FOLDER}")

    large_dataset = []  # List to hold the combined data
    child_data_info = []  # List to store information about each child's data (ID and shape)
    excluded_samples = []  # List to store excluded child IDs
    total_rows_expected = 0  # Counter for the expected total number of rows in the dataset

    log_message("Starting creation of the large dataset")

    # Iterate through each file in the folder and process the data
    for file_name in sorted(os.listdir(combined_folder)):
        if file_name.startswith("IMU_CombineSegmentedData_") and file_name.endswith(".npy"):
            child_id = file_name.split('_')[-1].split('.')[0] 
            
            # Skip IDs that are marked as excluded
            if child_id in EXCLUDED_IDS:
                log_message(f"Skipping child ID: {child_id} as it is in the excluded list.")
                excluded_samples.append(child_id)
                continue
            
            file_path = os.path.join(combined_folder, file_name)
            try:
                # Load the .npy file
                child_data = np.load(file_path)
                log_message(f"Loaded data from {file_path} with shape {child_data.shape}")

                # Combine the data
                if len(large_dataset) == 0:
                    large_dataset = child_data
                else:
                    large_dataset = np.concatenate((large_dataset, child_data), axis=0) 
                log_message(f"Updated large dataset shape: {large_dataset.shape}")

                # Record information about the child's data (ID and shape)
                child_data_info.append((child_id, child_data.shape))
                
                # Update the total expected number of rows
                total_rows_expected += child_data.shape[0]

            except Exception as e:
                log_message(f"Error loading data from {file_path}: {e}")

    # Save the combined dataset to a `.npy` file
    np.save(output_file, large_dataset)
    log_message(f"Large dataset created and saved to {output_file} with shape {large_dataset.shape}\n")

    # Create a summary log
    log_message("===== Summary =====")
    log_message(f"Total number of files included: {len(child_data_info)}")
    for child_id, shape in child_data_info:
        log_message(f"Child ID: {child_id}, Data Shape: {shape}")

    # Validate if the total number of rows matches the expected count
    if large_dataset.shape[0] == total_rows_expected:
        log_message("Validation successful: The total number of rows in the big dataset matches the sum of all individual files.")
    else:
        log_message(f"Validation failed: Expected {total_rows_expected} rows, but got {large_dataset.shape[0]} rows in the big dataset.")
    
    # Log the excluded samples
    log_message("===== Excluded Samples =====")
    log_message(f"Excluded child IDs: {', '.join(excluded_samples)}")
    
    return child_data_info, excluded_samples 

# Function to create the classification file based on the combined dataset
def create_classification_file(child_data_info, classification_file_path, output_file_path, excluded_samples):
    """
    Creates a classification CSV file for the large dataset based on child data information and binary classification labels.

    Parameters:
    - child_data_info (list): A list of tuples containing child IDs and their data dimensions.
    - classification_file_path (str): The path to the binary classification file.
    - output_file_path (str): The path where the new classification file will be saved.
    - excluded_samples (list): A list of child IDs that were excluded from the dataset.
    """
    # Load classification data
    classifications = read_binary_classification(classification_file_path)
    log_message(f"Loaded binary classification data from {classification_file_path}")

    classification_data = []  # List to store classification data
    current_index = 2  

    segment_info_data = [] 

    fm_minus_count = 0 
    fm_plus_count = 0 

    # Iterate over the child data info and generate classification entries
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

        start_index = current_index
        # Add classification for each row of the child data
        for i in range(num_rows):
            classification_data.append([current_index - 1, quality, augmented])
            current_index += 1

        end_index = current_index - 1
        # Log classification information for the child
        log_message(f"Child ID: {child_id} - Assigned classification '{quality}' (Augmented: {augmented}) for rows {start_index} to {end_index}.")
        
        # Store segment info
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

    # Log the counts of FM- and FM+ classified children
    log_message("===== Summary of FM- and FM+ Children =====")
    log_message(f"Number of children classified as FM- (0): {fm_minus_count}")
    log_message(f"Number of children classified as FM+ (1): {fm_plus_count}")

# Call the function to create the big dataset
child_data_info, excluded_samples = create_big_dataset()

# Call the function to create the classification file
create_classification_file(child_data_info, BINARY_CLASSIFICATION_FILE, CLASSIFICATION_OUTPUT_FILE, excluded_samples)
