import os
import sys
import numpy as np
from datetime import datetime

# Add the configuration path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FM_QualityClassification import create_classification_file
from IMU_CombineSegmentedData import combine_segmented_imu_data  # Import the script for combining IMU data
from MasterThesis_Config import SEGMENTED_IMU_DATA_FRAMES_FOLDER, BINARY_CLASSIFICATION_FILE_TXT, THREE_CLASS_CLASSIFICATION_FILE_TXT, VALIDATION_OUTPUT_FILE, COMBINED_DATA_FOLDER, IMU_SUFFIXES, EXCEL_FILE_PATH, LOGS_FOLDER

# Ensure the logs folder exists
if not os.path.exists(LOGS_FOLDER):
    os.makedirs(LOGS_FOLDER)
    print(f"Created logs folder: {LOGS_FOLDER}")

# Function to save results to a file with date and time at the beginning
def save_results_to_file(content, file_path):
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"Log created on: {current_time}\n{'='*50}\n"
    content = header + content  # Prepend the date/time header to the content
    with open(file_path, 'w') as file:
        file.write(content)

# Check if classification files exist
def check_classification_files():
    if not os.path.exists(BINARY_CLASSIFICATION_FILE_TXT):
        print(f"Binary classification file '{BINARY_CLASSIFICATION_FILE_TXT}' not found. Creating it.")
        create_classification_file(EXCEL_FILE_PATH, binary=True)
    else:
        print(f"Binary classification file '{BINARY_CLASSIFICATION_FILE_TXT}' found.")

    if not os.path.exists(THREE_CLASS_CLASSIFICATION_FILE_TXT):
        print(f"Three-class classification file '{THREE_CLASS_CLASSIFICATION_FILE_TXT}' not found. Creating it.")
        create_classification_file(EXCEL_FILE_PATH, binary=False)
    else:
        print(f"Three-class classification file '{THREE_CLASS_CLASSIFICATION_FILE_TXT}' found.")

# Function to load classification data
def load_classifications(file_path):
    classifications = {}
    with open(file_path, 'r') as file:
        for line in file:
            id_part, quality_part = line.strip().split(',')
            classifications[int(id_part.split(':')[1].strip())] = quality_part.split(':')[1].strip()
    return classifications

# Function to check data consistency
def check_data_consistency(samples, frames, features):
    return all(count == samples[0] for count in samples) and all(count == 100 for count in frames) and all(count == 6 for count in features)

# Function to process each file
def process_file(file_path, unique_id, min_samples, min_samples_newborn_id, samples, frames, features):
    data = np.load(file_path)
    samples.append(data.shape[0])
    frames.append(data.shape[1])
    features.append(data.shape[2])

    if data.shape[0] < min_samples:
        min_samples, min_samples_newborn_id = data.shape[0], unique_id

    if np.isnan(data).any():
        return f"Warning: NaN values detected in file {file_path}.", min_samples, min_samples_newborn_id
    elif np.isinf(data).any():
        return f"Warning: Infinite values detected in file {file_path}.", min_samples, min_samples_newborn_id
    else:
        return f"No NaN or infinite values - data is OK.", min_samples, min_samples_newborn_id

# Check classification files
check_classification_files()

# Load classifications
binary_classifications = load_classifications(BINARY_CLASSIFICATION_FILE_TXT)
three_class_classifications = load_classifications(THREE_CLASS_CLASSIFICATION_FILE_TXT)

# Extract unique newborn IDs
unique_ids = sorted(set(f.split(' - ')[1] for f in os.listdir(SEGMENTED_IMU_DATA_FRAMES_FOLDER) if any(suffix in f for suffix in IMU_SUFFIXES)))

inconsistent_ids = []
healthy_count = sick_count = unknown_count = 0
min_samples = float('inf')
min_samples_newborn_id = None
validation_results = ""  # Initialize a string to store all validation results

# Define the number of samples to be considered (e.g., 130) and whether to limit samples
num_samples = 130
limit_samples = False  # Set to False to use all available data samples

# Process each newborn
for unique_id in unique_ids:
    validation_results += f"\n===== Child ID: {unique_id} =====\n"
    
    # Get classifications
    id_int = int(unique_id)
    binary_quality = binary_classifications.get(id_int, 'Unknown')
    three_class_quality = three_class_classifications.get(id_int, 'Unknown')
    validation_results += f"Binary classification: {binary_quality}\nThree-class classification: {three_class_quality}\n"

    # Count health classifications
    if binary_quality == 'FM+':
        healthy_count += 1
    elif binary_quality == 'FM-':
        sick_count += 1
    else:
        unknown_count += 1

    samples, frames, features = [], [], []
    validation_results += "\nFiles associated with child:\n"

    # Process each file
    for suffix in IMU_SUFFIXES:
        file_name = f"{unique_id} - {suffix}"
        try:
            file_path = os.path.join(SEGMENTED_IMU_DATA_FRAMES_FOLDER, next(f for f in os.listdir(SEGMENTED_IMU_DATA_FRAMES_FOLDER) if file_name in f))
            result, min_samples, min_samples_newborn_id = process_file(file_path, unique_id, min_samples, min_samples_newborn_id, samples, frames, features)
            validation_results += f" - {file_path}: {result}\n"
        except StopIteration:
            validation_results += f" - {file_name}: File does not exist.\n"
            samples.append(None)
            frames.append(None)
            features.append(None)

    # Check consistency
    validation_results += "\nData consistency check:\n"
    if check_data_consistency(samples, frames, features):
        validation_results += f"All IMU files for newborn {unique_id} have consistent dimensions: {samples[0]} samples, 100 frames, 6 features\n"
    else:
        inconsistent_ids.append(unique_id)

    # Combine IMU data for the newborn and save the result, considering whether to limit samples
    combine_result = combine_segmented_imu_data(
        unique_id=unique_id,
        folder_path=SEGMENTED_IMU_DATA_FRAMES_FOLDER,
        imu_suffixes=IMU_SUFFIXES,
        output_folder=COMBINED_DATA_FOLDER,
        num_samples=num_samples,
        limit_samples=limit_samples
    )
    validation_results += f"\n{combine_result}\n"

# Summary of results
validation_results += "\n===== Summary of Results =====\n"
if inconsistent_ids:
    validation_results += "Inconsistencies were detected in the following newborns:\n"
    for id in inconsistent_ids:
        validation_results += f" - Child ID: {id}\n"
else:
    validation_results += "All IMU files have consistent dimensions for each newborn.\n"

validation_results += f"\nSummary of health classification based on binary classification:\n"
validation_results += f" - Healthy (FM+): {healthy_count}\n"
validation_results += f" - Potential Issues (FM-): {sick_count}\n"
validation_results += f" - Unknown: {unknown_count}\n"

if min_samples_newborn_id is not None:
    validation_results += f"\nThe minimum number of samples recorded in IMU data is: {min_samples} for newborn {min_samples_newborn_id}\n"
else:
    validation_results += "\nNo valid IMU data found to determine the minimum number of samples.\n"

# Save validation results to file
save_results_to_file(validation_results, VALIDATION_OUTPUT_FILE)

# Print the validation results to the console as well
print(validation_results)
