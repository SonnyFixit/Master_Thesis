import os
import numpy as np
import csv
from datetime import datetime

# Paths to the necessary directories and files
input_folder = r"C:\GitRepositories\Master_Thesis\Data\synchronised_imu_kinect3d_data_frames"
output_folder = r"C:\GitRepositories\Master_Thesis\Data_Kinect_Combined"
combined_file_path = os.path.join(output_folder, 'Kinect_Combined_60sDataset.npy')
log_file_path = os.path.join(output_folder, 'Kinect_Combined_60sDataset_Log.txt')
classification_output_file = os.path.join(output_folder, 'Kinect_QualityClassification_Binary_60sData.csv')
segment_info_output_file = os.path.join(output_folder, 'Kinect_Segmented_Group_Info_60s.csv')

# Define the path to the binary classification file (Corrected path)
BINARY_CLASSIFICATION_FILE = r"C:\GitRepositories\Master_Thesis\FM_QualityClassification\FM_QualityClassification_Binary.csv"

# IDs to be excluded
EXCLUDED_IDS = ["042"]

# Function to log messages
def log_message(message):
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)

# Function to read binary classification
def read_binary_classification(file_path):
    classifications = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                child_id = row[0].strip()
                quality = row[1].strip()
                classifications[child_id.zfill(3)] = quality
    return classifications

# Function to create the combined 60-second Kinect dataset
def create_combined_kinect_60s_dataset(input_folder=input_folder, output_file=combined_file_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        log_message(f"Created output folder: {output_folder}")

    combined_dataset = []
    child_data_info = []  # Stores information about children and their data dimensions
    excluded_samples = []
    total_60s_segments = 0

    log_message("Starting creation of the combined 60-second Kinect dataset")

    # Iterate through each file in the folder
    for file_name in sorted(os.listdir(input_folder)):
        if "KinectDataFrames.npy" in file_name:
            child_id = file_name.split('-')[1].strip().split()[0]  # Extract child ID from the file name

            # Skip selected IDs
            if child_id in EXCLUDED_IDS:
                log_message(f"Skipping child ID: {child_id} as it is in the excluded list.")
                excluded_samples.append(child_id)
                continue

            file_path = os.path.join(input_folder, file_name)
            try:
                child_data = np.load(file_path)
                log_message(f"Loaded data from {file_path} with shape {child_data.shape}")

                # Calculate the number of full 60-second samples possible
                num_full_60s_samples = child_data.shape[0] // 1800  # 1800 frames = 60 seconds at 30 FPS

                if num_full_60s_samples > 0:
                    # Reshape and collect all full 60-second samples
                    full_60s_samples = child_data[:num_full_60s_samples * 1800].reshape((num_full_60s_samples, 1800, 99))
                    
                    # Combine data into the large dataset
                    if len(combined_dataset) == 0:
                        combined_dataset = full_60s_samples
                    else:
                        combined_dataset = np.concatenate((combined_dataset, full_60s_samples), axis=0)

                    log_message(f"Added {num_full_60s_samples} full 60-second samples for child ID {child_id} to the combined dataset")
                    total_60s_segments += num_full_60s_samples

                    # Add information about the child and data dimensions
                    child_data_info.append((child_id, full_60s_samples.shape))

            except Exception as e:
                log_message(f"Error loading data from {file_path}: {e}")

    # Save the combined dataset to a new file
    np.save(output_file, combined_dataset)
    log_message(f"Combined 60-second Kinect dataset created and saved to {output_file} with shape {combined_dataset.shape}\n")

    # Create log summary
    log_message("===== Summary =====")
    log_message(f"Total number of files included: {len(child_data_info)}")
    for child_id, shape in child_data_info:
        log_message(f"Child ID: {child_id}, Data Shape: {shape}")

    # Log excluded samples
    log_message("===== Excluded Samples =====")
    log_message(f"Excluded child IDs: {', '.join(excluded_samples)}")

    return child_data_info, excluded_samples

def create_classification_file_60s(child_data_info, classification_file_path, output_file_path, excluded_samples):
    # Load classification data from binary classification file
    classifications = read_binary_classification(classification_file_path)
    log_message(f"Loaded binary classification data from {classification_file_path}")

    # Initialize classification data to write
    classification_data = []
    segment_info_data = []

    # Initialize counters for FM- and FM+ children
    fm_minus_count = 0
    fm_plus_count = 0
    segment_id = 1  # Start from 1 for the Segment ID

    # Iterate over child data info to generate classification
    for child_id, shape in child_data_info:
        num_60s_segments = shape[0]
        classification = classifications.get(child_id)
        if classification is None:
            log_message(f"No classification found for child ID: {child_id}, skipping.")
            continue

        # Increment counters based on classification
        if classification == '0':
            fm_minus_count += 1
        elif classification == '1':
            fm_plus_count += 1

        # Add classification for each 60-second segment corresponding to this child
        for i in range(num_60s_segments):
            classification_data.append([segment_id, child_id, classification])  # Use Segment ID, Child ID, Classification
            segment_info_data.append([segment_id, child_id, classification])
            segment_id += 1

        log_message(f"Child ID: {child_id} - Assigned classification '{classification}' for 60-second segments.")

    # Write the classification data to a new CSV file
    with open(output_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Segment ID", "Child ID", "Classification"])  # Write header
        csv_writer.writerows(classification_data)

    log_message(f"Classification file created and saved to {output_file_path} with {len(classification_data)} rows")

    # Write the segment info data to a new CSV file
    with open(segment_info_output_file, 'w', newline='') as segment_csvfile:
        segment_csv_writer = csv.writer(segment_csvfile)
        segment_csv_writer.writerow(["Segment ID", "Child ID", "Classification"])  # Header
        segment_csv_writer.writerows(segment_info_data)

    log_message(f"Segment information file created and saved to {segment_info_output_file} with {len(segment_info_data)} rows")

    # Log the counts of FM- and FM+ children
    log_message("===== Summary of FM- and FM+ Children =====")
    log_message(f"Number of children classified as FM- (0): {fm_minus_count}")
    log_message(f"Number of children classified as FM+ (1): {fm_plus_count}")

# Main execution
if __name__ == "__main__":
    # Create the combined 60-second Kinect dataset
    child_data_info, excluded_samples = create_combined_kinect_60s_dataset()

    # Create classification file for the combined dataset
    create_classification_file_60s(child_data_info, BINARY_CLASSIFICATION_FILE, classification_output_file, excluded_samples)
