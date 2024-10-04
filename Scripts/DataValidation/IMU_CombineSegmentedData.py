import os
import sys
import numpy as np
from datetime import datetime
from MasterThesis_Config import SEGMENTED_IMU_DATA_FRAMES_FOLDER, COMBINED_DATA_FOLDER, IMU_SUFFIXES

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



"""
IMU Data Combination Script

This script is designed to combine segmented IMU (Inertial Measurement Unit) data for a specific child identified by a unique ID. 
The script concatenates data from multiple suffixes, producing a single output file per child. Optionally, it allows limiting the 
number of samples per segment to a predefined count. This is particularly useful in processing and preparing data for further analysis, 
such as machine learning model training.

Key Features:
- Combines segmented IMU data for a given child across multiple suffixes.
- Supports limiting the number of samples per segment if desired.
- Logs all steps to a log file with timestamps, providing visibility into the process.
- Ensures that the combined data is stored in a specified output folder.

Techniques:
- **Data Concatenation**: Combines segmented data across multiple suffixes along the feature axis.
- **Sample Limiting**: Optionally limits the number of samples to a fixed count per segment.

Dependencies:
- Numpy is used to handle numerical operations and loading/saving of data files.
- Standard Python modules such as `os` for path operations and `datetime` for logging.

Parameters:
- `unique_id`: A string representing the unique identifier for the child whose data is being combined.
- `folder_path`: Path to the folder containing segmented IMU data.
- `imu_suffixes`: List of suffixes representing different IMU data segments.
- `output_folder`: Path to the folder where the combined data will be saved.
- `num_samples`: Number of samples to limit each file to if `limit_samples` is True.
- `limit_samples`: Boolean flag to enable or disable sample limitation.

Example Usage:
    result = combine_segmented_imu_data(unique_id="001", limit_samples=True)
    print(result)
"""

# Path to the log file
LOG_FILE_PATH = r'C:\GitRepositories\Master_Thesis\logs\IMU_Segmented_Combined_Log.txt'

def log_message(message):
    """
    Logs a message to the log file with a timestamp.
    
    Args:
        message (str): The message to log.
    """
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)

def combine_segmented_imu_data(unique_id, folder_path=SEGMENTED_IMU_DATA_FRAMES_FOLDER, imu_suffixes=IMU_SUFFIXES, output_folder=COMBINED_DATA_FOLDER, num_samples=130, limit_samples=False):
    """
    Combines segmented IMU data for a given unique ID by concatenating data from different suffixes.
    
    Args:
        unique_id (str): Unique identifier for the child.
        folder_path (str): Path to the folder containing segmented IMU data files.
        imu_suffixes (list): List of suffixes for the IMU data files.
        output_folder (str): Folder where the combined data will be saved.
        num_samples (int): Number of samples to limit each file to if limit_samples is True.
        limit_samples (bool): If True, limits the data to num_samples per file; if False, uses all available data.

    Returns:
        str: Message indicating the result of the combination process.
    """
    combined_data = []
    log_message(f"Starting data combination for child ID: {unique_id}")

    for suffix in imu_suffixes:
        file_name = f"{unique_id} - {suffix}"
        try:
            file_path = os.path.join(folder_path, next(f for f in os.listdir(folder_path) if file_name in f))
            data = np.load(file_path)
            log_message(f"Loaded data from {file_path} with shape {data.shape}")

            # Apply sample limitation if enabled
            if limit_samples:
                data = data[:num_samples, :, :]
                log_message(f"Limiting data to {num_samples} samples.")
            else:
                log_message(f"Using all available data samples.")

            # Combine the data along the feature axis
            if len(combined_data) == 0:
                combined_data = data
            else:
                combined_data = np.concatenate((combined_data, data), axis=2)
                log_message(f"Combined data shape updated to {combined_data.shape}")

        except StopIteration:
            log_message(f" - {file_name}: File does not exist or cannot be combined.")
            return f" - {file_name}: File does not exist or cannot be combined."

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        log_message(f"Created output folder: {output_folder}")

    # Save the combined data to a new file
    combined_file_path = os.path.join(output_folder, f"IMU_CombineSegmentedData_{unique_id}.npy")
    np.save(combined_file_path, combined_data)
    log_message(f"Combined data saved to {combined_file_path} with shape {combined_data.shape}\n")

    return f"Combined data saved to {combined_file_path} with shape {combined_data.shape}."

# Example usage:
# result = combine_segmented_imu_data(unique_id="001", limit_samples=True)
# print(result)
