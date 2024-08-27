import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MasterThesis_Config import SEGMENTED_IMU_DATA_FRAMES_FOLDER, COMBINED_DATA_FOLDER, IMU_SUFFIXES

def combine_segmented_imu_data(unique_id, folder_path=SEGMENTED_IMU_DATA_FRAMES_FOLDER, imu_suffixes=IMU_SUFFIXES, output_folder=COMBINED_DATA_FOLDER, num_samples=130):
    combined_data = []

    for suffix in imu_suffixes:
        file_name = f"{unique_id} - {suffix}"
        try:
            file_path = os.path.join(folder_path, next(f for f in os.listdir(folder_path) if file_name in f))
            data = np.load(file_path)

            # Trim or use all available samples if shorter than num_samples
            data = data[:num_samples, :, :]

            if len(combined_data) == 0:
                combined_data = data
            else:
                combined_data = np.concatenate((combined_data, data), axis=2)
        except StopIteration:
            return f" - {file_name}: File does not exist or cannot be combined."

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the combined data to a new file
    combined_file_path = os.path.join(output_folder, f"IMU_CombineSegmentedData_{unique_id}.npy")
    np.save(combined_file_path, combined_data)
    
    return f"Combined data saved to {combined_file_path} with shape {combined_data.shape}."

# Example usage:
# result = combine_segmented_imu_data(unique_id="001")
# print(result)
