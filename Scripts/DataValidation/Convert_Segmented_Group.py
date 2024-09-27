import pandas as pd
import os

"""
IMU Segmented Data Expansion Script

This script reads a classification file and a segmented group information file and expands the data by generating
a unique Segment ID for each individual sample within the range specified by the Start Row and End Row. 
The expanded data is then saved to a new CSV file on the desktop.

Files:
1. `classification_file`: Contains classification information about each sample (e.g., original or augmented).
2. `segment_info_file`: Contains segmented group info, specifying the start and end row for each child sample.
3. `output_file`: The expanded data is saved to this CSV file on the desktop.

Process:
- For each row in the segmented group info, the script calculates the number of samples between the start and end row.
- A unique Segment ID is generated for each sample, and the sample is appended to the expanded dataset.
- The final expanded data is saved in the specified output file.
"""

# Paths to the CSV files
classification_file = r"C:\GitRepositories\Master_Thesis\FM_QualityClassification\FM_QualityClassification_Binary_Updated.csv"
segment_info_file = r"C:\GitRepositories\Master_Thesis\FM_QualityClassification\IMU_Segmented_Group_Info_Augmented.csv"

# Path to save the new file to the desktop
output_file = os.path.join(os.path.expanduser('~'), 'Desktop', 'Expanded_IMU_Segmented_Data.csv')

# Load data from the CSV files
df_classification = pd.read_csv(classification_file)
df_segment_info = pd.read_csv(segment_info_file)

# New list for storing expanded samples
expanded_data = []
current_segment_id = 1  # Start with Segment ID = 1

# Iterate over each row in the segment_info to expand samples
for _, row in df_segment_info.iterrows():
    child_id = row['Child ID']
    classification = row['Classification']
    augmented = row['Augmented']
    start_row = row['Start Row']
    end_row = row['End Row']
    
    # Calculate the number of samples based on the range (Start Row -> End Row)
    num_samples = end_row - start_row + 1
    
    # Generate rows for each sample in the range Start Row -> End Row with unique Segment IDs
    for segment_id in range(num_samples):
        expanded_data.append({
            'Segment ID': current_segment_id,  # Unique Segment ID
            'Child ID': child_id,
            'Classification': classification,
            'Augmented': augmented
        })
        current_segment_id += 1  # Increment Segment ID for each sample

# Convert the list to a DataFrame
expanded_df = pd.DataFrame(expanded_data)

# Reorder the columns
expanded_df = expanded_df[['Segment ID', 'Child ID', 'Classification', 'Augmented']]

# Save to a CSV file on the desktop
expanded_df.to_csv(output_file, index=False)

print(f"Processing completed. The file has been saved to the desktop as: {output_file}")
