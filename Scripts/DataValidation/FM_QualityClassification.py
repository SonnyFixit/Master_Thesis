import os
import sys
import pandas as pd
from datetime import datetime

"""
IMU Data Classification Script for Master Thesis

This script processes FM (Functional Movement) quality data from an Excel file and creates classification files 
for binary and multi-class models. It includes functionality for logging, duplicate checking, and creating files 
for different classification variants.

Key Features:
1. **Classification File Creation**: 
   - Generates classification files (TXT and CSV) for binary and three-class FM classification.
   - Handles two binary classification variants: 
     - **Standard Binary**: FM+ (positive) and FM- (negative) based on FM quality.
     - **New Binary (FM- includes FM+)**: Considers FM- to include both FM- and FM+ classes.
   - Removes duplicates to ensure each subject ID appears once in the classification.

2. **Logging**: 
   - Logs the number of samples for each class (FM-, FM+, FM++) in different classification variants.
   - Logs selected columns from the Excel file, such as Subject ID, Record Length, and FM Quality.

3. **Configurable Paths**: 
   - File paths and folder locations are managed via an external configuration file `MasterThesis_Config.py`.
"""

# Add configuration path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import paths from configuration file
from MasterThesis_Config import (
    EXCEL_FILE_PATH, 
    BINARY_CLASSIFICATION_FILE_TXT, 
    BINARY_CLASSIFICATION_FILE_CSV, 
    THREE_CLASS_CLASSIFICATION_FILE_TXT, 
    THREE_CLASS_CLASSIFICATION_FILE_CSV, 
    CLASSIFICATION_FOLDER, 
    CLASSIFICATION_LOG_FILE
)

# Paths for the new binary classification variant
NEW_BINARY_CLASSIFICATION_FILE_TXT = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary_Negative.txt')
NEW_BINARY_CLASSIFICATION_FILE_CSV = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary_Negative.csv')

def create_classification_file(excel_file_path, binary=True, variant='standard'):
    """
    Create classification file from Excel data with logging and duplicate ID checks.

    Parameters:
    - excel_file_path: Path to the Excel file containing FM quality data.
    - binary: Boolean indicating if binary classification is used. If False, three-class classification is created.
    - variant: 'standard' for regular binary classification, 'negative' for binary where FM- includes FM+.

    The function processes each row in the Excel file, classifies the data, and generates both TXT and CSV files.
    """

    # Ensure the classification folder exists
    if not os.path.exists(CLASSIFICATION_FOLDER):
        os.makedirs(CLASSIFICATION_FOLDER)
        print(f"Created classification folder: {CLASSIFICATION_FOLDER}")

    # Set output paths based on the classification type
    if binary:
        if variant == 'standard':
            output_file_path_txt = BINARY_CLASSIFICATION_FILE_TXT
            output_file_path_csv = BINARY_CLASSIFICATION_FILE_CSV
        elif variant == 'negative':
            output_file_path_txt = NEW_BINARY_CLASSIFICATION_FILE_TXT
            output_file_path_csv = NEW_BINARY_CLASSIFICATION_FILE_CSV
    else:
        output_file_path_txt = THREE_CLASS_CLASSIFICATION_FILE_TXT
        output_file_path_csv = THREE_CLASS_CLASSIFICATION_FILE_CSV

    # Read data from the Excel file
    df = pd.read_excel(excel_file_path, usecols=[0, 10])
    df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.strip().replace({r'\*': ''}, regex=True)

    classification_dict = {}
    classification_list = []

    for i, row in df.iterrows():
        base_id = ''.join(filter(str.isdigit, str(row.iloc[0]).strip()))
        fm_quality = str(row.iloc[1]).strip()
        
        # Log the original values for analysis
        print(f"Processing index {i}: ID={base_id}, FM Quality={fm_quality}")

        # Skip empty or questionable classifications
        if fm_quality.lower() == 'nan' or fm_quality == '' or '?' in fm_quality:
            print(f"Skipping index {i} due to empty or questionable FM Quality")
            continue

        # Classification logic based on binary or three-class classification
        if binary:
            if variant == 'standard':
                fm_quality = "FM+" if "FM+" in fm_quality or "FM++" in fm_quality else "FM-" if "FM-" in fm_quality else 'Unknown'
            elif variant == 'negative':
                if fm_quality in ["FM-", "FM-/FM-", "FM+"]:
                    fm_quality = "FM-"
                elif fm_quality in ["FM++", "FM+/FM++", "FM++/FM++"]:
                    fm_quality = "FM+"
                else:
                    print(f"Skipping index {i} as FM Quality does not fit binary negative criteria")
                    continue  # Skip unclassified values

        # Only classify known categories
        if fm_quality in ["FM-", "FM+", "FM++"]:
            # Check for duplicate IDs
            if base_id in classification_dict:
                print(f"Duplicate ID found: {base_id} at index {i}, skipping this entry.")
                continue  # Skip duplicates
            classification_dict[base_id] = fm_quality
            # Map classification to numeric value for machine learning
            numeric_class = 0 if fm_quality == "FM-" else 1 if fm_quality == "FM+" else 2
            classification_list.append([base_id, numeric_class])
            print(f"Index {i} classified as {fm_quality} (numeric: {numeric_class})")
        else:
            print(f"Skipping index {i} as FM Quality is {fm_quality}")

    # Save the classification as a text file
    with open(output_file_path_txt, 'w') as file:
        for child_id, fm_quality in classification_dict.items():
            file.write(f"ID: {child_id}, FM Quality: {fm_quality}\n")
    print(f"Classification file created: {output_file_path_txt}")

    # Save the classification as a CSV file for machine learning
    classification_df = pd.DataFrame(classification_list, columns=['ID', 'FM Quality Numeric'])

    # Check for duplicates in the DataFrame before saving
    if classification_df['ID'].duplicated().any():
        print("Warning: Duplicate IDs found in the CSV file, removing duplicates before saving.")
        classification_df = classification_df.drop_duplicates(subset='ID')
    
    classification_df.to_csv(output_file_path_csv, index=False)
    print(f"Classification CSV file created: {output_file_path_csv}")

def log_classification_counts():
    """
    Log the classification counts from the generated classification files (binary and three-class).
    This function calculates the distribution of FM-, FM+, and FM++ classes and logs them to a file.
    """
    counts_binary = {'FM-': 0, 'FM+': 0, 'Unknown': 0}
    counts_three_class = {'FM-': 0, 'FM+': 0, 'FM++': 0, 'Unknown': 0}
    counts_negative_binary = {'FM-': 0, 'FM+': 0, 'Unknown': 0}

    # Count occurrences in the standard binary classification file
    binary_df = pd.read_csv(BINARY_CLASSIFICATION_FILE_CSV)
    for _, row in binary_df.iterrows():
        value = row['FM Quality Numeric']
        if value == 0:
            counts_binary['FM-'] += 1
        elif value == 1:
            counts_binary['FM+'] += 1
        else:
            counts_binary['Unknown'] += 1

    # Count occurrences in the three-class classification file
    three_class_df = pd.read_csv(THREE_CLASS_CLASSIFICATION_FILE_CSV)
    for _, row in three_class_df.iterrows():
        value = row['FM Quality Numeric']
        if value == 0:
            counts_three_class['FM-'] += 1
        elif value == 1:
            counts_three_class['FM+'] += 1
        elif value == 2:
            counts_three_class['FM++'] += 1
        else:
            counts_three_class['Unknown'] += 1

    # Count occurrences in the new binary classification file
    negative_binary_df = pd.read_csv(NEW_BINARY_CLASSIFICATION_FILE_CSV)
    for _, row in negative_binary_df.iterrows():
        value = row['FM Quality Numeric']
        if value == 0:
            counts_negative_binary['FM-'] += 1
        elif value == 1:
            counts_negative_binary['FM+'] += 1
        else:
            counts_negative_binary['Unknown'] += 1

    # Log classification counts
    log_entry = f"Classification Log - Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_entry += "-" * 50 + "\n"
    log_entry += "Binary Classification Counts (Standard):\n"
    log_entry += f" - FM-: {counts_binary['FM-']}\n"
    log_entry += f" - FM+: {counts_binary['FM+']}\n"
    log_entry += f" - Unknown: {counts_binary['Unknown']}\n\n"

    log_entry += "Three-Class Classification Counts:\n"
    log_entry += f" - FM-: {counts_three_class['FM-']}\n"
    log_entry += f" - FM+: {counts_three_class['FM+']}\n"
    log_entry += f" - FM++: {counts_three_class['FM++']}\n"
    log_entry += f" - Unknown: {counts_three_class['Unknown']}\n\n"

    log_entry += "Binary Classification Counts (New - FM- includes FM+):\n"
    log_entry += f" - FM-: {counts_negative_binary['FM-']}\n"
    log_entry += f" - FM+: {counts_negative_binary['FM+']}\n"
    log_entry += f" - Unknown: {counts_negative_binary['Unknown']}\n\n"

    # Save log to the file (overwrite mode 'w')
    with open(CLASSIFICATION_LOG_FILE, 'w') as log_file:
        log_file.write(log_entry)
    print(f"Classification log updated: {CLASSIFICATION_LOG_FILE}")

def log_excel_columns():
    """
    Log selected columns from the Excel file such as Subject ID, Record Length, and FM Quality.
    This logs additional information for each subject to the classification log.
    """
    df = pd.read_excel(EXCEL_FILE_PATH, usecols=["Subject ID", "IMU/iPhone record length (in seconds)", "FM quality"])

    log_header = f"Excel Columns Log - Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_header += "-" * 50 + "\n"
    log_header += f"{'Subject ID':<20} {'Record Length (s)':<40} {'FM Quality':<20}\n"
    log_header += "-" * 50 + "\n"

    log_rows = ""
    for _, row in df.iterrows():
        log_rows += f"{str(row['Subject ID']):<20} {str(row['IMU/iPhone record length (in seconds)']):<40} {str(row['FM quality']):<20}\n"

    # Append log to the file
    with open(CLASSIFICATION_LOG_FILE, 'a') as log_file:
        log_file.write(log_header + log_rows + "\n")
    print(f"Excel columns log updated: {CLASSIFICATION_LOG_FILE}")

def clear_log_file():
    """
    Clears the classification log file.
    This function erases any previous log entries to provide a clean slate for new logs.
    """
    if os.path.exists(CLASSIFICATION_LOG_FILE):
        with open(CLASSIFICATION_LOG_FILE, 'w') as log_file:
            log_file.write("")  # Clear the content of the file
        print(f"Log file cleared: {CLASSIFICATION_LOG_FILE}")

# Clear the log file before running the rest of the operations
clear_log_file()

# Example usage
create_classification_file(EXCEL_FILE_PATH, binary=True, variant='standard')
create_classification_file(EXCEL_FILE_PATH, binary=True, variant='negative')
create_classification_file(EXCEL_FILE_PATH, binary=False)
log_classification_counts()
log_excel_columns()
