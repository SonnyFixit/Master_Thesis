import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MasterThesis_Config import EXCEL_FILE_PATH, BINARY_CLASSIFICATION_FILE_TXT, BINARY_CLASSIFICATION_FILE_CSV, THREE_CLASS_CLASSIFICATION_FILE_TXT, THREE_CLASS_CLASSIFICATION_FILE_CSV, CLASSIFICATION_FOLDER

def create_classification_file(excel_file_path, binary=True):
    """Create classification file from Excel data."""
    
    # Ensure the classification folder exists
    if not os.path.exists(CLASSIFICATION_FOLDER):
        os.makedirs(CLASSIFICATION_FOLDER)
        print(f"Created classification folder: {CLASSIFICATION_FOLDER}")

    if binary:
        output_file_path_txt = BINARY_CLASSIFICATION_FILE_TXT
        output_file_path_csv = BINARY_CLASSIFICATION_FILE_CSV
    else:
        output_file_path_txt = THREE_CLASS_CLASSIFICATION_FILE_TXT
        output_file_path_csv = THREE_CLASS_CLASSIFICATION_FILE_CSV

    df = pd.read_excel(excel_file_path, usecols=[0, 10])
    df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.strip().replace({r'\*': ''}, regex=True)

    classification_dict = {}
    classification_list = []
    for i, row in df.iterrows():
        base_id = ''.join(filter(str.isdigit, str(row.iloc[0]).strip()))
        fm_quality = str(row.iloc[1]).strip()
        if fm_quality.lower() == 'nan' or fm_quality == '':
            continue
        if binary:
            fm_quality = "FM+" if "FM+" in fm_quality or "FM++" in fm_quality else "FM-" if "FM-" in fm_quality else None
        if fm_quality in ["FM-", "FM+", "FM++"]:
            classification_dict[base_id] = fm_quality
            # Map classification to numeric value for machine learning
            numeric_class = 0 if fm_quality == "FM-" else 1 if fm_quality == "FM+" else 2
            classification_list.append([base_id, numeric_class])
    
    # Save the text file
    with open(output_file_path_txt, 'w') as file:
        for child_id, fm_quality in classification_dict.items():
            file.write(f"ID: {child_id}, FM Quality: {fm_quality}\n")
    print(f"Classification file created: {output_file_path_txt}")

    # Save the CSV file for machine learning
    classification_df = pd.DataFrame(classification_list, columns=['ID', 'FM Quality Numeric'])
    classification_df.to_csv(output_file_path_csv, index=False)
    print(f"Classification CSV file created: {output_file_path_csv}")
