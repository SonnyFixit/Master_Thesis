import os

"""
Master Thesis Configuration Script

This script sets up paths and files necessary for running the Master's thesis project on 
the analysis of infants' general movements using machine learning techniques. 
It defines key folders, files, and configurations related to data processing, 
classification, logging, and neural network training.

---

Important:
If this script is being run on a different machine, paths should be manually set 
according to the local system structure by adjusting the BASE_PATH and other relevant paths.

---

Folders and Files:
1. BASE_PATH: The base directory for the repository. All paths are relative to this.
2. DATA_FOLDER: The folder containing raw and segmented IMU data.
3. CLASSIFICATION_FOLDER: Directory for classification-related files (binary and three-class).
4. LOGS_FOLDER: Directory where log files are stored, including classification and validation logs.
5. AUGMENTED_DATA_FOLDER: Folder for storing augmented IMU data.
6. Paths for classification files (TXT and CSV) for both binary and three-class classifications.
7. IMU_SUFFIXES: List of suffixes for various IMU data files related to different limbs.

"""

# Base path to the repository
BASE_PATH = r'C:\GitRepositories\Master_Thesis'

# Paths to folders
DATA_FOLDER = os.path.join(BASE_PATH, 'Data')  # Folder containing raw and processed data
SEGMENTED_IMU_DATA_FRAMES_FOLDER = os.path.join(DATA_FOLDER, 'segmented_imu_data_frames')  
COMBINED_DATA_FOLDER = os.path.join(BASE_PATH, 'Data_IMU_Segmented_Combined')  
DOCUMENTS_REVIEW_FOLDER = r'D:\Master_Thesis\MasterThesis_Documents_Review' 

# Classification folder
CLASSIFICATION_FOLDER = os.path.join(BASE_PATH, 'FM_QualityClassification') 

# Log folder
LOGS_FOLDER = os.path.join(BASE_PATH, 'Logs')

# Classification files (binary and three-class formats)
BINARY_CLASSIFICATION_FILE_TXT = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary.txt')
BINARY_CLASSIFICATION_FILE_CSV = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary.csv')  
THREE_CLASS_CLASSIFICATION_FILE_TXT = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_ThreeClass.txt') 
THREE_CLASS_CLASSIFICATION_FILE_CSV = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_ThreeClass.csv')  

# Additional paths for the new binary classification variant (FM- includes FM+)
NEW_BINARY_CLASSIFICATION_FILE_TXT = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary_Negative.txt')
NEW_BINARY_CLASSIFICATION_FILE_CSV = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary_Negative.csv')  

# Path to the validation log file
VALIDATION_OUTPUT_FILE = os.path.join(LOGS_FOLDER, 'IMU_Validation_Log.txt')

# Path to the classification log file
CLASSIFICATION_LOG_FILE = os.path.join(LOGS_FOLDER, 'FM_Classification_Log.txt')

# Updated Path to the Excel file with data (new location)
EXCEL_FILE_PATH = os.path.join(BASE_PATH, 'Data acquisition log file.xlsx') 

# IMU file suffixes for different limbs (used in data preprocessing)
IMU_SUFFIXES = ['DataFramesLF.npy', 'DataFramesLH.npy', 'DataFramesRF.npy', 'DataFramesRH.npy']

# Paths for Neural Network Training
NN_COMBINED_DATA_FOLDER = COMBINED_DATA_FOLDER  # Folder for combined IMU data used for NN training

# Boolean to select which classification file to use (standard or negative)
USE_NEGATIVE_CLASSIFICATION = True  # Set to True to use the 'negative' classification set, False for the standard set

# Set the Neural Network classification file based on the USE_NEGATIVE_CLASSIFICATION value
if USE_NEGATIVE_CLASSIFICATION:
    NN_CLASSIFICATION_FILE = NEW_BINARY_CLASSIFICATION_FILE_CSV 
else:
    NN_CLASSIFICATION_FILE = BINARY_CLASSIFICATION_FILE_CSV 

# Folder for augmented data
AUGMENTED_DATA_FOLDER = os.path.join(BASE_PATH, 'Data_IMU_Augmented')

# New log file for the augmentation process
AUGMENTATION_LOG_FILE = os.path.join(LOGS_FOLDER, 'IMU_Augmentation_Log.txt')  

# Ensure that the new folder for augmented data exists, and create it if necessary
if not os.path.exists(AUGMENTED_DATA_FOLDER):
    os.makedirs(AUGMENTED_DATA_FOLDER) 
    print(f"Created folder: {AUGMENTED_DATA_FOLDER}")  

