import os

# Base path to the repository
BASE_PATH = r'C:\GitRepositories\Master_Thesis'

# Paths to folders
DATA_FOLDER = os.path.join(BASE_PATH, 'Data')
SEGMENTED_IMU_DATA_FRAMES_FOLDER = os.path.join(DATA_FOLDER, 'segmented_imu_data_frames')
COMBINED_DATA_FOLDER = os.path.join(BASE_PATH, 'Data_IMU_Segmented_Combined')
DOCUMENTS_REVIEW_FOLDER = r'D:\Master_Thesis\MasterThesis_Documents_Review'  # Not used for the Excel file anymore

# Classification folder
CLASSIFICATION_FOLDER = os.path.join(BASE_PATH, 'FM_QualityClassification')

# Log folder
LOGS_FOLDER = os.path.join(BASE_PATH, 'Logs')

# Classification files
BINARY_CLASSIFICATION_FILE_TXT = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary.txt')
BINARY_CLASSIFICATION_FILE_CSV = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary.csv')
THREE_CLASS_CLASSIFICATION_FILE_TXT = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_ThreeClass.txt')
THREE_CLASS_CLASSIFICATION_FILE_CSV = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_ThreeClass.csv')

# Additional paths for the new binary classification variant
NEW_BINARY_CLASSIFICATION_FILE_TXT = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary_Negative.txt')
NEW_BINARY_CLASSIFICATION_FILE_CSV = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary_Negative.csv')

# Path to the validation log file
VALIDATION_OUTPUT_FILE = os.path.join(LOGS_FOLDER, 'IMU_Validation_Log.txt')

# Path to the classification log file
CLASSIFICATION_LOG_FILE = os.path.join(LOGS_FOLDER, 'FM_Classification_Log.txt')

# Updated Path to the Excel file with data (new location)
EXCEL_FILE_PATH = os.path.join(BASE_PATH, 'Data acquisition log file.xlsx')

# IMU file suffixes
IMU_SUFFIXES = ['DataFramesLF.npy', 'DataFramesLH.npy', 'DataFramesRF.npy', 'DataFramesRH.npy']

# Paths for Neural Network Training
NN_COMBINED_DATA_FOLDER = COMBINED_DATA_FOLDER

# Boolean to select which classification file to use
USE_NEGATIVE_CLASSIFICATION = True  # Set to True to use the 'negative' classification set, False for the standard set

# Set the NN classification file based on the USE_NEGATIVE_CLASSIFICATION value
if USE_NEGATIVE_CLASSIFICATION:
    NN_CLASSIFICATION_FILE = NEW_BINARY_CLASSIFICATION_FILE_CSV
else:
    NN_CLASSIFICATION_FILE = BINARY_CLASSIFICATION_FILE_CSV
    
# Extend configuration file with new paths for augmentation
# Base path to the repository
BASE_PATH = r'C:\GitRepositories\Master_Thesis'

# New folder for augmented data
AUGMENTED_DATA_FOLDER = os.path.join(BASE_PATH, 'Data_IMU_Augmented')

# New log file for the augmentation process
AUGMENTATION_LOG_FILE = os.path.join(LOGS_FOLDER, 'IMU_Augmentation_Log.txt')

# Ensure that the new folder exists
if not os.path.exists(AUGMENTED_DATA_FOLDER):
    os.makedirs(AUGMENTED_DATA_FOLDER)

