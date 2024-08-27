import os

# Base path to the repository
BASE_PATH = r'C:\GitRepositories\Master_Thesis'

# Paths to folders
DATA_FOLDER = os.path.join(BASE_PATH, 'Data')
SEGMENTED_IMU_DATA_FRAMES_FOLDER = os.path.join(DATA_FOLDER, 'segmented_imu_data_frames')
COMBINED_DATA_FOLDER = os.path.join(BASE_PATH, 'Data_IMU_Segmented_Combined')
DOCUMENTS_REVIEW_FOLDER = r'D:\Master_Thesis\MasterThesis_Documents_Review'

# Classification folder
CLASSIFICATION_FOLDER = os.path.join(BASE_PATH, 'FM_QualityClassification')

# Log folder
LOGS_FOLDER = os.path.join(BASE_PATH, 'Logs')

# Classification files
BINARY_CLASSIFICATION_FILE_TXT = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary.txt')
BINARY_CLASSIFICATION_FILE_CSV = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_Binary.csv')
THREE_CLASS_CLASSIFICATION_FILE_TXT = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_ThreeClass.txt')
THREE_CLASS_CLASSIFICATION_FILE_CSV = os.path.join(CLASSIFICATION_FOLDER, 'FM_QualityClassification_ThreeClass.csv')

# Path to the validation log file
VALIDATION_OUTPUT_FILE = os.path.join(LOGS_FOLDER, 'IMU_Validation_Log.txt')

# Path to the Excel file with data
EXCEL_FILE_PATH = os.path.join(DOCUMENTS_REVIEW_FOLDER, 'Data acquisition log file.xlsx')

# IMU file suffixes
IMU_SUFFIXES = ['DataFramesLF.npy', 'DataFramesLH.npy', 'DataFramesRF.npy', 'DataFramesRH.npy']
