import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from datetime import datetime
import psutil

# Set environment variables for BLAS libraries
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Adding path to configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MasterThesis_Config import NN_COMBINED_DATA_FOLDER, NN_CLASSIFICATION_FILE, LOGS_FOLDER

# Enhanced function for logging
def log_message(message, log_file_path):
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')
    print(message)

# Function to create training and validation sets dynamically during training
def create_training_validation_sets(data_file, classification_file, segment_info_file, log_file_path):
    log_message("Loading dataset for training/validation set creation...", log_file_path)
    data = np.load(data_file)
    max_index = data.shape[0]
    df_classification = pd.read_csv(classification_file, index_col=0)
    df_segment_info = pd.read_csv(segment_info_file)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    unique_ids = df_segment_info['Child ID'].unique()
    labels = df_segment_info.groupby('Child ID')['Classification'].first().values

    fold_sets = []

    fold = 1
    for train_idx, test_idx in skf.split(unique_ids, labels):
        log_message(f"\n--- Creating training and validation sets for fold {fold} ---", log_file_path)

        train_ids = unique_ids[train_idx]
        test_ids = unique_ids[test_idx]

        log_message(f"Training IDs for fold {fold}: {list(train_ids)}", log_file_path)
        log_message(f"Validation IDs for fold {fold}: {list(test_ids)}", log_file_path)

        train_segments = []
        train_labels = []
        val_segments = []
        val_labels = []

        # Training set
        for child_id in train_ids:
            child_data = df_segment_info[df_segment_info['Child ID'] == child_id]
            start_row = child_data['Start Row'].values[0] - 1
            end_row = child_data['End Row'].values[0] - 1
            segments = list(range(start_row, end_row + 1))
            valid_segments = [seg for seg in segments if 0 <= seg < max_index and seg in df_classification.index]
            if not valid_segments:
                log_message(f"No valid segments found for Child ID {child_id}. Skipping...", log_file_path)
                continue
            train_segments.extend(valid_segments)
            train_labels.extend(df_classification.loc[valid_segments, 'Classification'].values)

        # Validation set
        for child_id in test_ids:
            child_data = df_segment_info[df_segment_info['Child ID'] == child_id]
            start_row = child_data['Start Row'].values[0] - 1
            end_row = child_data['End Row'].values[0] - 1
            segments = list(range(start_row, end_row + 1))
            valid_segments = [seg for seg in segments if 0 <= seg < max_index and seg in df_classification.index]
            if not valid_segments:
                log_message(f"No valid segments found for Child ID {child_id}. Skipping...", log_file_path)
                continue
            val_segments.extend(valid_segments)
            val_labels.extend(df_classification.loc[valid_segments, 'Classification'].values)

        fold_sets.append((train_segments, train_labels, val_segments, val_labels))
        fold += 1

    log_message("--- Training and validation sets creation completed ---", log_file_path)
    return fold_sets

# Definition of the dataset class
class IMUBigDataset(Dataset):
    def __init__(self, data_file, classification_file):
        self.data = np.load(data_file)
        df_classification = pd.read_csv(classification_file)
        self.labels = torch.tensor(df_classification['Classification'].values, dtype=torch.long)
        self.lengths = [100] * len(self.labels)
        if len(self.labels) != len(self.data):
            raise ValueError("Mismatch between the number of labels and the number of data samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Definition of the CNN model
class IMUCNN(nn.Module):
    def __init__(self, input_size=24, num_classes=2):
        super(IMUCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 12, 128)  # Adjusted input size based on downsampled output from convolutions
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Reshape input: (batch_size, num_frames, num_features) -> (batch_size, num_features, num_frames)
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Main function to train the model
def train_cnn_on_bigdata(data_file, classification_file, segment_info_file, log_file_path):
    # Load the dataset
    dataset = IMUBigDataset(data_file, classification_file)
    fold_sets = create_training_validation_sets(data_file, classification_file, segment_info_file, log_file_path)

    fold = 1
    for train_segments, train_labels, val_segments, val_labels in fold_sets:
        log_message(f"\n--- Training for fold {fold} ---", log_file_path)

        # Create Subsets based on segment indices
        train_dataset = Subset(dataset, train_segments)
        val_dataset = Subset(dataset, val_segments)

        # Adjusting class weights to emphasize specific classes
        class_counts = np.bincount(train_labels)
        weight_for_class_0 = max(class_counts) / class_counts[0] * 3.0
        weight_for_class_1 = 1.0
        class_weights = torch.tensor([weight_for_class_0, weight_for_class_1], dtype=torch.float)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

        # Model initialization
        model = IMUCNN(input_size=24, num_classes=2)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_batches = len(train_loader)
            log_message(f"\n--- Epoch {epoch+1}/{num_epochs} started for fold {fold} ---", log_file_path)

            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            log_message(f"Resource Utilization at Epoch Start: CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%", log_file_path)

            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                    log_message(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{total_batches}, "
                                f"Loss: {running_loss/(i+1):.4f}", log_file_path)

            avg_loss = running_loss / total_batches
            log_message(f"--- Fold {fold}, Epoch {epoch+1}/{num_epochs} completed with Average Loss: {avg_loss:.4f} ---", log_file_path)

            scheduler.step(avg_loss)

        # Evaluation
        model.eval()
        all_labels, all_predictions = [], []
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(val_loader):
                outputs = model(inputs)
                probabilities = nn.functional.softmax(outputs, dim=1)
                predicted = (probabilities[:, 1] > 0.3).long()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                for i in range(len(labels)):
                    segment_id = val_segments[idx * val_loader.batch_size + i]
                    actual_label = labels[i].item()
                    predicted_label = predicted[i].item()
                    result = "Correct" if actual_label == predicted_label else "Incorrect"
                    log_message(f"Segment ID: {segment_id}, Actual: {actual_label}, Predicted: {predicted_label}, Result: {result}", log_file_path)

        # Calculating metrics
        accuracy = (np.array(all_labels) == np.array(all_predictions)).mean()
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        cm = confusion_matrix(all_labels, all_predictions)

        # Logging results
        log_message("\n" + "-" * 50, log_file_path)
        log_message(f"--- Fold {fold} results ---", log_file_path)
        log_message(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}", log_file_path)
        log_message(f"Confusion Matrix:\n{cm}", log_file_path)

        fold += 1

    log_message("--- Cross-validation completed ---", log_file_path)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    data_file = "C:\\GitRepositories\\Master_Thesis\\Data_IMU_Segmented_Combined\\IMU_Segmented_BigDataset.npy"
    classification_file = "C:\\GitRepositories\\Master_Thesis\\FM_QualityClassification\\FM_QualityClassification_Binary_BigData.csv"
    segment_info_file = "C:\\GitRepositories\\Master_Thesis\\FM_QualityClassification\\IMU_Segmented_Group_Info.csv"
    log_file_path = os.path.join(LOGS_FOLDER, 'CNN_BigData_Log.txt')

    train_cnn_on_bigdata(data_file, classification_file, segment_info_file, log_file_path)
