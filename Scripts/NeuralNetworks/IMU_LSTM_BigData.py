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

# Definition of the dataset class
class IMUBigDataset(Dataset):
    def __init__(self, data_file, classification_file):
        self.data = np.load(data_file)  # Load the entire dataset as a numpy array
        df_classification = pd.read_csv(classification_file)
        
        # Read classifications and determine lengths from the data shape
        self.labels = torch.tensor(df_classification['Classification'].values, dtype=torch.long)
        self.lengths = [100] * len(self.labels)  # Each sample in the big dataset has 100 timesteps (fixed length)

        # Sanity check
        if len(self.labels) != len(self.data):
            raise ValueError("Mismatch between the number of labels and the number of data samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Definition of the LSTM model
class IMULSTM(nn.Module):
    def __init__(self, input_size=24, hidden_size=256, num_layers=3, num_classes=2, dropout=0.75):
        super(IMULSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        hn = self.dropout(hn[-1])
        out = self.fc(hn)
        return out

# Function to create training and validation sets and log details
def create_training_validation_sets(data_file, classification_file, segment_info_file, log_file_path):
    # Load data and segment information
    log_message("Loading dataset...", log_file_path)
    data = np.load(data_file)
    df_classification = pd.read_csv(classification_file)
    df_segment_info = pd.read_csv(segment_info_file)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = df_segment_info['Classification'].values
    unique_ids = df_segment_info['Child ID'].unique()

    fold_info = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(unique_ids, labels), 1):
        log_message(f"\nCreating training and validation sets for fold {fold}...", log_file_path)
        
        # Select IDs for training and validation
        train_ids = unique_ids[train_idx]
        test_ids = unique_ids[test_idx]

        # Prepare train and validation sets
        train_indices = df_segment_info[df_segment_info['Child ID'].isin(train_ids)].index
        val_indices = df_segment_info[df_segment_info['Child ID'].isin(test_ids)].index

        fold_info.append({
            'fold': fold,
            'train_ids': train_ids,
            'val_ids': test_ids,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'train_data': data[train_indices],
            'val_data': data[val_indices],
            'train_labels': df_classification.loc[train_indices, 'Classification'].values,
            'val_labels': df_classification.loc[val_indices, 'Classification'].values,
        })

        log_message(f"Training set for fold {fold} includes {len(train_indices)} segments.", log_file_path)
        log_message(f"Validation set for fold {fold} includes {len(val_indices)} segments.", log_file_path)

        # Detailed segment information
        log_message("Details of Training Segments:", log_file_path)
        for idx in train_indices:
            child_id = df_segment_info.at[idx, 'Child ID']
            segment_range = (df_segment_info.at[idx, 'Start Row'], df_segment_info.at[idx, 'End Row'])
            classification = df_segment_info.at[idx, 'Classification']
            log_message(f"Child ID: {child_id}, Segment Range: {segment_range}, Classification: {classification}", log_file_path)
        
        log_message("Details of Validation Segments:", log_file_path)
        for idx in val_indices:
            child_id = df_segment_info.at[idx, 'Child ID']
            segment_range = (df_segment_info.at[idx, 'Start Row'], df_segment_info.at[idx, 'End Row'])
            classification = df_segment_info.at[idx, 'Classification']
            log_message(f"Child ID: {child_id}, Segment Range: {segment_range}, Classification: {classification}", log_file_path)

    return fold_info

# Main function to train the model
def train_lstm_on_bigdata(data_file, classification_file, segment_info_file, log_file_path):
    # Log model configuration
    log_message("Model Configuration:", log_file_path)
    log_message(f"Input Size: {24}, Hidden Size: {256}, Num Layers: {3}, Dropout: {0.4}", log_file_path)
    log_message(f"Batch Size: {256}, Learning Rate: {0.001}, Optimizer: AdamW", log_file_path)
    log_message("-" * 50, log_file_path)
    
    # Load and prepare training and validation sets
    fold_info = create_training_validation_sets(data_file, classification_file, segment_info_file, log_file_path)

    # Cross-validation training
    for fold_data in fold_info:
        fold = fold_data['fold']
        log_message(f"Training for fold {fold}...", log_file_path)

        # Prepare DataLoaders with optimized parameters
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(fold_data['train_data'], dtype=torch.float32),
            torch.tensor(fold_data['train_labels'], dtype=torch.long)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(fold_data['val_data'], dtype=torch.float32),
            torch.tensor(fold_data['val_labels'], dtype=torch.long)
        )
        
        # Prepare DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=12, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=12, persistent_workers=True)

        # Model initialization
        model = IMULSTM(input_size=24, hidden_size=256, num_layers=3, num_classes=2, dropout=0.6)
        
        # Use weighted loss function
        class_weights = torch.tensor([2.0, 1.0], dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        # Training model
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_batches = len(train_loader)
            log_message(f"Epoch {epoch+1}/{num_epochs} started for fold {fold}.", log_file_path)

            # Log resource utilization
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            log_message(f"Resource Utilization at Epoch Start: CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%", log_file_path)

            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs, torch.tensor([inputs.size(1)] * inputs.size(0)))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Log every batch progress
                if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                    log_message(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{total_batches}, "
                                f"Loss: {running_loss/(i+1):.4f}", log_file_path)
            
            # Log the average loss for the epoch
            avg_loss = running_loss / total_batches
            log_message(f"Fold {fold}, Epoch {epoch+1}/{num_epochs} completed with Average Loss: {avg_loss:.4f}", log_file_path)

            # Update scheduler
            scheduler.step(avg_loss)

        # Model evaluation
        model.eval()
        all_labels, all_predictions = [], []
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(test_loader):
                outputs = model(inputs, torch.tensor([inputs.size(1)] * inputs.size(0)))
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                # Log detailed evaluation with Segment ID, Actual, and Predicted values
                for i in range(len(labels)):
                    segment_id = fold_data['val_indices'][idx * test_loader.batch_size + i]
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
        log_message(f"Fold {fold} results:", log_file_path)
        log_message(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}", log_file_path)
        log_message(f"Confusion Matrix:\n{cm}", log_file_path)

    log_message("Cross-validation completed.", log_file_path)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    # Paths to data
    data_file = "C:\\GitRepositories\\Master_Thesis\\Data_IMU_Segmented_Combined\\IMU_Segmented_BigDataset.npy"
    classification_file = "C:\\GitRepositories\\Master_Thesis\\FM_QualityClassification\\FM_QualityClassification_Binary_BigData.csv"
    segment_info_file = "C:\\GitRepositories\\Master_Thesis\\FM_QualityClassification\\IMU_Segmented_Group_Info.csv"
    log_file_path = os.path.join(LOGS_FOLDER, 'LSTM_BigData_Log.txt')

    # Start training
    train_lstm_on_bigdata(data_file, classification_file, segment_info_file, log_file_path)
