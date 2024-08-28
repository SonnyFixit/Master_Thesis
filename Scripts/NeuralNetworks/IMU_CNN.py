import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm
from datetime import datetime

# Adding path to the configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MasterThesis_Config import NN_COMBINED_DATA_FOLDER, NN_CLASSIFICATION_FILE, LOGS_FOLDER

# Definition of the IMU dataset class
class IMUDataset(Dataset):
    def __init__(self, data_folder, classification_file):
        self.data_folder = data_folder
        df_classification = pd.read_csv(classification_file)
        
        self.data = []
        self.labels = []
        self.lengths = []  # List storing the sequence lengths for each example
        self.child_ids = []  # List storing child IDs
        
        for index, row in df_classification.iterrows():
            child_id = row['ID']
            label = row['FM Quality Numeric']
            
            # Try to find the corresponding IMU file
            file_found = False
            for file_name in os.listdir(data_folder):
                if file_name.startswith(f"IMU_CombineSegmentedData_{int(child_id):03d}.npy"):
                    imu_file_path = os.path.join(data_folder, file_name)
                    imu_data = np.load(imu_file_path)
                    self.data.append(torch.tensor(imu_data, dtype=torch.float32))
                    self.labels.append(label)
                    self.lengths.append(imu_data.shape[0])  # Save sequence length
                    self.child_ids.append(child_id)  # Save child ID
                    file_found = True
                    break
            
            if not file_found:
                print(f"Warning: No file found for ID {child_id}")
        
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.lengths[idx], self.child_ids[idx]

# Function to pad sequences in a batch
def collate_fn(batch):
    data, labels, lengths, child_ids = zip(*batch)
    data_padded = pad_sequence(data, batch_first=True)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return data_padded, labels, lengths, child_ids

# Definition of the CNN model with additional convolutional layers and dropout
class IMUCNN(nn.Module):
    def __init__(self):
        super(IMUCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=24, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)  # New convolutional layer
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer
        self.fc1 = None  # To be initialized dynamically later
        self.fc2 = nn.Linear(128, 2)  # Assuming binary classification (0 or 1)

    def forward(self, x, lengths):
        batch_size, X, num_frames, num_features = x.shape
        
        # Reshape: (batch_size, X, num_frames, num_features) -> (batch_size * X, num_features, num_frames)
        x = x.view(batch_size * X, num_features, num_frames)
        
        # Pass through convolutional layers and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))  # New convolutional layer
        
        # Reshape data to (batch_size, X, -1) and merge samples into batch_size
        x = x.view(batch_size, X, -1).mean(dim=1)
        
        # Dynamic initialization of fc1 based on data size
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128)  # Automatic size adjustment
        
        # Dropout before the fully connected layer
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Paths to data and classification file
combined_data_folder = NN_COMBINED_DATA_FOLDER
classification_file = NN_CLASSIFICATION_FILE

# Path to the log file
log_file_path = os.path.join(LOGS_FOLDER, 'CNN_Test_Log.txt')

# Function to save results to the log
def save_log(content, file_path):
    with open(file_path, 'a') as log_file:
        log_file.write(content + '\n')

# Loading the dataset
print("Loading data...")
dataset = IMUDataset(combined_data_folder, classification_file)
print(f"Loaded {len(dataset)} samples.")

# Stratified 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_index, test_index in skf.split(np.zeros(len(dataset)), dataset.labels):
    print(f"Training for fold {fold}...")
    
    train_dataset = Subset(dataset, train_index)
    test_dataset = Subset(dataset, test_index)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    model = IMUCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Model training
    num_epochs = 50  # Increased number of epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        for i, (inputs, labels, lengths, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update every 10 batches
            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                progress = (i + 1) / total_batches * 100
                print(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{total_batches} ({progress:.2f}%), Loss: {running_loss/(i+1):.4f}")
    
    print(f'Fold {fold}, Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

    # Model evaluation
    model.eval()
    all_labels = []
    all_predictions = []
    all_child_ids = []

    with torch.no_grad():
        for inputs, labels, lengths, child_ids in test_loader:
            outputs = model(inputs, lengths)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_child_ids.extend(child_ids)

    # Calculating metrics
    accuracy = (np.array(all_labels) == np.array(all_predictions)).mean()
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    cm = confusion_matrix(all_labels, all_predictions)

    # Logging results with additional empty lines for spacing
    save_log("\n\n\n", log_file_path)  # Adding extra spacing before each fold log
    log_header = f"Test Log - Fold {fold} - Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_header += "-" * 50
    save_log(log_header, log_file_path)

    log_metrics = f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\nConfusion Matrix:\n{cm}\n"
    save_log(log_metrics, log_file_path)
    
    for idx, child_id in enumerate(all_child_ids):
        actual_label = all_labels[idx]
        predicted_label = all_predictions[idx]
        result = "Correct" if actual_label == predicted_label else "Incorrect"
        log_entry = f"Child ID: {child_id}, Actual: {actual_label}, Predicted: {predicted_label}, Result: {result}"
        save_log(log_entry, log_file_path)

    fold += 1

print("Cross-validation completed.")
