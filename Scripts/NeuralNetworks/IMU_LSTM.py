import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm
from datetime import datetime

# Adding path to the configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MasterThesis_Config import NN_COMBINED_DATA_FOLDER, NN_CLASSIFICATION_FILE, LOGS_FOLDER

# Set debug mode to control debug logs
debug_mode = False  # Set this to True to enable debug logs

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

# Adjusted collate function to handle correct data shape
def collate_fn(batch):
    data, labels, lengths, child_ids = zip(*batch)
    data = [x.view(-1, 24) for x in data]  # Flatten the (100, 24) into (100*sequence_length, 24)
    data_padded = pad_sequence(data, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    if debug_mode:
        print(f"[DEBUG] Batch data padded shape: {data_padded.shape}, lengths: {lengths}")
    return data_padded, labels, lengths, child_ids

# Definition of the LSTM model with multiple layers and dropout
class IMULSTM(nn.Module):
    def __init__(self, input_size=24, hidden_size=256, num_layers=3, num_classes=2, dropout=0.75):
        super(IMULSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)  # Fully connected layer for output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        if debug_mode:
            print(f"[DEBUG] Input shape: {x.shape}, lengths: {lengths}")
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        hn = self.dropout(hn[-1])  # Apply dropout to the last hidden state
        if debug_mode:
            print(f"[DEBUG] LSTM output shape: {hn.shape}")
        out = self.fc(hn)
        if debug_mode:
            print(f"[DEBUG] Output after fully connected layer shape: {out.shape}")
        return out

# Paths to data and classification file
combined_data_folder = NN_COMBINED_DATA_FOLDER
classification_file = NN_CLASSIFICATION_FILE

# Path to the log file
log_file_path = os.path.join(LOGS_FOLDER, 'LSTM_Test_Log.txt')

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
    
    model = IMULSTM(input_size=24, hidden_size=256, num_layers=3, num_classes=2, dropout=0.4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adjust weight_decay if needed
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Model training without Early Stopping
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        for i, (inputs, labels, lengths, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = model(inputs, lengths)
                loss = criterion(outputs, labels)
            except RuntimeError as e:
                print(f"[ERROR] Error during forward pass: {e}")
                print(f"[ERROR] Input shape: {inputs.shape}, lengths: {lengths}")
                continue
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update every 10 batches
            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                progress = (i + 1) / total_batches * 100
                print(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{total_batches} ({progress:.2f}%), Loss: {running_loss/(i+1):.4f}")
        
        # Step scheduler
        val_loss = running_loss / len(train_loader)
        scheduler.step(val_loss)
    
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
