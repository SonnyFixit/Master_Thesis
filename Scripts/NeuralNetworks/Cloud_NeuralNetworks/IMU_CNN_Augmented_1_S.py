import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from datetime import datetime
from google.colab import drive

# Montowanie Google Drive
drive.mount('/content/drive')

# Funkcja do logowania wiadomości
def log_message(message, log_file_path):
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')
    print(message)

# Funkcja do tworzenia zbiorów treningowych i walidacyjnych z augmentacją
def create_training_validation_sets_with_augmentation(data_file, classification_file, segment_info_file, log_file_path):
    log_message("Loading dataset for training/validation set creation with augmentation...", log_file_path)
    
    data = np.load(data_file)
    max_index = data.shape[0]
    
    # Read classification and segment info files
    df_classification = pd.read_csv(classification_file, index_col=0)
    df_segment_info = pd.read_csv(segment_info_file)

    # Strip spaces from columns just in case
    df_segment_info.columns = df_segment_info.columns.str.strip()

    # Handle potential 'Segment ID' column issues
    if 'Segment ID' in df_segment_info.columns:
        df_segment_info.rename(columns={'Segment ID': 'Segment_ID'}, inplace=True)
    else:
        log_message("Error: 'Segment ID' column not found in segment info file.", log_file_path)
        return

    # Log general statistics about the dataset
    num_fm_plus = df_segment_info[df_segment_info['Classification'] == 1]['Child ID'].nunique()
    num_fm_minus = df_segment_info[df_segment_info['Classification'] == 0]['Child ID'].nunique()
    num_augmented_samples = df_classification[df_classification['Augmented'] == 1].shape[0]

    log_message(f"Total number of FM+ children: {num_fm_plus}", log_file_path)
    log_message(f"Total number of FM- children: {num_fm_minus}", log_file_path)
    log_message(f"Total number of augmented samples: {num_augmented_samples}", log_file_path)

    # Proceed with creating the training and validation sets
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    unique_ids = df_segment_info['Child ID'].unique()
    labels = df_segment_info.groupby('Child ID')['Classification'].first().values

    fold_sets = []
    fold = 1
    for train_idx, test_idx in skf.split(unique_ids, labels):
        log_message(f"\n--- Creating training and validation sets for fold {fold} ---", log_file_path)

        train_ids = unique_ids[train_idx]
        test_ids = unique_ids[test_idx]

        # Exclude augmented children from the validation set
        test_ids = [child_id for child_id in test_ids if child_id < 104]  # Assuming augmented children have IDs >= 104

        log_message(f"Number of children in training set for fold {fold}: {len(train_ids)}", log_file_path)
        log_message(f"Training IDs for fold {fold}: {list(train_ids)}", log_file_path)  # Log training IDs
        
        log_message(f"Number of children in validation set for fold {fold} (without augmented): {len(test_ids)}", log_file_path)
        log_message(f"Validation IDs for fold {fold}: {list(test_ids)}", log_file_path)  # Log validation IDs

        train_segments = []
        train_labels = []
        val_segments = []
        val_labels = []

        # Create training set
        for child_id in train_ids:
            child_data = df_segment_info[df_segment_info['Child ID'] == child_id]
            valid_segments = child_data['Segment_ID'].values - 1  
            valid_segments = [seg for seg in valid_segments if 0 <= seg < max_index and seg in df_classification.index]

            if not valid_segments:
                log_message(f"No valid segments found for Child ID {child_id}. Skipping...", log_file_path)
                continue

            augmented_segments = df_classification.loc[valid_segments, 'Augmented'] == 1
            real_segments = df_classification.loc[valid_segments, 'Augmented'] == 0

            train_segments.extend(df_classification.loc[valid_segments][augmented_segments].index.tolist())
            train_labels.extend(df_classification.loc[valid_segments][augmented_segments]['Classification'].values)

            train_segments.extend(df_classification.loc[valid_segments][real_segments].index.tolist())
            train_labels.extend(df_classification.loc[valid_segments][real_segments]['Classification'].values)

        # Create validation set
        for child_id in test_ids:
            child_data = df_segment_info[df_segment_info['Child ID'] == child_id]
            valid_segments = child_data['Segment_ID'].values - 1  
            valid_segments = [seg for seg in valid_segments if 0 <= seg < max_index and seg in df_classification.index]

            if not valid_segments:
                log_message(f"No valid segments found for Child ID {child_id}. Skipping...", log_file_path)
                continue

            real_segments = df_classification.loc[valid_segments, 'Augmented'] == 0
            val_segments.extend(df_classification.loc[valid_segments][real_segments].index.tolist())
            val_labels.extend(df_classification.loc[valid_segments][real_segments]['Classification'].values)

        fold_sets.append((train_segments, train_labels, val_segments, val_labels))
        fold += 1

    log_message("--- Training and validation sets creation completed with augmentation ---", log_file_path)
    return fold_sets


# Klasa dataset
class IMUBigDataset(Dataset):
    def __init__(self, data_file, classification_file):
        self.data = np.load(data_file)
        df_classification = pd.read_csv(classification_file)
        self.labels = torch.tensor(df_classification['Classification'].values, dtype=torch.long)
        self.lengths = [100] * len(self.labels)  # Each sample in the big dataset has 100 timesteps (fixed length)
        if len(self.labels) != len(self.data):
            raise ValueError("Mismatch between the number of labels and the number of data samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Definicja modelu CNN
class IMUCNN(nn.Module):
    def __init__(self, input_size=24, num_classes=2):
        super(IMUCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.5)

        # Obliczenie rozmiaru po ostatniej warstwie konwolucyjnej i pooling
        self.flatten_size = self.calculate_flatten_size()

        # Warstwy w pełni połączone
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def calculate_flatten_size(self):
        # Symulacja przejścia przez warstwy, aby obliczyć rozmiar wyjściowy
        sample_input = torch.zeros(1, 24, 100)  # Próbka danych z tym samym wymiarem co dane wejściowe
        x = self.pool(torch.relu(self.conv1(sample_input)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        flatten_size = x.view(1, -1).size(1)  # Obliczenie rozmiaru wyjściowego po flatten
        return flatten_size

    def forward(self, x):
        x = x.transpose(1, 2)  # Zmiana wymiarów dla Conv1d
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten przed wejściem do warstwy w pełni połączonej
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Główna funkcja treningowa
def train_cnn_on_bigdata(data_file, classification_file, segment_info_file, log_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Sprawdzenie, czy GPU jest dostępne
    dataset = IMUBigDataset(data_file, classification_file)
    fold_sets = create_training_validation_sets_with_augmentation(data_file, classification_file, segment_info_file, log_file_path)

    fold = 1
    for train_segments, train_labels, val_segments, val_labels in fold_sets:
        log_message(f"\n--- Training for fold {fold} ---", log_file_path)

        # Tworzenie zbiorów treningowych i walidacyjnych
        train_dataset = Subset(dataset, train_segments)
        val_dataset = Subset(dataset, val_segments)

        # Ustawienia klas wag z większym naciskiem na negatywne próbki
        class_counts = np.bincount(train_labels)
        weight_for_class_0 = max(class_counts) / class_counts[0] * 2.0  # Zwiększona waga dla negatywnej klasy
        weight_for_class_1 = 1.0
        class_weights = torch.tensor([weight_for_class_0, weight_for_class_1], dtype=torch.float).to(device)

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2, persistent_workers=True)

        # Inicjalizacja modelu
        model = IMUCNN(input_size=24, num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        num_epochs = 25
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_batches = len(train_loader)
            log_message(f"\n--- Epoch {epoch+1}/{num_epochs} started for fold {fold} ---", log_file_path)

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
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

        # Ewaluacja
        model.eval()
        all_labels, all_predictions = [], []
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probabilities = nn.functional.softmax(outputs, dim=1)
                predicted = (probabilities[:, 1] > 0.5).long()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                for i in range(len(labels)):
                    segment_id = val_segments[idx * val_loader.batch_size + i]
                    actual_label = labels[i].item()
                    predicted_label = predicted[i].item()
                    result = "Correct" if actual_label == predicted_label else "Incorrect"
                    log_message(f"Segment ID: {segment_id}, Actual: {actual_label}, Predicted: {predicted_label}, Result: {result}", log_file_path)

        # Obliczanie metryk
        accuracy = (np.array(all_labels) == np.array(all_predictions)).mean()
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        cm = confusion_matrix(all_labels, all_predictions)

        # Logowanie wyników
        log_message("\n" + "-" * 50, log_file_path)
        log_message(f"--- Fold {fold} results ---", log_file_path)
        log_message(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}", log_file_path)
        log_message(f"Confusion Matrix:\n{cm}", log_file_path)

        fold += 1

    log_message("--- Cross-validation completed ---", log_file_path)

if __name__ == "__main__":
    # Ścieżki do plików na Google Drive
    data_file = "/content/drive/My Drive/IMU_Combine_1_Second_Augmented_Dataset.npy"
    classification_file = "/content/drive/My Drive/FM_QualityClassification_Binary_BigData_Augmented.csv"
    segment_info_file = "/content/drive/My Drive/IMU_Segmented_Group_Info_Augmented.csv"
    log_file_path = "/content/drive/My Drive/CNN_BigData_Augmented_Log.txt"

    train_cnn_on_bigdata(data_file, classification_file, segment_info_file, log_file_path)