import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Definicja klasy dataset dla IMU
class IMUDataset(Dataset):
    def __init__(self, data_folder, classification_file):
        self.data_folder = data_folder
        df_classification = pd.read_csv(classification_file)
        
        self.data = []
        self.labels = []
        self.lengths = []  # Lista przechowująca długości sekwencji dla każdego przykładu
        
        for index, row in df_classification.iterrows():
            child_id = row['ID']
            label = row['FM Quality Numeric']
            
            # Spróbuj znaleźć odpowiedni plik IMU
            file_found = False
            for file_name in os.listdir(data_folder):
                if file_name.startswith(f"IMU_CombineSegmentedData_{int(child_id):03d}.npy"):
                    imu_file_path = os.path.join(data_folder, file_name)
                    imu_data = np.load(imu_file_path)
                    self.data.append(torch.tensor(imu_data, dtype=torch.float32))
                    self.labels.append(label)
                    self.lengths.append(imu_data.shape[0])  # Zapisz długość sekwencji
                    file_found = True
                    break
            
            if not file_found:
                print(f"Warning: Nie znaleziono pliku dla ID {child_id}")
        
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.lengths[idx]

# Funkcja do padania sekwencji w batchu
def collate_fn(batch):
    data, labels, lengths = zip(*batch)
    data_padded = pad_sequence(data, batch_first=True)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return data_padded, labels, lengths

# Definicja modelu CNN
class IMUCNN(nn.Module):
    def __init__(self):
        super(IMUCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=24, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        
        # Początkowo ustawiamy fc1 na None, zainicjalizujemy później
        self.fc1 = None
        self.fc2 = nn.Linear(128, 2)  # Zakładając binarną klasyfikację (0 lub 1)

    def forward(self, x, lengths):
        batch_size, X, num_frames, num_features = x.shape
        
        # Przekształcenie danych na (batch_size * X, 24, 100)
        x = x.view(-1, num_frames, num_features).permute(0, 2, 1)
        
        # Przekazanie przez pierwszą warstwę konwolucyjną i pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Spłaszczenie
        x = x.view(x.size(0), -1)
        
        # Dynamically set the input size for fc1 based on the current shape of x
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128)
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Ścieżki do danych i pliku z klasyfikacją
combined_data_folder = r'C:\GitRepositories\Master_Thesis\Data_IMU_Segmented_Combined'
classification_file = r'C:\GitRepositories\Master_Thesis\FM_QualityClassification_Binary.csv'

# Ładowanie datasetu
print("Ładowanie danych...")
dataset = IMUDataset(combined_data_folder, classification_file)
print(f"Załadowano {len(dataset)} próbek.")

# Podział na zbiór treningowy i testowy
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

print(f"Zbiór treningowy: {len(train_dataset)} próbek, Zbiór testowy: {len(test_dataset)} próbek.")

# Inicjalizacja modelu
model = IMUCNN()
print("Model CNN został utworzony.")

# Definicja kryterium oraz optymalizatora
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening modelu
num_epochs = 20
print("Rozpoczęcie treningu...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels, lengths) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Przód
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        
        # Tył
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

print("Trening zakończony.")

# Ewaluacja modelu
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels, lengths in test_loader:
        outputs = model(inputs, lengths)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test set: {100 * correct / total:.2f} %')
