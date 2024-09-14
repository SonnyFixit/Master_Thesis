import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Define confusion matrices for each fold (updated LSTM-CNN results)
confusion_matrices_lstm_cnn = {
    "Fold 1": [[1, 660], [59, 5268]],
    "Fold 2": [[27, 330], [310, 6080]],
    "Fold 3": [[50, 390], [210, 4630]],
    "Fold 4": [[3, 330], [210, 5310]],
    "Fold 5": [[55, 750], [230, 5585]]
}

# Metrics for each fold (updated LSTM-CNN results)
metrics_lstm_cnn = {
    "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
    "Accuracy": [0.8835, 0.9054, 0.8893, 0.9085, 0.8563],
    "Precision": [0.8000, 0.9050, 0.8600, 0.8950, 0.8005],
    "Recall": [0.8835, 0.9054, 0.8893, 0.9085, 0.8563],
    "F1 Score": [0.8397, 0.9052, 0.8744, 0.9017, 0.8274]
}

# Output directory for saving images
output_dir = os.path.join(os.path.expanduser("~"), 'Desktop', 'Visualization_Results_LSTM_CNN')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created or exists already: {os.path.abspath(output_dir)}")

# Define a color palette similar to the provided image
color_palette = sns.color_palette("Greens", as_cmap=True)

# Function to plot confusion matrix with specific labels and values for updated LSTM-CNN results
def plot_confusion_matrix(matrix, fold_name, show_colorbar=False):
    # Create a new matrix with labels and values
    labels = np.array([['True Negative (TN)\n' + str(matrix[0][0]), 'False Positive (FP)\n' + str(matrix[0][1])],
                       ['False Negative (FN)\n' + str(matrix[1][0]), 'True Positive (TP)\n' + str(matrix[1][1])]])

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=labels, fmt='', cmap=color_palette, cbar=show_colorbar, annot_kws={"size": 12})
    plt.title(f'Confusion Matrix - {fold_name} (LSTM-CNN)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0.5, 1.5], ['Predicted Negative (0)', 'Predicted Positive (1)'])
    plt.yticks([0.5, 1.5], ['Actual Negative (0)', 'Actual Positive (1)'])
    plt.tight_layout()
    file_path = os.path.join(output_dir, f'Confusion_Matrix_{fold_name}_LSTM_CNN.png')
    plt.savefig(file_path)
    plt.close()
    print(f"Confusion matrix saved to: {os.path.abspath(file_path)}")

# Plot all confusion matrices for updated LSTM-CNN results
for fold, matrix in confusion_matrices_lstm_cnn.items():
    # Show colorbar only for Fold 5
    plot_confusion_matrix(matrix, fold, show_colorbar=(fold == "Fold 5"))

# Create a summary table with metrics for updated LSTM-CNN results
metrics_df = pd.DataFrame(metrics_lstm_cnn)

# Display the table with colored headers
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')

# Add color to header and index columns
table.auto_set_font_size(False)
table.set_fontsize(10)
table[0, 0].set_facecolor('#A9A9A9')  # Darker gray for headers
for key, cell in table.get_celld().items():
    if key[0] == 0 or key[1] == 0:
        cell.set_facecolor('#A9A9A9')  # Slightly darker gray color for headers

table_file_path = os.path.join(output_dir, 'Cross_Validation_Metrics_Summary_LSTM_CNN.png')
plt.savefig(table_file_path)
plt.show()
print(f"Metrics summary table saved to: {os.path.abspath(table_file_path)}")

# Save the metrics to a CSV file for updated LSTM-CNN results
csv_file_path = os.path.join(output_dir, 'Cross_Validation_Metrics_Summary_LSTM_CNN.csv')
metrics_df.to_csv(csv_file_path, index=False)
print(f"Metrics saved to CSV file: {os.path.abspath(csv_file_path)}")
