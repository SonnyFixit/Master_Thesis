import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Define confusion matrices for each fold based on the provided screenshots
confusion_matrices_lstm = {
    "Fold 1": [[5, 6], [13, 99]],
    "Fold 2": [[4, 9], [8, 83]],
    "Fold 3": [[2, 4], [27, 89]],
    "Fold 4": [[5, 4], [17, 99]],
    "Fold 5": [[3, 11], [8, 104]]
}

# Metrics for each fold (example values - adjust if needed)
metrics_lstm = {
    "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
    "Accuracy": [0.828, 0.825, 0.735, 0.810, 0.860],
    "Precision": [0.923, 0.880, 0.958, 0.940, 0.892],
    "Recall": [0.865, 0.888, 0.732, 0.830, 0.910],
    "F1 Score": [0.894, 0.884, 0.830, 0.885, 0.900]
}

# Output directory for saving images
output_dir = r'C:\New_Output_Folder'  # Change to the desired directory

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define a consistent color palette (green, similar to the screenshot)
color_palette_lstm = sns.light_palette("green", as_cmap=True)

# Function to plot confusion matrix with specific labels and values
def plot_confusion_matrix_lstm(matrix, fold_name, show_colorbar=False):
    # Create a new matrix with labels and values
    labels = np.array([['True Negative (TN)\n' + str(matrix[0][0]), 'False Positive (FP)\n' + str(matrix[0][1])],
                       ['False Negative (FN)\n' + str(matrix[1][0]), 'True Positive (TP)\n' + str(matrix[1][1])]])

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=labels, fmt='', cmap=color_palette_lstm, cbar=show_colorbar, annot_kws={"size": 12})
    plt.title(f'Confusion Matrix - {fold_name} (LSTM)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0.5, 1.5], ['Predicted Negative (0)', 'Predicted Positive (1)'])
    plt.yticks([0.5, 1.5], ['Actual Negative (0)', 'Actual Positive (1)'])
    plt.tight_layout()

    # Save the confusion matrix plot
    file_path = os.path.join(output_dir, f'Confusion_Matrix_{fold_name}_LSTM.png')
    plt.savefig(file_path)
    print(f'Confusion matrix for {fold_name} saved at: {file_path}')  # Print full path
    plt.close()

# Plot all confusion matrices
for fold, matrix in confusion_matrices_lstm.items():
    # Show colorbar only for Fold 5
    plot_confusion_matrix_lstm(matrix, fold, show_colorbar=(fold == "Fold 5"))

# Create a summary table with metrics
metrics_df_lstm = pd.DataFrame(metrics_lstm)

# Display the table with colored headers
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=metrics_df_lstm.values, colLabels=metrics_df_lstm.columns, cellLoc='center', loc='center')

# Add color to header and index columns
table.auto_set_font_size(False)
table.set_fontsize(10)
table[0, 0].set_facecolor('#D3D3D3')
for key, cell in table.get_celld().items():
    if key[0] == 0 or key[1] == 0:
        cell.set_facecolor('#D3D3D3')  # Light gray color for headers

summary_image_path = os.path.join(output_dir, 'Cross_Validation_Metrics_Summary_LSTM.png')
plt.savefig(summary_image_path)
print(f'Summary metrics table saved at: {summary_image_path}')  # Print full path
plt.show()

# Save the metrics to a CSV file
csv_file_path = os.path.join(output_dir, 'Cross_Validation_Metrics_Summary_LSTM.csv')
metrics_df_lstm.to_csv(csv_file_path, index=False)
print(f'Summary metrics CSV saved at: {csv_file_path}')  # Print full path
