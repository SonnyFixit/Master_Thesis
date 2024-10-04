import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

"""
IMU Confusion Matrix and Performance Metrics Visualization

This script generates confusion matrix heatmaps and performance metrics visualizations for a classification model, 
specifically for multiple folds of cross-validation. The confusion matrices represent the predicted and true labels 
for a binary classification task, and the performance metrics (Accuracy, Precision, Recall, F1 Score) are displayed 
in a table format.

Key Features:
- Generates labeled heatmaps for confusion matrices with custom color palettes.
- Saves each confusion matrix as a PNG file in a specified output directory.
- Creates a table of performance metrics for each fold, saving it both as an image and a CSV file for further analysis.

Dependencies:
- **Matplotlib**: For creating and saving visualizations.
- **Seaborn**: For enhanced styling and heatmap plotting.
- **Pandas**: For handling performance metrics in tabular format.
- **Numpy**: For numerical operations (in this case, constructing confusion matrix labels).

Parameters:
- `confusion_matrices_cnn`: Dictionary containing confusion matrices for each fold of a cross-validation experiment.
- `metrics_cnn`: Dictionary of performance metrics (Accuracy, Precision, Recall, F1 Score) for each fold.
- `output_dir`: Path to the directory where the visualizations will be saved. Defaults to a folder on the Desktop.

Example Usage:
    Run the script to generate confusion matrix heatmaps and metrics summaries for CNN model performance.

"""

# Define confusion matrices for each fold (these are examples; you can replace them with any confusion matrices)
confusion_matrices_cnn = {
    "Fold 1": [[839, 0], [953, 7900]],
    "Fold 2": [[359, 0], [301, 7699]],
    "Fold 3": [[775, 0], [579, 8393]],
    "Fold 4": [[705, 0], [220, 7085]],
    "Fold 5": [[501, 0], [483, 7554]]
}

# Example performance metrics for each fold (replace these with your own metrics)
metrics_cnn = {
    "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
    "Accuracy": [0.9017, 0.9640, 0.9406, 0.9725, 0.9434],
    "Precision": [0.9540, 0.9804, 0.9660, 0.9791, 0.9712],
    "Recall": [0.9017, 0.9640, 0.9406, 0.9725, 0.9434],
    "F1 Score": [0.9167, 0.9690, 0.9477, 0.9742, 0.9518]
}

# Specify output directory for saving visualizations; the desktop is used here as an example.
# You can change the directory to any path of your choice.
output_dir = os.path.join(os.path.expanduser("~"), 'Desktop', 'Confusion_Matrix_and_Metrics')

# Create the output directory if it doesn't already exist
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created or exists already: {os.path.abspath(output_dir)}")

# Custom color palette based on a red color, modify this to match the color scheme of your choice
custom_color = sns.light_palette("#ff0000", as_cmap=True)

# Generalized function to plot and save confusion matrix heatmap with labels for TP, TN, FP, FN
def plot_confusion_matrix(matrix, fold_name, show_colorbar=False):
    """
    Plots a confusion matrix heatmap with customized labels and saves it as an image.

    Parameters:
    - matrix: A 2x2 confusion matrix (list or array).
    - fold_name: A string representing the name of the fold (e.g., 'Fold 1').
    - show_colorbar: A boolean to show or hide the color bar (optional).
    """
    # Define text labels with values for each cell in the matrix
    labels = np.array([['True Negative (TN)\n' + str(matrix[0][0]), 'False Positive (FP)\n' + str(matrix[0][1])],
                       ['False Negative (FN)\n' + str(matrix[1][0]), 'True Positive (TP)\n' + str(matrix[1][1])]])

    # Create the confusion matrix heatmap with labeled values
    plt.figure(figsize=(6, 4))  # Adjust figure size as needed
    sns.heatmap(matrix, annot=labels, fmt='', cmap=custom_color, cbar=show_colorbar, annot_kws={"size": 12})
    
    # Add title and axis labels to the heatmap
    plt.title(f'Confusion Matrix - {fold_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Customize axis tick labels (for binary classification)
    plt.xticks([0.5, 1.5], ['Predicted Negative (0)', 'Predicted Positive (1)'])
    plt.yticks([0.5, 1.5], ['Actual Negative (0)', 'Actual Positive (1)'])
    
    # Ensure tight layout for proper rendering
    plt.tight_layout()

    # Save the generated heatmap as an image file
    file_path = os.path.join(output_dir, f'Confusion_Matrix_{fold_name}.png')
    plt.savefig(file_path)
    plt.close()  # Close the figure to free up memory
    print(f"Confusion matrix saved to: {os.path.abspath(file_path)}")

# Iterate over each fold and plot/save confusion matrices
for fold, matrix in confusion_matrices_cnn.items():
    # Show colorbar only for a specific fold if desired (e.g., last fold); adjust as necessary
    plot_confusion_matrix(matrix, fold, show_colorbar=(fold == "Fold 5"))

# Create a summary table for performance metrics
metrics_df = pd.DataFrame(metrics_cnn)

# Plot the metrics as a table and save it as an image
fig, ax = plt.subplots(figsize=(10, 4))  # Adjust figure size as needed
ax.axis('tight')  # Remove chart borders for cleaner table
ax.axis('off')  # Hide axes for table display

# Create a table with the metrics DataFrame and add headers
table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')

# Adjust font size and styling for table
Visualization_1_Sec_IMU_Augmented_CNNtable.auto_set_font_size(False)
table.set_fontsize(10)

# Apply color styling to header cells (optional, for visual appeal)
header_color = '#A9A9A9'  # Custom header color
for key, cell in table.get_celld().items():
    if key[0] == 0:  # Header row
        cell.set_facecolor(header_color)  # Apply header color

# Save the metrics table as an image file
table_file_path = os.path.join(output_dir, 'Metrics_Summary.png')
plt.savefig(table_file_path)
plt.show()  # Display the table
print(f"Metrics summary table saved to: {os.path.abspath(table_file_path)}")

# Save the metrics data to a CSV file for further analysis or reporting
csv_file_path = os.path.join(output_dir, 'Metrics_Summary.csv')
metrics_df.to_csv(csv_file_path, index=False)
print(f"Metrics saved to CSV file: {os.path.abspath(csv_file_path)}")
