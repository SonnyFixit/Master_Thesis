import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define confusion matrices for each fold based on the provided screenshots
confusion_matrices = {
    "Fold 1": [[4, 7], [14, 97]],
    "Fold 2": [[3, 10], [9, 82]],
    "Fold 3": [[1, 5], [28, 87]],
    "Fold 4": [[4, 5], [18, 97]],
    "Fold 5": [[2, 11], [10, 101]]
}

# Output directory for saving images
output_dir = os.path.join(os.path.expanduser("~"), 'Desktop', 'Visualization_Results')
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created or already exists: {os.path.abspath(output_dir)}")

# Define a consistent color palette for the red color scheme
color_palette = sns.light_palette("red", as_cmap=True)

# Function to plot confusion matrix with specific labels and values
def plot_confusion_matrix(matrix, fold_name, show_colorbar=False):
    # Create a new matrix with labels and values
    labels = np.array([['True Negative (TN)\n' + str(matrix[0][0]), 'False Positive (FP)\n' + str(matrix[0][1])],
                       ['False Negative (FN)\n' + str(matrix[1][0]), 'True Positive (TP)\n' + str(matrix[1][1])]])

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=labels, fmt='', cmap=color_palette, cbar=show_colorbar, annot_kws={"size": 12})
    plt.title(f'Confusion Matrix - {fold_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0.5, 1.5], ['Predicted Negative (0)', 'Predicted Positive (1)'])
    plt.yticks([0.5, 1.5], ['Actual Negative (0)', 'Actual Positive (1)'])
    plt.tight_layout()

    # Save the confusion matrix plot
    file_path = os.path.join(output_dir, f'Confusion_Matrix_{fold_name}.png')
    plt.savefig(file_path)
    plt.close()
    print(f"Confusion matrix saved to: {os.path.abspath(file_path)}")

# Plot all confusion matrices
for fold, matrix in confusion_matrices.items():
    # Show colorbar only for Fold 5
    plot_confusion_matrix(matrix, fold, show_colorbar=(fold == "Fold 5"))
