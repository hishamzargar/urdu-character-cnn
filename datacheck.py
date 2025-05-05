
import numpy as np
import matplotlib.pyplot as plt


# Define the path to your dataset
dataset_path = 'data/uhat_dataset.npz' 

urdu_dataset = np.load(dataset_path)
x_train, y_train = urdu_dataset['x_chars_train'], urdu_dataset['y_chars_train']
x_test, y_test = urdu_dataset['x_chars_test'], urdu_dataset['y_chars_test']


# Define the path to your dataset
dataset_path = 'data/uhat_dataset.npz' # Make sure this path is correct

print(f"Attempting to load dataset from: {dataset_path}")

try:
    # Load the dataset file
    urdu_dataset = np.load(dataset_path)

    # Check the available arrays in the file
    print(f"Arrays found in the file: {list(urdu_dataset.files)}")

    # --- Check Training Labels ---
    label_key_train = 'y_chars_train' # Assuming this is the correct key
    if label_key_train in urdu_dataset:
        y_train = urdu_dataset[label_key_train]
        print(f"\nAnalyzing '{label_key_train}' with shape: {y_train.shape}")

        # Find unique labels
        unique_train_labels = np.unique(y_train)
        # The number of unique labels is the number of classes
        num_classes_train = len(unique_train_labels) # or unique_train_labels.size

        print(f"Unique labels found in training set: {unique_train_labels}")
        print(f"Number of unique classes in training set: {num_classes_train}")

        # Sanity check: using max label (assumes labels are 0, 1, 2, ..., N-1)
        max_label_train = np.max(y_train)
        print(f"(Max label value found in training set: {max_label_train})")
        if max_label_train == num_classes_train - 1:
             print("(Max label check consistent with unique count)")
        else:
             print("(Warning: Max label check might indicate non-contiguous labels starting from 0)")

    else:
        print(f"\nKey '{label_key_train}' not found in the dataset.")
        num_classes_train = None

    # --- Optionally Check Test Labels Too ---
    label_key_test = 'y_chars_test' # Assuming this is the correct key
    if label_key_test in urdu_dataset:
        y_test = urdu_dataset[label_key_test]
        print(f"\nAnalyzing '{label_key_test}' with shape: {y_test.shape}")
        unique_test_labels = np.unique(y_test)
        num_classes_test = len(unique_test_labels)
        print(f"Unique labels found in test set: {unique_test_labels}")
        print(f"Number of unique classes in test set: {num_classes_test}")
    else:
        print(f"\nKey '{label_key_test}' not found in the dataset.")
        num_classes_test = None

    print("\n--- Final Result ---")
    if num_classes_train is not None:
         print(f"Number of classes based on '{label_key_train}': {num_classes_train}")
    elif num_classes_test is not None:
         print(f"Number of classes based on '{label_key_test}': {num_classes_test}")
    else:
         print("Could not determine number of classes from assumed keys.")


except FileNotFoundError:
    print(f"Error: Dataset file not found at {dataset_path}. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")


"""# --- Add this section after loading data ---
print("\n--- Visualizing Data Samples ---")

# y_train and y_test contain the integer labels directly from the file
y_train_labels = y_train # Use y_train directly
y_test_labels = y_test   # Use y_test directly

# Calculate validation split index
val_split_fraction = 0.2
split_index = int(len(x_train) * (1 - val_split_fraction))
x_val_samples = x_train[split_index:]
# Use the original y_train to get the validation labels before potential modification
y_val_labels = y_train[split_index:] # Get labels from the original y_train slice

def plot_samples(images, labels, title):
    plt.figure(figsize=(12, 12))
    num_samples = min(25, len(images)) # Plot up to 25
    print(f"Displaying {num_samples} samples for: {title}")
    for i in range(num_samples):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray') # Assuming grayscale
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.show()

# Plot samples
plot_samples(x_train[:25], y_train_labels[:25], "Training Set Samples (Start)")
plot_samples(x_val_samples[:25], y_val_labels[:25], f"Validation Split Samples (Last {val_split_fraction*100}%)")
plot_samples(x_test[:25], y_test_labels[:25], "Test Set Samples")
print("--- Finished Visualizing Data Samples ---\n")"""