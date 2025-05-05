# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt # Optional for visualization

print(f"Using TensorFlow version: {tf.__version__}")

# --- 1. Define Constants & Load Data ---
num_classes = 40
# *** IMPORTANT: Verify these dimensions match your actual image data! ***
# If your images are not 28x28 grayscale, change these values.
img_height = 28
img_width = 28
channels = 1 # Grayscale images have 1 channel
input_shape = (img_height, img_width, channels)

print("Loading dataset...")
# Make sure this path is correct for your system
dataset_path = 'data/uhat_dataset.npz'
try:
    urdu_dataset = np.load(dataset_path)
    x_train, y_train = urdu_dataset['x_chars_train'], urdu_dataset['y_chars_train']
    x_test, y_test = urdu_dataset['x_chars_test'], urdu_dataset['y_chars_test']
    print(f"Dataset loaded successfully from {dataset_path}.")
except FileNotFoundError:
    print(f"Error: Dataset file not found at {dataset_path}.")
    print("Please check the path. Exiting.")
    exit() # Stop if data can't be loaded
except KeyError as e:
    print(f"Error: Key {e} not found in the dataset file. Check the .npz structure.")
    exit()

print(f"Initial shapes: x_train={x_train.shape}, y_train={y_train.shape}")
print(f"Initial shapes: x_test={x_test.shape}, y_test={y_test.shape}")

# --- 2. Preprocess Data ---
# Verify data range and normalize (assuming 0-255)
print("Preprocessing data...")
if x_train.max() > 1.0:
    print("Normalizing pixel values (dividing by 255.0)...")
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
else:
    print("Pixel values seem to be already normalized (max <= 1.0).")
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')


# Reshape for CNN: Add the channel dimension if it's missing
if x_train.ndim == 3: # Check if data is (samples, height, width)
    print(f"Reshaping data to include channel dimension: {input_shape}")
    x_train = x_train.reshape((-1, img_height, img_width, channels))
    x_test = x_test.reshape((-1, img_height, img_width, channels))
elif x_train.ndim == 4 and x_train.shape[-1] == channels: # Already has channel dim
     print("Data already has the correct channel dimension.")
else:
    print(f"Error: Unexpected data shape {x_train.shape}. Expected (samples, {img_height}, {img_width}) or (samples, {img_height}, {img_width}, {channels}).")
    exit()

print(f"Reshaped data shapes: x_train={x_train.shape}, x_test={x_test.shape}")

# One-hot encode labels
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print(f"One-hot encoded label shapes: y_train={y_train.shape}, y_test={y_test.shape}")


#data augmentation
# This is a very powerful technique against overfitting. 
# Create slightly modified versions of your training images on-the-fly during training.

data_augmentation = Sequential([
    RandomRotation(0.05),
    RandomZoom(0.05),
], name="data_augmentation")

# --- 3. Define the Basic CNN Model ---
print("\nBuilding Basic CNN model...")
model = Sequential([
    # Input layer - specify shape (height, width, channels)
    Input(shape=input_shape),
    data_augmentation,

    # Convolutional Layer 1: Learns local patterns.
    # 32 filters: Extracts 32 different basic feature types.
    # kernel_size=(3, 3): Uses a 3x3 pixel window to scan the image.
    # activation='relu': Standard activation function.
    Conv2D(32, kernel_size=(3, 3), activation='relu'), # Using 'same' padding initially
    #BatchNormalization(),
    #Activation('relu'),
    
    # Pooling Layer 1: Downsamples the feature maps.
    # pool_size=(2, 2): Reduces height and width by half. Makes features more robust.
    #MaxPooling2D(pool_size=(2, 2)),

    #Block 2

    #Conv2D(64, kernel_size=(3,3), padding='same', use_bias=False),

    #BatchNormalization(),
    #Activation('relu'),

    MaxPooling2D(pool_size=(2, 2)),

    # Flatten the results: Converts 2D feature maps to a 1D vector.
    Flatten(),

    # Dense Hidden Layer: Learns combinations of features extracted by Conv layers.
   #  Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),

    # Dropout Layer: Helps prevent overfitting.
    Dropout(0.5), # Randomly ignores 50% of neurons during training.

    # Output Layer: Final classification into one of the 40 classes.
    Dense(num_classes, activation='softmax') # Softmax for multi-class probability output.

], name="cnn_v2_simple")

# --- 4. Compile the Model ---
print("Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),             # Adam optimizer is generally a good default
    loss='categorical_crossentropy', # Correct loss for one-hot encoded multi-class
    metrics=['accuracy']          # Track accuracy during training
)

# --- 5. Model Summary ---
print("\nModel Summary:")
model.summary()

# --- 6. Prepare Callbacks ---
# Stop training early if validation loss doesn't improve for 'patience' epochs
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,            # Wait 10 epochs for improvement before stopping
    verbose=1,              # Print message when stopping
    restore_best_weights=True # Use weights from the best epoch found
)

# --- 7. Train the Model ---
print("\nStarting Training (max 100 epochs, will stop early if no improvement)...")
batch_size = 64 # Common batch size for CNNs, adjust if needed (32, 128)
epochs = 100    # Set a high number, EarlyStopping will manage the actual duration

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,              # Show progress bar per epoch
    validation_split=0.2,   # Use 20% of training data for validation (adjust as needed)
    callbacks=[early_stopping_callback] # Apply early stopping
)

print("\nTraining Finished.")

print("\n--- Plotting Training History ---")

# Check if history object and necessary keys exist
required_keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
if hasattr(history, 'history') and all(key in history.history for key in required_keys):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Determine the number of epochs actually run
    epochs_run = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6)) # Adjust figure size (width, height)

    # Plot Training & Validation Accuracy
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    plt.plot(epochs_run, acc, label='Training Accuracy', marker='o', linestyle='-')
    plt.plot(epochs_run, val_acc, label='Validation Accuracy', marker='o', linestyle='-')
    plt.title('Training and Validation Accuracy (v2_simple)') # Added model name
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # Find the best validation accuracy epoch
    best_epoch_acc = np.argmax(val_acc) + 1
    plt.axvline(best_epoch_acc, linestyle='--', color='r', label=f'Best Val Acc (Epoch {best_epoch_acc})')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
    plt.plot(epochs_run, loss, label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs_run, val_loss, label='Validation Loss', marker='o', linestyle='-')
    plt.title('Training and Validation Loss (v2_simple)') # Added model name
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # Find the best validation loss epoch (usually where EarlyStopping restores from)
    best_epoch_loss = np.argmin(val_loss) + 1
    plt.axvline(best_epoch_loss, linestyle='--', color='r', label=f'Best Val Loss (Epoch {best_epoch_loss})')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.suptitle('Model Training History (cnn_v2_simple)', fontsize=16) # Overall title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    # --- Suggestion: Save the plot ---
    plot_filename = "v2.png"
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
    except Exception as e:
        print(f"Could not save plot: {e}")
    # --- End Suggestion ---
    plt.show() # Display the plots
    print(f"Plots displayed. Best validation loss was at epoch {best_epoch_loss}.")


else:
    print("Could not generate plots. History object might be missing required keys.")
    if hasattr(history, 'history'):
        print(f"Keys available in history: {list(history.history.keys())}")

# --- 8. Evaluate the Model ---
# Evaluation uses the weights restored by EarlyStopping (best val_loss)
print("\nEvaluating Model on Test Data...")
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f}')


# --- 9. Ideas for Building Upon This ---
#print("\n--- How to Build Upon This Basic CNN ---")
#print("1. **Add More Conv/Pool Layers:** Create deeper networks to learn more complex features.")
#print("   Example: Add `Conv2D(64, ...), MaxPooling2D(...)` before the Flatten layer.")
#print("2. **Experiment with Filters/Kernels:** Try different numbers of filters (e.g., 64 instead of 32) or kernel sizes (e.g., (5, 5)).")
#print("3. **Use Batch Normalization:** Add `BatchNormalization()` layers after Conv or Dense layers to potentially stabilize training.")
#print("4. **Tune Dropout:** Adjust the dropout rate (e.g., 0.3, 0.4) or try `SpatialDropout2D` after Conv layers.")
#print("5. **Data Augmentation:** Use `ImageDataGenerator` or Keras preprocessing layers to artificially increase training data diversity (rotations, shifts, zooms).")
#print("6. **Adjust Learning Rate:** If Adam isn't working well, try `Adam(learning_rate=0.0001)`.")