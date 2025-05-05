import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split # For reliable validation split
import numpy as np
import matplotlib.pyplot as plt # Optional for visualization

print(f"Using TensorFlow version: {tf.__version__}")

# --- 1. Define Constants & Load Data ---
num_classes = 40
img_height = 28
img_width = 28
channels = 1
input_shape = (img_height, img_width, channels)
num_conv_layers = 10
decay_rate_gamma = 0.98
initial_learning_rate = 0.0001 # Using the low LR from previous successful attempts
batch_size = 64 # Consistent batch size

print("Loading dataset...")
dataset_path = 'data/uhat_dataset.npz' # Make sure path is correct
try:
    urdu_dataset = np.load(dataset_path)
    # Using _full suffix before splitting
    x_train_full, y_train_full = urdu_dataset['x_chars_train'], urdu_dataset['y_chars_train']
    x_test, y_test = urdu_dataset['x_chars_test'], urdu_dataset['y_chars_test']
    print(f"Dataset loaded successfully.")
    print(f"Initial shapes: x_train_full={x_train_full.shape}, y_train_full={y_train_full.shape}")
    print(f"Initial shapes: x_test={x_test.shape}, y_test={y_test.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Preprocess Data ---
print("Preprocessing data...")
# Normalize pixels
if x_train_full.max() > 1.0:
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
else:
    x_train_full = x_train_full.astype('float32')
    x_test = x_test.astype('float32')

# Reshape for CNN (add channel dimension)
if x_train_full.ndim == 3:
    x_train_full = x_train_full.reshape((-1, img_height, img_width, channels))
    x_test = x_test.reshape((-1, img_height, img_width, channels))

# One-hot encode labels
y_train_full_cat = to_categorical(y_train_full, num_classes)
y_test_cat = to_categorical(y_test, num_classes) # Also encode test labels
print(f"Processed shapes: x_train_full={x_train_full.shape}, y_train_full_cat={y_train_full_cat.shape}")
print(f"Processed shapes: x_test={x_test.shape}, y_test_cat={y_test_cat.shape}")

# --- Create Dedicated Validation Set using sklearn ---
print("Splitting data into training and validation sets (80/20 split)...")
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full_cat,
    test_size=0.2, # Use 20% of the original training data for validation
    random_state=42, # For reproducible splits
    shuffle=True # Shuffle before splitting
)
print(f"Train shapes: x={x_train.shape}, y={y_train.shape}")
print(f"Validation shapes: x={x_val.shape}, y={y_val.shape}")

# --- 3. Define the Described CNN Model ---
print("\nBuilding described CNN model (10 Conv layers, no pooling)...")
model = Sequential(name=f"cnn_{num_conv_layers}_conv_no_pool")
model.add(Input(shape=input_shape))

# Add Convolutional Blocks in a loop
for i in range(num_conv_layers):
    num_filters = 16 * (i + 1)
    layer_name_base = f"conv_block_{i+1}" # Layer indexing from 1
    print(f"Adding Conv Block {i+1}: {num_filters} filters")

    # Conv2D -> BatchNormalization -> Activation
    model.add(Conv2D(num_filters,
                     kernel_size=(3, 3),
                     padding='same', # Keep dimensions same since no pooling
                     use_bias=False, # Recommended before BN
                     name=f"{layer_name_base}_conv"))
    model.add(BatchNormalization(name=f"{layer_name_base}_bn"))
    model.add(Activation('relu', name=f"{layer_name_base}_relu"))
    # NO POOLING LAYER HERE

# Flatten and Output Layer
model.add(Flatten(name="flatten"))
print("Added Flatten layer.")
model.add(Dropout(0.2, name="dropout"))
print("Added Dropout layer with rate 0.4.")
model.add(Dense(num_classes, activation='softmax', name="output_dense"))
print("Added Flatten and Output Dense layer.")

# --- 4. Define Learning Rate Schedule and Optimizer ---
# Calculate decay steps to decay once per epoch
steps_per_epoch = len(x_train) // batch_size
if steps_per_epoch == 0: steps_per_epoch = 1 # Avoid division by zero

lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=steps_per_epoch,
    decay_rate=decay_rate_gamma, # Use gamma = 0.98
    staircase=True # Decay applied at discrete intervals
)
print(f"Defined ExponentialDecay LR schedule: initial={initial_learning_rate}, steps_per_epoch={steps_per_epoch}, decay_rate={decay_rate_gamma}")

optimizer = Adam(learning_rate=lr_schedule)

# --- 5. Compile the Model ---
print("Compiling model...")
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6. Model Summary ---
print("\nModel Summary:")
model.summary()

# --- 7. Prepare Callbacks ---
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=15, # Allowing a bit more patience as convergence might be different
    verbose=1,
    restore_best_weights=True
)
print(f"Using EarlyStopping with patience={early_stopping_callback.patience}")

# --- 8. Train the Model ---
print("\nStarting Training (max 100 epochs)...")
epochs = 100 # Max epochs, early stopping will likely trigger sooner

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_val, y_val), # Use the dedicated validation set
    callbacks=[early_stopping_callback]
)

print("\nTraining Finished.")

print("\n--- Plotting Training History ---")
# Check if history object and necessary keys exist
# Sometimes keys might differ slightly depending on TF version or compilation settings
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
    plt.title('Training and Validation Accuracy')
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
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # Find the best validation loss epoch (usually where EarlyStopping restores from)
    best_epoch_loss = np.argmin(val_loss) + 1
    plt.axvline(best_epoch_loss, linestyle='--', color='r', label=f'Best Val Loss (Epoch {best_epoch_loss})')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.suptitle('Model Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.show() # Display the plots
    print(f"Plots displayed. Best validation loss was at epoch {best_epoch_loss}.")

else:
    print("Could not generate plots. History object might be missing required keys.")
    if hasattr(history, 'history'):
        print(f"Keys available in history: {list(history.history.keys())}")
# --- END PLOTTING CODE SECTION ---

# --- 9. Evaluate the Model ---
# Evaluation uses the weights restored by EarlyStopping (best val_loss epoch)
print("\nEvaluating Model on Test Data...")
score = model.evaluate(x_test, y_test_cat, verbose=0) # Ensure test labels are one-hot encoded
print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f}')