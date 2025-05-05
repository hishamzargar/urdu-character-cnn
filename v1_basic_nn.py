# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt # Optional: for visualizing data/results

print(f"Using TensorFlow version: {tf.__version__}")

# 1. Load the MNIST dataset
# MNIST contains 60,000 training images and 10,000 testing images
# Each image is a 28x28 grayscale picture of a handwritten digit (0-9)
urdu_dataset = np.load('data/uhat_dataset.npz') #assuming data directory and this file are in the same directory
x_train, y_train = urdu_dataset['x_chars_train'], urdu_dataset['y_chars_train']
x_test, y_test = urdu_dataset['x_chars_test'], urdu_dataset['y_chars_test']

# Print shapes to understand the data
print(f"Training data shape: {x_train.shape}") # (60000, 28, 28)
print(f"Training labels shape: {y_train.shape}") # (60000,)
print(f"Test data shape: {x_test.shape}")       # (10000, 28, 28)
print(f"Test labels shape: {y_test.shape}")       # (10000,)

# 2. Preprocess the data
# Normalize pixel values from 0-255 to 0.0-1.0
# Neural networks generally work better with smaller input values.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels (integers 0-9) to one-hot encoded vectors
# Example: 5 becomes [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
num_classes = 40
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Training labels shape after one-hot encoding: {y_train.shape}") # (60000, 10)
print(f"Test labels shape after one-hot encoding: {y_test.shape}")       # (10000, 10)


# 3. Define the Model Architecture (Basic Neural Network)
# We use a Sequential model, which is a linear stack of layers.
model = Sequential([
    # Flatten layer: Converts the 28x28 image into a 1D vector of 784 pixels.
    # This is needed because Dense layers expect 1D input.
    Flatten(input_shape=(28, 28)),

    # Hidden layer: A Dense (fully connected) layer with 128 neurons.
    # 'relu' (Rectified Linear Unit) is a common activation function.
    Dense(1024,
          activation='relu',
          #adding l2 regularization
          kernel_regularizer=regularizers.l2(0.001)),

    #trying droupout 0.5
    Dropout(0.5),

    # Output layer: A Dense layer with 10 neurons (one for each digit class).
    # 'softmax' activation converts the outputs into probability-like values
    # for each class, summing to 1.
    Dense(num_classes, activation='softmax')
])

# 4. Compile the Model
# Before training, we need to configure the learning process.
model.compile(
    # Optimizer: 'sgd' (Stochastic Gradient Descent) is a basic optimizer.
    # Other common choices are 'adam', 'rmsprop'.
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),

    # Loss function: 'categorical_crossentropy' is standard for multi-class
    # classification when labels are one-hot encoded.
    # If labels were integers, use 'sparse_categorical_crossentropy'.
    loss='categorical_crossentropy',

    # Metrics: What to monitor during training and evaluation.
    # 'accuracy' measures the proportion of correctly classified images.
    metrics=['accuracy']
)

# Print a summary of the model's layers and parameters
print("\nModel Summary:")
model.summary()

# 5. Train the Model
print("\nStarting Training...")
batch_size = 128  # Number of samples per gradient update
epochs = 100       # Number of times to iterate over the entire training dataset


#Use Early Stopping: Prevent the model from training too long and making 
# overfitting worse. Stop training when the validation loss stops improving.

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# The `fit` method trains the model.
# `validation_split=0.1` reserves 10% of the training data for validation
# to monitor performance on unseen data during training.
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, # Set to 1 or 2 to see progress per epoch
                    validation_split=0.1,
                    callbacks=[early_stopping_callback])

print("\nTraining Finished.")

# 6. Evaluate the Model
# Assess the model's performance on the test dataset (data it has never seen).
print("\nEvaluating Model on Test Data...")
score = model.evaluate(x_test, y_test, verbose=0)

print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f}')

# Optional: Visualize Training History (Loss and Accuracy)
plt.figure(figsize=(12, 4))

# # Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# # Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Optional: Make a prediction on a single test image
# import numpy as np
# image_index = 0 # Choose an image index from the test set
# plt.imshow(x_test[image_index].reshape(28, 28), cmap='gray') # Need original test set for plotting
# pred = model.predict(x_test[image_index:image_index+1]) # Predict needs batch dimension
# print(f"Predicted probabilities: {pred}")
# print(f"Predicted label: {np.argmax(pred)}")
# print(f"True label (one-hot): {y_test[image_index]}")
# print(f"True label: {np.argmax(y_test[image_index])}")