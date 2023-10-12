# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Define the path to the dataset
data_dir = "Dataset"

# Define a list of classes (folder names)
classes = os.listdir(data_dir)

# Initialize lists to store image data and labels
data = []
labels = []

# Load and preprocess the dataset
for i, c in enumerate(classes):
    class_dir = os.path.join(data_dir, c)
    for image_name in os.listdir(class_dir):
        img = cv2.imread(os.path.join(class_dir, image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, (128, 128))  # Resize to a consistent size
        data.append(img)
        labels.append(i)

# Convert data and labels to NumPy arrays
data = np.array(data) / 255.0  # Normalize pixel values to the range [0, 1]
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define a simple CNN model
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(classes), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Save the model for future use
model.save("medical_mnist_model.h5")