import zipfile
import os

# Unzip the file
with zipfile.ZipFile('DATASET.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/DATASET')  # Extract the contents to /content/dataset

# Verify that the dataset was extracted
!ls /content/DATASET

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the extracted dataset
dataset_path = '/content/DATASET'

# Create an ImageDataGenerator object
datagen = ImageDataGenerator(rescale=1./255)  # Normalize the images

# Load the dataset from the directory
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(200, 200),  # Resize images to 200x200 pixels
    color_mode='grayscale',  # Convert images to grayscale
    batch_size=32,           # Batch size of 32 images
    class_mode='categorical'  # Assuming you have multiple classes
)

print(f"Classes: {train_generator.class_indices}")  # Shows class mapping
print(f"Number of images: {train_generator.samples}")  # Shows the total number of images

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation to artificially increase dataset size
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # Random rotation
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2, # Vertical shift
    shear_range=0.2,        # Shear transformation
    zoom_range=0.2,         # Zoom in/out
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='nearest'     # Filling missing pixels after transformation
)

train_generator = datagen.flow_from_directory(
    '/content/DATASET',
    target_size=(200, 200),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)



from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Set the path to your dataset (replace with the actual path to your 'train' folder)
dataset_path = '/content/DATASET'

# Step 2: Create an ImageDataGenerator object
datagen = ImageDataGenerator(rescale=1./255)  # Normalize images

# Step 3: Load the dataset from the folder
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(200, 200),  # Resize images to 200x200 pixels
    color_mode='grayscale',  # Convert images to grayscale
    batch_size=32,           # Load 32 images at a time
    class_mode='categorical'  # Assuming your labels are categorical (for multi-class classification)
)

# Step 4: Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)),
    MaxPool2D(2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPool2D(2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10  # Adjust the number of epochs as needed
)

# Step 5: Compile the model
import keras
METRICS = [
    'accuracy',
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall')
]
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=METRICS)

# Step 6: Train the model with the dataset
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10  # Adjust based on your needs
)

import cv2
import numpy as np

def remove_white_border(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(contours[0])
    # Crop the image to remove the white border
    return image[y:y+h, x:x+w]

# Apply it to each image in the training set
def preprocess_images(image_path):
    img = cv2.imread(image_path)
    img_no_border = remove_white_border(img)
    return img_no_border


import numpy as np

# Set the number of images to predict
num_images_to_predict = 100
predicted_labels = []
actual_labels = []
filenames = []

# Initialize counters
current_count = 0

# Loop through batches until you reach the desired number of images
for batch_images, batch_labels in train_generator:
    # Loop through each image in the current batch
    for i in range(len(batch_images)):
        if current_count >= num_images_to_predict:
            break  # Stop if you've reached the desired number of images
        
        # Get the current image and expand dimensions for prediction
        image = np.expand_dims(batch_images[i], axis=0)
        
        # Predict the class (Cancerous or Non-Cancerous)
        prediction = model.predict(image)
        
        # Assuming binary classification (0 = Cancerous, 1 = Non-Cancerous)
        if prediction[0][0] < 0.5:
            predicted_labels.append("Non-Cancerous")
        else:
            predicted_labels.append("Cancerous")
        
        # Store the actual label and filename for reference
        actual_labels.append("Cancerous" if np.argmax(batch_labels[i]) == 0 else "Non-Cancerous")
        filenames.append(train_generator.filenames[train_generator.batch_index * train_generator.batch_size + i])
        
        current_count += 1

    if current_count >= num_images_to_predict:
        break  # Stop once you've predicted 100 images

# Display the predictions
for idx in range(num_images_to_predict):
    print(f"Image: {filenames[idx]}, Predicted: {predicted_labels[idx]}, Actual: {actual_labels[idx]}")