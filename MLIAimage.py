# Import required libraries
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np

# Set the path to the dataset folder
dataset_path = "Photo moi 2"

# Set the image size and channels
img_width, img_height = 3456, 4608
channels = 3

# Set the number of epochs and batch size for training the model
epochs = 50
batch_size = 1

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Use Keras' ImageDataGenerator to preprocess the images and generate training and validation sets
train_datagen = image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = image.ImageDataGenerator(rescale=1./255)

# Check the number of classes and class indices
training_set = train_datagen.flow_from_directory(dataset_path, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')
test_set = test_datagen.flow_from_directory(dataset_path, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')

# Fit the model to the training data and validate it with the test data
model.fit(training_set, epochs=epochs, validation_data=test_set)

# Save the trained model for future use
model.save('model.h5')

# Load the saved model
model = load_model('model.h5')

# Set the path to the image file you want to predict
image_path = 'Photo moi 2\IMG_20230302_134613.jpg'

# Load the image and preprocess it
img = image.load_img(image_path, target_size=(img_width, img_height))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img /= 255.

# Make the prediction
prediction = model.predict(img)

# Print the prediction (0 if not the person, 1 if it is)
print("RES : ", prediction[0])
