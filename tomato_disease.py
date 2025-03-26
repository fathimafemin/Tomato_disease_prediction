# -*- coding: utf-8 -*-
"""TOMATO_DISEASE.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1V2TTn0ocujkvEluyt8WDC_RAEv8RT_eA
"""

!curl -L -o /content/archive.zip https://www.kaggle.com/api/v1/datasets/download/kaustubhb999/tomatoleaf
!unzip /content/archive.zip

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prompt: show number of files from each of list of folders

import os

train_dir = "/content/tomato/train"
folders = [os.path.join(train_dir, folder) for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]
folders.sort()
print(len(folders), 'folders in train')
print(folders)

print('Folders and files in Train')
for folder in folders:
  num_files = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
  print(f"Folder: {folder}, Number of files: {num_files}")

val_dir = "/content/tomato/val"
val_folders = [os.path.join(val_dir, folder) for folder in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, folder))]

print(len(val_folders), 'folders in val')
val_folders.sort()
print(val_folders)


print('Folders and files in Val')
for folder in val_folders:
  num_files = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
  print(f"Folder: {folder}, Number of files: {num_files}")

# # prompt: ['/content/tomato/train/Tomato___Bacterial_spot', '/content/tomato/train/Tomato___Early_blight', '/content/tomato/train/Tomato___Late_blight', '/content/tomato/train/Tomato___Leaf_Mold', '/content/tomato/train/Tomato___Septoria_leaf_spot', '/content/tomato/train/Tomato___Spider_mites Two-spotted_spider_mite', '/content/tomato/train/Tomato___Target_Spot', '/content/tomato/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus', '/content/tomato/train/Tomato___Tomato_mosaic_virus', '/content/tomato/train/Tomato___healthy']
# # preprosess the images using open cv without over exherting the ram usage and train

# import os
# import random
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import cv2
# import numpy as np

# def preprocess_image(image_path, target_size=(224, 224)):
#   img = cv2.imread(image_path)
# #   # Resize the image
# #   img = cv2.resize(img, target_size)
# #   # Normalize pixel values to the range [0, 1]
# #   img = img.astype(np.float32) / 255.0
# #   return img


# # def process_images_in_folder(folder_path, target_size=(224, 224), max_images_per_folder=None):
# #     image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]
# #     if max_images_per_folder:
# #         image_paths = image_paths[:max_images_per_folder]

# #     images = []
# #     labels = []
# #     for image_path in image_paths:
# #         preprocessed_img = preprocess_image(image_path, target_size)
# #         if preprocessed_img is not None:
# #             images.append(preprocessed_img)
# #             label = os.path.basename(folder_path)
# #             labels.append(label)
# #     return np.array(images), np.array(labels)

# # # Process images and labels for training and validation
# # final_images = []
# # final_labels = []
# # for folder in folders:
# #     imgs, lbls = process_images_in_folder(folder, max_images_per_folder=1000)
# #     if imgs is not None:
# #       final_images.extend(imgs)
# #       final_labels.extend(lbls)

# # # Convert labels to numerical values using LabelEncoder
# # from sklearn.preprocessing import LabelEncoder
# # le = LabelEncoder()
# # final_labels_encoded = le.fit_transform(final_labels)

# # print(len(final_labels_encoded))
# # print(len(final_images))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import layers

# Define the data generator for training and validation sets
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # Rescale pixel values

# Load images in batches from folders for training
train_generator = datagen.flow_from_directory(
    '/content/tomato/train',  # Folder with your image data
    target_size=(224, 224),   # Image target size
    batch_size=32,            # Load 32 images at a time
    class_mode='sparse', # Multiclass classification
    subset='training'         # Use this for training set
)

# Load images in batches for validation
validation_generator = datagen.flow_from_directory(
    '/content/tomato/train',  # Folder with your image data
    target_size=(224, 224),   # Image target size
    batch_size=32,            # Load 32 images at a time
    class_mode='sparse', # Multiclass classification
    subset='validation',       # Use this for validation set
)

# Define the CNN model
model = Sequential()

model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10 , activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# the model using the generator
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")