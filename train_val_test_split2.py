# Python script to split labeled image dataset into Train, Validation, and Test folders.

import glob
from pathlib import Path
import random
import os
import shutil

# Define paths 
image_path = '/content/images/images'
train_path = '/content/images/train' 
val_path = '/content/images/validation'
test_path = '/content/images/test'

# Get image and annotation lists
jpg_images = list(Path(image_path).glob('*.jpg'))
jpgs_annotations = [Path(image_path) / Path(str(f).replace('.jpg','.xml')) for f in jpg_images]



print("JPG images found:", len(jpg_images))


# Determine split proportions
total_files = len(jpg_images) 
train_split = 0.8
val_split = 0.1
test_split = 0.1

num_train = int(total_files * train_split)
num_val = int(total_files * val_split)
num_test = total_files - num_train - num_val

print("Total images:", total_files)
print("Training images:", num_train) 
print("Validation images:", num_val)
print("Test images:", num_test)

# Shuffle file lists
shuffled_images = random.sample(jpg_images, len(jpg_images))
shuffled_annotations = [jpgs_annotations[jpg_images.index(f)] for f in shuffled_images]

# Split files
train_images = shuffled_images[:num_train]
train_annotations = shuffled_annotations[:num_train]

val_images = shuffled_images[num_train:num_train+num_val]  
val_annotations = shuffled_annotations[num_train:num_train+num_val]

test_images = shuffled_images[num_train+num_val:]
test_annotations = shuffled_annotations[num_train+num_val:]

# Move files to folders
for image in train_images:
  shutil.move(image, train_path)
for annotation in train_annotations:
  shutil.move(annotation, train_path)
  
for image in val_images:
  shutil.move(image, val_path)
for annotation in val_annotations:
  shutil.move(annotation, val_path)
  
for image in test_images:
  shutil.move(image, test_path)
for annotation in test_annotations:
  shutil.move(annotation, test_path)
