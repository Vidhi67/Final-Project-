import os
import cv2
from keras.models import load_model
import numpy as np
import random

##### Adjust this section as needed #####
model_path = './UNetW_final.h5'
Training_data = r'../Preprocessing/preprocessed_img/'
save_dir = r'./Masks/'
batch_size = 32 # Adjust batch size according to your system's memory
#num_samples = 1000  # Number of random images to process (1000 taken as test case)
#########################################

# Create save directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  

model = load_model(model_path)
width = 512
height = 512

# Prepare training data
training_data = [x for x in sorted(os.listdir(Training_data))]
num_images = len(training_data)
#random_samples = random.sample(training_data, min(num_samples, len(training_data)))

# Function to load images in batches
def load_batch(batch_files, directory):
    batch_data = []
    for file in batch_files:
        image_path = os.path.join(directory, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            resized_image = cv2.resize(image, (width, height))
            batch_data.append(resized_image)
    return np.array(batch_data).reshape(len(batch_data), width, height, 1)

# Process images in batches
for i in range(0, num_images, batch_size):
    batch_files = training_data[i:i + batch_size]
    x_train_data = load_batch(batch_files, Training_data)
    
    # Normalize data if needed
    x_train_data = x_train_data.astype('float32') / 255.0
    
    # Running inferences 
    result = model.predict(x_train_data, verbose=1)
    result = (result > 0.5).astype(np.uint8)
    
    for mask, image_filename in zip(result, batch_files):
        # Extraction of Image Basename
        image_base = os.path.basename(image_filename)
        
        # Convert mask to 0-255 scale and reshape if needed
        mask_scaled = (mask * 255).astype(np.uint8)
        mask_2d = np.reshape(mask_scaled, (width, height))
        
        # Construct save path with matching image name
        save_path = os.path.join(save_dir, image_base)
        
        # Save the mask image
        cv2.imwrite(save_path, mask_2d)

print(f"Saved masks to {save_dir}")
