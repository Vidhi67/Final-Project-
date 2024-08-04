import os
import cv2
import random
import shutil

##### Adjust this section as needed #####
input_dir = './Masks/'      # Directory containing the binary masks
output_dir = './data/labels/' # Directory where YOLO format labels will be saved
output_image_train_dir = './data/images/train/'
output_image_val_dir = './data/images/val/'
output_label_train_dir = './data/labels/train/'
output_label_val_dir = './data/labels/val/'
image_file_dir = '../Preprocessing/images/'
#########################################

# Ensure output directories exist
for dir_path in [output_image_train_dir, output_image_val_dir, output_label_train_dir, output_label_val_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all image files
image_files = [f for f in os.listdir(image_file_dir) if os.path.isfile(os.path.join(image_file_dir, f))]

# Shuffle the list of image files
random.shuffle(image_files)

num_images = 1000
selected_images = image_files[:num_images]
split_index = int(0.7 * len(selected_images))

# Split into training and validation sets
train_files = image_files[:split_index]
val_files = image_files[split_index:]



for filename in os.listdir(input_dir):
     image_path = os.path.join(input_dir, filename)

     # Load the binary mask
     mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
     _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
     H, W = mask.shape
     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

     # Convert the contours to polygons
     polygons = []
     for cnt in contours:
         if cv2.contourArea(cnt) > 200:  # Filter out small areas
             polygon = []
             for point in cnt:
                 x, y = point[0]
                 # Normalize the coordinates
                 polygon.append(x / W)
                 polygon.append(y / H)
             polygons.append(polygon)

     # Write the polygons to a file in YOLO format
     label_file_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.txt')
     with open(label_file_path, 'w') as f:
         for polygon in polygons:
             f.write('0 ')  # YOLO format: class_id (0 for a single class)
             f.write(' '.join(map(str, polygon)))
             f.write('\n')

print(f"Labels saved to {output_dir}")
def copy_files(file_list, source_image_dir, source_label_dir, target_image_dir, target_label_dir):
    for filename in file_list:
        # Copy image file
        shutil.copy(os.path.join(source_image_dir, filename), os.path.join(target_image_dir, filename))

        # Copy corresponding label file
        label_filename = f'{os.path.splitext(filename)[0]}.txt'
        if os.path.exists(os.path.join(source_label_dir, label_filename)):
            shutil.copy(os.path.join(source_label_dir, label_filename), os.path.join(target_label_dir, label_filename))


# Copy training and validation files
copy_files(train_files, image_file_dir, output_dir, output_image_train_dir, output_label_train_dir)
copy_files(val_files, image_file_dir, output_dir, output_image_val_dir, output_label_val_dir)

print("Dataset split and files copied successfully.")
