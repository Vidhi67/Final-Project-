The code is divided into 3 parts and should be executed in the order listed below.

•	Preprocessing ............................

o	Dataset_Preprocessing: Takes the glaucoma csv and solves the label disagreement. After than splits it into train and balanced test csv. For the feature glaucoma it takes the RG instances and split it into 90/10 ratio for train and test. 

o	Image_Preprocessing: It applies CLAHE contrast enhancement as well as Trims the margins of an image where the pixel intensity is near zero, padding black pixels to maintain square ratio of an image and resizing to desired output_size. For the output_size I am using 2000 x 2000.
Before running the following file, you need to run the files mention in Yolov8 folder in same order as they are listed. 

o	OD_segmentation: This helps to find the Region of interest using Yolov8-segmentation. Here comes the use of Yolov8 folder before running this file. We used that pretrained Yolov8 model to find the region of interest using the bboxing method. Meaning it would detect the optic disc finds it center and crop the image in 512x512 size. 

•	Yolov8 .....................................

o	UNet_mask_new: This code uses the pretrained model UNetw to find the optic disc mask in the preprocessed image making it easier for the Yolov8 model to concentrate on detecting the optic disc within the defined region. 

o	Mask_preprocessing: This code takes the masked image and creates a label. Meaning changing into the polygon to get the size of it. With that it copies the mask image and its label in the data folder with 70/30 ratio for train and val used by Yolov8-segmentation. 

o	Yolov8_training: Now this masks and labels from the data folder where used for the training phase of Yolov8.

o	Yolov8-inference:  This helps to identify how well the model is performing on the entire dataset, giving the number of missing masks for the images. 

•	Models _training .......................

o	Data_utils: This code contains helper functions and a custom dataset class used by the Vision Transformer (ViT) training and validation scripts. The helper functions address class imbalance in the training set by applying data augmentation techniques to generate additional instances for the minority class, ensuring a balanced dataset. The custom GlaucomaDataset class handles image loading, applies the specified transformations, and manages label extraction, ensuring the dataset is prepared correctly for training and evaluation.

o	ViT_glaucoma_training: This code trains a Vision Transformer (ViT) model for glaucoma detection, handling both ROI and non-ROI images sequentially. It sets up data loading, applies data augmentation, and manages class imbalance with a weighted loss function also model is fine-tuned on ImageNet-pretrained weights with customized learning rates.

o	ViT_glaucoma_validation: This code evaluates the model's accuracy using a balanced test dataset and applies random data augmentation to enhance accuracy. It calculates sensitivity at 95% specificity and overall accuracy, providing better insights into the model's performance.

o	ViT_10_features_training_with_validation: This code is designed for multilabel classification, focusing specifically on referable glaucoma (RG) instances using both ROI and non-ROI images. It trains and validates the Vision Transformer (ViT) model in each epoch, similar to the approach used in ViT_glaucoma_training. The model processes only RG instances, and during training, it provides insights into the hamming loss to evaluate how well the model is learning.



