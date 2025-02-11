import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import numpy as np

# Define the transformations for both the image and the mask
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Horizontal Flip
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Brightness and Contrast Adjustment
    A.ElasticTransform(alpha=2, sigma=50, p=0.5, interpolation=0),  # Geometric Distortion
], additional_targets={'mask': 'mask'})  # Apply same transformations to the mask

def augment_and_save(input_img_dir, input_mask_dir, output_img_dir, output_mask_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(input_img_dir) if f.endswith(('.jpg', '.png'))])
    mask_files = sorted([f for f in os.listdir(input_mask_dir) if f.endswith(('.jpg', '.png'))])

    for img_file, mask_file in zip(img_files, mask_files):
        img_path = os.path.join(input_img_dir, img_file)
        mask_path = os.path.join(input_mask_dir, mask_file)

        img = Image.open(img_path) 
        mask = Image.open(mask_path) 

        # Apply augmentation
        augmented = transform(image=np.array(img), mask = np.array(mask, dtype=np.int64))
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        
        aug_mask = aug_mask.astype(bool)  # Binary mask for binary segmentation
            
        # Save augmented image (use PIL to save)
        aug_img_pil = Image.fromarray(aug_img)
        aug_img_pil.save(os.path.join(output_img_dir, 'aug_' + img_file))

        # Save the augmented mask (as multi-class or binary mask)
        aug_mask_pil = Image.fromarray(aug_mask)
        aug_mask_pil.save(os.path.join(output_mask_dir, 'aug_' + mask_file))

    print("Data augmentation complete!")

# Directories for input images and masks, and output directories for augmented data
input_img_dir = 'DeepCrack/train/imgs'
input_mask_dir = 'DeepCrack/train/masks'
output_img_dir = 'DeepCrack/train/imgs'
output_mask_dir = 'DeepCrack/train/masks'

# Call the augmentation and saving function
augment_and_save(input_img_dir, input_mask_dir, output_img_dir, output_mask_dir)
