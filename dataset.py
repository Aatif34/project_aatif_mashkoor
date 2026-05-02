import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Importing parameters directly from config.py as strictly required
from config import resize_x, resize_y, batch_size

class Food101Dataset(Dataset):
    def __init__(self, img_dir, apply_augmentations=False):
        self.img_dir = img_dir
        self.apply_augmentations = apply_augmentations
        self.img_paths = []
        self.labels = []
        
        # Automatically assign labels based on folder names
        self.classes = sorted(os.listdir(img_dir))
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(img_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.jpg'):
                        self.img_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.img_paths)
        
    def augment_image(self, image):
        # 1. Random Horizontal Flipping
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1) 
            
        # 2. Random Rotation (-15 to 15 degrees)
        if np.random.rand() > 0.5:
            angle = np.random.randint(-15, 15)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
            
        # 3. Random Brightness Adjustment
        if np.random.rand() > 0.5:
            value = np.random.randint(-30, 30)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, value)
            v[v > 255] = 255
            v[v < 0] = 0
            final_hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
            
        return image

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Read image using OpenCV
        image = cv2.imread(img_path)
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize using config variables
        image = cv2.resize(image, (resize_x, resize_y))
        
        # Apply augmentations if specified (usually only for training data)
        if self.apply_augmentations:
            image = self.augment_image(image)
            
        # Normalize to [0, 1] and convert to PyTorch tensor format (Channels, Height, Width)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        label = self.labels[idx]
        
        return torch.tensor(image), torch.tensor(label, dtype=torch.long)

def get_food_loader(img_dir, apply_augmentations=False, shuffle=True):
    dataset = Food101Dataset(img_dir, apply_augmentations)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
