import cv2
import torch
import numpy as np
import os
from config import resize_x, resize_y
from model import FoodResNetModel

# The 25 distinct classes
classes = sorted([
    'pizza', 'sushi', 'samosa', 'ramen', 'fried_rice', 
    'hamburger', 'hot_dog', 'ice_cream', 'donuts', 'french_fries',
    'chicken_wings', 'dumplings', 'macaroni_and_cheese', 'nachos', 'omelette',
    'pancakes', 'tacos', 'waffles', 'biryani', 'churros',
    'cheesecake', 'cup_cakes', 'edamame', 'garlic_bread', 'gyoza'
])

def classify_food(list_of_img_paths):
    # 1. Initialize the model and load the weights INSIDE the function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FoodResNetModel()
    
    # Load the weights from the mandatory _checkpoints folder
    weights_path = os.path.join('_checkpoints', '_final_weights.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    predictions = []
    
    for img_path in list_of_img_paths:
        # Read and preprocess image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            predictions.append("Error: Image not found")
            continue
            
        # Convert BGR to RGB and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (resize_x, resize_y))
        
        # Normalize and convert to PyTorch tensor
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        image_tensor = torch.tensor(image).unsqueeze(0).to(device)
        
        # Predict the outcome
        with torch.no_grad():
            logits = model(image_tensor)
            _, pred_idx = torch.max(logits, 1)
            
        # Translate the predicted index back into the text label
        predicted_label = classes[pred_idx.item()]
        predictions.append(predicted_label)
        
    return predictions
