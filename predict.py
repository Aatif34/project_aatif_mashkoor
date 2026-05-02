import cv2
import torch
import numpy as np
from config import resize_x, resize_y

# The 25 distinct classes (must be sorted alphabetically to match the dataset.py folder loading logic)
classes = sorted([
    'pizza', 'sushi', 'samosa', 'ramen', 'fried_rice', 
    'hamburger', 'hot_dog', 'ice_cream', 'donuts', 'french_fries',
    'chicken_wings', 'dumplings', 'macaroni_and_cheese', 'nachos', 'omelette',
    'pancakes', 'tacos', 'waffles', 'biryani', 'churros',
    'cheesecake', 'cup_cakes', 'edamame', 'garlic_bread', 'gyoza'
])

def classify_food(list_of_img_paths, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set the model to evaluation mode (important for ResNet so it doesn't apply dropout/batchnorm updates)
    model.eval()
    
    predictions = []
    
    for img_path in list_of_img_paths:
        # 1. Read and preprocess image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            predictions.append("Error: Image not found")
            continue
            
        # Convert BGR to RGB and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (resize_x, resize_y))
        
        # Normalize and convert to PyTorch tensor shape (Channels, Height, Width)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        # Add a batch dimension because the model expects [batch_size, channels, height, width]
        image_tensor = torch.tensor(image).unsqueeze(0).to(device)
        
        # 2. Predict the outcome
        with torch.no_grad():
            logits = model(image_tensor)
            _, pred_idx = torch.max(logits, 1)
            
        # 3. Translate the predicted index back into the text label (e.g., "pizza")
        predicted_label = classes[pred_idx.item()]
        predictions.append(predicted_label)
        
    return predictions
