# Food Item Classification from Images

Author: Aatif Mashkoor  
Date: May 2026  

## Project Overview
This project uses Transfer Learning with a pretrained ResNet50 Convolutional Neural Network (CNN) to classify images of food into 25 distinct categories. The dataset used is a subset of the Food-101 dataset.

## Directory Structure
The project strictly follows the required grading format:
* `_checkpoints/`: Contains the `_final_weights.pth` file after training.
* `data/`: Contains 10 raw `.jpg` sample images per class.
* `config.py`: Hyperparameters and configuration variables.
* `dataset.py`: Custom PyTorch Dataset and DataLoader using OpenCV.
* `model.py`: Modified ResNet50 model setup.
* `train.py`: The `train_model` loop function.
* `predict.py`: The `classify_food` inference function.
* `interface.py`: Standardized import mappings for automated grading.

## Installation Instructions
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/Aatif34/project_aatif_mashkoor.git](https://github.com/Aatif34/project_aatif_mashkoor.git)
   cd project_aatif_mashkoor

2. Install the required Python packages:

Bash
pip install torch torchvision opencv-python numpy

Execution Instructions
Because this project is structured as a module for automated grading (via interface.py), the files contain functions rather than direct execution scripts.

To Predict/Classify Images:
The prediction function automatically loads the trained model weights from the _checkpoints folder.

Python
from predict import classify_food

# Run classification on a list of image paths
image_paths = ["data/pizza/img1.jpg", "data/sushi/img1.jpg"]
predictions = classify_food(image_paths)
print(predictions)
