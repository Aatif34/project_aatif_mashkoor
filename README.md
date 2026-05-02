Food Item Classification from Images
Aatif Mashkoor | May 2026

What is this project?
Basically the idea is simple — you give it a photo of food and it tells you what it is. The model is a pretrained ResNet50 fine-tuned on 25 food categories from the Food-101 dataset. I used transfer learning because training from scratch on Colab would take forever and honestly there's no point when pretrained weights already exist.
The 25 classes include things like pizza, sushi, ramen, tacos, waffles, steak, samosa, dumplings and more — mostly visually distinct foods so the model actually has something meaningful to learn.

Folder structure
The project follows the submission format required for grading:
project_aatif_mashkoor/
├── checkpoints/
│   └── final_weights.pth
├── data/
│   └── (10 sample images per class)
├── config.py
├── dataset.py
├── model.py
├── train.py
├── predict.py
└── interface.py

How to set it up
Clone the repo and install the dependencies:
bashgit clone https://github.com/Aatif34/project_aatif_mashkoor.git
cd project_aatif_mashkoor
pip install torch torchvision opencv-python numpy

How to run predictions
The predict function loads the trained weights automatically from the checkpoints folder. Just pass it a list of image paths:
pythonfrom predict import classify_food

image_paths = ["data/pizza/img1.jpg", "data/sushi/img1.jpg"]
predictions = classify_food(image_paths)
print(predictions)
That's it. No extra setup needed as long as final_weights.pth is in the checkpoints folder.
