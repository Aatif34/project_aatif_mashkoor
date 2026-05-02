import torch.nn as nn
from torchvision import models
from config import num_classes

class FoodResNetModel(nn.Module):
    def __init__(self):
        super(FoodResNetModel, self).__init__()
        
        # Load the pretrained ResNet50 model for transfer learning 
        # Using the DEFAULT weights gives us the best available ImageNet pretraining
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze the parameters of the earlier layers 
        # This ensures we are only training the final classification layer to save time and compute
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # The original ResNet50 outputs 1000 classes. 
        # We need to replace the final fully connected (fc) layer to output our 25 classes[cite: 87, 104].
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        # Pass the input through the modified ResNet50
        return self.resnet(x)
