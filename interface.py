# Standardized imports for the grading script

from model import FoodResNetModel as TheModel
from train import train_model as the_trainer
from predict import classify_food as the_predictor
from dataset import Food101Dataset as TheDataset
from dataset import get_food_loader as the_dataloader

from config import batch_size as the_batch_size
from config import num_epochs as total_epochs
