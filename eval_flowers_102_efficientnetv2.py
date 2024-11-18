import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader
import timm  # Timm has a bunch of models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing parameters
IMG_SIZE = 224
BATCH_SIZE = 32

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # mean for ImageNet
                         [0.229, 0.224, 0.225])   # std for ImageNet
])

# Load the Flowers-102 dataset
test_set = Flowers102(root='.', split='test', download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Load the EfficientNetV2 model
model = timm.create_model('tf_efficientnetv2_b0', pretrained=True)
model.eval()
model.to(device)

# Adjust the classifier to match the number of classes (102)
num_features = model.get_classifier().in_features
model.classifier = torch.nn.Linear(num_features, 102)
model.classifier.to(device)

# Since we haven't trained the classifier layer yet, it will have random weights
# For evaluation purposes, the predictions won't be meaningful without training
# However, we proceed to demonstrate the evaluation process

# Generate predictions and compute classification statistics
y_pred = []
y_true = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Classification report
print('Classification Report')
print(classification_report(y_true, y_pred, digits=4))

# Confusion matrix
print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
print(cm)
