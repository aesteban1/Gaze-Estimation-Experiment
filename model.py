import cv2 as cv
import numpy as np
import pandas as pd
import glob
from glob import glob
import shutil
import os
import torch
from torch import nn
import torchvision
from torchvision import models
from facenet_pytorch import MTCNN
import torchvision.transforms as T
from torchvision.transforms import ToTensor
from PIL import Image
from IPython.display import display
from skimage.io import imread
from skimage.io import imshow
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

cwd = os.getcwd()
root_mpi_dir = os.path.join(cwd, 'Data\MPIIGaze')
Data_mpi_dir = os.path.join(root_mpi_dir, 'Data')
img_dir = os.path.join(Data_mpi_dir, 'Original')
ann_dir = os.path.join(root_mpi_dir, 'Annotation Subset')

def read_annot(in_path):
    r_dir = os.path.splitext(os.path.basename(in_path))[0]
    c_df = pd.read_table(in_path, header = None, sep = ' ')
    c_df.columns = ['path' if i<0 else ('x{}'.format(i//2) if i % 2 == 0 else 'y{}'.format(i//2)) for i, x in enumerate(c_df.columns, -1)]
    c_df['path'] = c_df['path'].map(lambda x: os.path.join(img_dir, r_dir, x))
    c_df['group'] = r_dir
    c_df['exists'] = c_df['path'].map(os.path.exists)
    return c_df

all_annot_df = pd.concat([read_annot(c_path) for c_path in glob(os.path.join(ann_dir, '*'))], ignore_index=True)
print(all_annot_df.shape[0], 'annotations')
print('Missing %2.2f%%' % (100-100*all_annot_df['exists'].mean()))
all_annot_df = all_annot_df[all_annot_df['exists']].drop('exists', 1)

train_df, test_df = train_test_split(all_annot_df, test_size=0.99, random_state=42)


from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        image_path = sample['path']
        image = Image.open(image_path).convert('RGB')

        eye_corners = sample[['x0', 'y0', 'x1', 'y1']].values.astype(float)

        if self.transform:
            image = self.transform(image)

        return image, eye_corners
    
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

custom_dataset = CustomDataset(data=train_df, transform=transform) #input training data to CustomDataset and apply transformations

# Define DataLoader and input CustomDataset to DataLoader
batch_size = 32
shuffle = True
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)


class ObjectLocalizationCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ObjectLocalizationCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 220, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(220, 216, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(216*32*159, 216), 
            nn.ReLU(inplace=True),
            nn.Linear(216, 2)
        )

        resNet18 = models.resnet18(pretrained=True)
        self.resnet_conv = nn.Sequential(*list(resNet18.children())[:-2])

        self.localization = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling to get 1x1x512
            nn.Flatten(),
            nn.Linear(512, 256),  # 512 number of output channels from ResNet18
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

        self.localization[2].weight.data.normal_(0, 0.01)  # Initialize Linear layers
        self.localization[2].bias.data.fill_(0.0)
        self.localization[4].weight.data.normal_(0, 0.01)
        self.localization[4].bias.data.fill_(0.0)

        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x_custom = self.features(x)
        print(x_custom.shape)

        x_resnet = self.resnet_conv(x)
        print(x_resnet.shape)

        x_resnet = x_resnet.view(-1, 512)
        x_combined = torch.cat((x_custom, x_resnet), dim=1)
        
        # Localization branch
        localization = self.localization(x_combined)

        # Classification branch
        classification = self.fc(x_combined)
        
        return classification, localization

model = ObjectLocalizationCNN(num_classes=4)
model.train()

# loss function and optimizer
criterion_cls = nn.MSELoss()
criterion_loc = nn.SmoothL1Loss()  # Loss function for eye corner regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for inputs, targets in tqdm(dataloader):
    inputs = inputs.to(device)
    targets = targets.to(device)

    eye_corners = targets
    
    # Zero the gradients
    optimizer.zero_grad()
    
    print(model.features)
    class_scores, bbox_coords = model(inputs)
    
    # classification and localization losses
    cls_loss = criterion_cls(class_scores, targets) 
    loc_loss = criterion_loc(bbox_coords, eye_corners) 
    
    total_loss = cls_loss + loc_loss
    
    # Backpropagation and update weights
    total_loss.backward()
    optimizer.step()