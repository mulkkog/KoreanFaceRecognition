import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_path):
        self.file = pd.read_csv(csv_path)
        self.root_path = image_path
        self.image_paths = self.file.file_path
        self.label = self.file.sex
        self.len = len(self.file)
        self.transform = transforms.Compose([
                transforms.Resize(160),
                transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.root_path, self.image_paths[index]))
        return self.transform(x), self.label[index]

    def __len__(self):
        return self.len