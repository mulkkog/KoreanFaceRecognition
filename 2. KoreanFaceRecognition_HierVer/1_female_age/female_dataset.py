import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

class FemaleDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_path):
        self.root_path = image_path
        self.file = pd.read_csv(csv_path)
        self.female = self.file[self.file['sex'].isin([1])]

        self.label = self.female.age.tolist()
        self.image_paths = self.female.file_path.tolist()

        self.len = len(self.image_paths)
        self.transform = transforms.Compose([
                transforms.Resize(160),
                #transforms.CenterCrop(128),
                transforms.ToTensor(),
        ])
        print(self.female)
        print(self.len)
    def __getitem__(self, index):
        x = Image.open(os.path.join(self.root_path, self.image_paths[index]))
        return self.transform(x), self.label[index]

    def __len__(self):
        return self.len

