import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN as mtcnn
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_path):
        self.file = pd.read_csv(csv_path)

        '''
        self.file = self.file.sort_values("sex", ascending=False)

        check = self.file.sort_values("sex", ascending=False).head(2214)
        print(check.head())
        print(check.tail())
        check2 = self.file.sort_values("sex", ascending=False).head(2215)
        print(check2.head())
        print(check2.tail())
        '''

        self.root_path = image_path
        self.image_paths = self.file.file_path
        self.label = self.file.age
        #self.len = 2214
        self.len = len(self.file)

        self.transform = transforms.Compose([
                transforms.Scale(128),
                #transforms.RandomCrop(128),
                #transforms.CenterCrop(128),
                transforms.ToTensor(),
            ])
        #self.data = self.data.values[:, 1:] / 255

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.root_path, self.image_paths[index]))
        return self.transform(x), self.label[index]

    def __len__(self):
        #return 2214
        return self.len

