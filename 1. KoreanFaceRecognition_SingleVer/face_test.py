import os
import torch
from dataset import Dataset
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

batch_size = 32
num_workers = False
device = 'cuda'
csv_path = '/home/bgkim/dataset/yap_dataset/dataset/label_male.csv'
image_path = '/home/bgkim/dataset/yap_dataset/dataset/'
#InceptionResnet = InceptionResnetV1(pretrained='vggface2', device='cuda:0', classify=True, num_classes=5).eval() #casia-webface
InceptionResnet = torch.load("male_model.pt")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(InceptionResnet.parameters(), lr=0.001, momentum=0.8)#optim.Adam(InceptionResnet.parameters(),lr=0.001, betas=(0.0 , 0.99))
InceptionResnet.to(device)

criterion.to(device)

epoch = 1
avg_loss = 0
avg_acc = 0


dataloader = torch.utils.data.DataLoader(
    Dataset(csv_path, image_path), batch_size=batch_size, shuffle=True, num_workers=num_workers
)

'''
def get_infinite_batches(data_loader):
    while True:
        for i, (images, label) in enumerate(data_loader):
            yield i, images, label
'''
train_num = round(len(dataloader) * 0.7)
test_num = round(len(dataloader) * 0.3)

#batch_data = get_infinite_batches(dataloader)
# index, data, label = batch_data.__next__()


def test(data, label):
    for p in InceptionResnet.parameters():
        p.requires_grad = False
    y = InceptionResnet(data)
    loss = criterion(y, label)
    return loss, y

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    #inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

for i in range(epoch):
    for index, (data, label) in enumerate(dataloader):
        typed = None
        #imshow(data[0]) #data ??? ???????????? ??????
        data = Variable(data.to(device))
        label = Variable(label.to(device))
        label -= 1 #0 , 1

        loss, output = test(data, label)
        typed = 'Test'
        # Accuracy
        #print(output)
        output = torch.argmax(output, dim=1) # [0.3 , 0.5] -> 1
        correct = (output == label).float().sum()

        avg_loss += loss.data
        avg_acc += correct / output.shape[0]
        print("[{}] Epoch {}/{}, Batch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(typed,i+1, epoch, index, correct / output.shape[0], loss.data, correct / output.shape[0]))

print("[Average] Loss {}, Accuracy {}".format(avg_loss/70,avg_acc/70))