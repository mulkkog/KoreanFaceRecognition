import torch
from female_dataset import FemaleDataset
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

batch_size = 64
device = 'cuda'
train_csv_path = '../dataset/label_train.csv'
test_csv_path = '../dataset/label_test.csv'
train_image_path = '../dataset/train'
test_image_path = '../dataset/test'

#InceptionResnet = InceptionResnetV1(pretrained='casia-webface', device='cuda:0', classify=True, num_classes=5).eval()
#InceptionResnet = torch.load('/home/bgkim/project/KoreanFaceRecognition/checkpoint/best_model_b64_re160.pt')
num_classes = 5
feature_extract = True
Resnet = models.resnet50(pretrained= True).eval()
Resnet.fc = nn.Linear(2048, num_classes)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(InceptionResnet.parameters(), lr=0.001, momentum=0.8)#optim.Adam(InceptionResnet.parameters(),lr=0.001, betas=(0.0 , 0.99))
#InceptionResnet.to(device)
optimizer = optim.SGD(Resnet.parameters(), lr=0.001, momentum=0.8)#optim.Adam(InceptionResnet.parameters(),lr=0.001, betas=(0.0 , 0.99))
Resnet.to(device)
criterion.to(device)
epoch = 100
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

ct = 0
'''
for child in InceptionResnet.children():
    ct += 1
    if ct < 5:
        for param in child.parameters():
            param.requires_grad = False
'''


test_loader = torch.utils.data.DataLoader(
    FemaleDataset(test_csv_path, test_image_path), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
)
train_loader = torch.utils.data.DataLoader(
    FemaleDataset(train_csv_path, train_image_path), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
)

def train(dataloader, model, criterion, optimizer, i, writer):
    model.train()
    # scheduler.step()
    total_acc = 0
    for index, (data, label) in enumerate(dataloader):
        typed = 'Train'
        data = Variable(data.to(device))
        label = Variable(label.to(device))
        label -= 1
        y = model(data)
        loss = criterion(y, label)  # label : [32, 1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        y = torch.argmax(y, dim=1) # [0.3 , 0.5] -> 1
        correct = (y == label).float().sum()
        acc = correct / y.shape[0]
        total_acc += acc
        print("[{}] Epoch {}/{}, Batch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(typed, i + 1, epoch, index + 1,
                                                                                     len(dataloader), loss.data,
                                                                                     acc))
    writer.add_scalar('train_accuracy', total_acc / len(dataloader), i+1)
    return total_acc / len(dataloader)


def test(dataloader, model, criterion, i, writer):
    typed = "Test"
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for index, (data, label) in enumerate(dataloader):
            data = Variable(data.to(device))
            label = Variable(label.to(device))
            label -= 1

            y = model(data)
            loss = criterion(y, label)

            # Accuracy
            y = torch.argmax(y, dim=1)  # [0.3 , 0.5] -> 1
            correct = (y == label).float().sum()
            acc = correct / y.shape[0]
            print("[{}] Epoch {}/{}, Batch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(typed, i + 1, epoch, index + 1,
                                                                                         len(dataloader), loss.data,
                                                                                         acc))
            total_acc += acc

        writer.add_scalar('test_accuracy', total_acc / len(dataloader), i+1)
        return total_acc / len(dataloader)

best_acc = 0
writer = SummaryWriter()
'''
for i in range(epoch):
    train(train_loader, InceptionResnet, criterion, optimizer, i, writer)
    cur_acc = test(test_loader, InceptionResnet, criterion, i, writer)
    print("Current Accuracy: ", cur_acc)
    if cur_acc > best_acc:
        torch.save(InceptionResnet, './checkpoint/female_model_2021_01_15_casia-webface.pt')
        best_acc = cur_acc
        print('saved')
'''
for i in range(epoch):
    train(train_loader, Resnet, criterion, optimizer, i, writer)
    cur_acc = test(test_loader, Resnet, criterion, i, writer)
    print("Current Accuracy: ", cur_acc)
    if cur_acc > best_acc:
        torch.save(Resnet, './checkpoint/female_model_2021_01_18_resnet50.pt')
        best_acc = cur_acc
        print('saved')

print(best_acc)
writer.close()