import torch
from torchvision import datasets, transforms

data_folder="/home/neeraj/Desktop/rob_fergus_computer_vison/data"

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.08, 1.)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(data_folder+'/train', transform=data_transform)
print(len(train_dataset))
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=5000,
                                          shuffle=False, num_workers=32)

mean = 0
std = 0
nb_samples = 0
for data in trainloader:
    batch_samples = data[0].size(0)
    print(batch_samples)
    data = data[0].view(batch_samples, data[0].size(1), -1)
    print("data shape", data.shape)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples
    print("nb_samples", nb_samples)
print("mean",mean/nb_samples)
print("std", std/nb_samples)