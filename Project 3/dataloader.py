import torch
from torchvision import datasets, transforms
from torch.utils import data
import matplotlib.pyplot as plt
import math



class Loader:

    def __init__(self, dataset, D1fraction, D2trainfract, D2valfract, bs=100):
        trans = transforms.ToTensor()
        gray = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

        if(dataset == 'MNIST'):
            self.datafull = datasets.MNIST(root="./data", train=True, download=True, transform=trans)
        elif(dataset == 'FMNIST'):
            self.datafull = datasets.FashionMNIST(root="./data", train=True, download=True, transform=trans)
        elif(dataset == 'CIFAR10'):
            self.datafull = datasets.CIFAR10(root="./data", train=True, download=True, transform=gray)
        elif(dataset == 'KMNIST'):
            self.datafull = datasets.KMNIST(root="./data", train=True, download=True, transform=trans)

        self.h, self.w = self.datafull[0][0].size()[1:]
        self.imgsize = int(self.h*self.w)
        self.classes = len(self.datafull.classes)

        #Split to D1 and D2
        self.D1, self.D2 = data.random_split(
            self.datafull, [math.floor(len(self.datafull)*D1fraction), math.ceil(len(self.datafull)*(1-D1fraction))]
        )

        #D1 split to Train and Val:
        self.D1_train, self.D1_val = data.random_split(self.D1, [math.floor(len(self.D1)*0.8), math.ceil(len(self.D1)*0.2)])

        #D2 split to Train, Val, Test:
        self.D2_train, self.D2_val, self.D2_test = data.random_split(
            self.D2, [math.floor(len(self.D2)*D2trainfract), math.ceil(len(self.D2)*D2valfract),
                      math.ceil(len(self.D2)*(1-D2valfract-D2trainfract))]
        )

        #All loaders;
        self.load_D1_train = data.DataLoader(dataset=self.D1_train, batch_size=bs, shuffle=True)
        self.load_D1_val = data.DataLoader(dataset=self.D1_val, batch_size=bs, shuffle=True)

        self.load_D2_train = data.DataLoader(dataset=self.D2_train, batch_size=bs, shuffle=True)
        self.load_D2_val = data.DataLoader(dataset=self.D2_val, batch_size=bs, shuffle=True)
        self.load_D2_test = data.DataLoader(dataset=self.D2_test, batch_size=bs, shuffle=True)


    def iterator(self, loader):
        diter = iter(loader)
        imgs, labs = diter.next()

        return imgs, labs
