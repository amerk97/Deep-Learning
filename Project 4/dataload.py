import torch
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
from stacked_mnist import DataMode, StackedMNISTData

#Make method to get data from stacked_mnist.py into a pytorch dataloader
#Lager et objekt, arver Dataset metoder. Må gå gjennom numpy også, og pytorch bruker egen tensorgreie
#Keras DS: [60 000, 28, 28, 1] --> Pytorch DS: [60 000, 1, 28, 28]


class DataGetter(torch.utils.data.Dataset):
    def __init__(self, img, lab):
        self.img = img
        self.lab = lab

    def __len__(self):
        return self.lab.shape[0]

    def __getitem__(self, i):
        return self.img[i], self.lab[i]

    #Need the len, getitem unsure


def get_pyloader(ds, train):
    img, lab = ds.get_full_data_set(training = train)
    #img = np.moveaxis(img, -1, 1)
    ds = DataGetter(np.moveaxis(img, -1, 1), lab)
    return torch.utils.data.DataLoader(ds, shuffle=True, batch_size=256)


