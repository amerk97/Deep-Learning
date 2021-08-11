import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim


class Encoder(torch.nn.Module):

    def __init__(self, lat_size, img_size):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(img_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, lat_size)
        )

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):

    def __init__(self, lat_size, img_size):
        super().__init__()

        self.dec = nn.Sequential(
            nn.Linear(lat_size, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, img_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dec(x)


class AutoEncoder(nn.Module):

    def __init__(self, lat_size, img_size, loss, opti, lr):
        super().__init__()

        self.encoder = Encoder(lat_size, img_size)
        self.decoder = Decoder(lat_size, img_size)

        if loss == 'MSE':
            self.crit = nn.MSELoss()
        elif loss == 'CE':
            self.crit = nn.CrossEntropyLoss()

        if opti == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif opti == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):

        return self.decoder(self.encoder(x))


class Head(nn.Module):
    def __init__(self, classes, lat_size):
        super().__init__()

        self.clfhead = nn.Sequential(
            nn.Linear(lat_size, classes)#
            #nn.Softmax(dim=1) #Trenger kanskje strengt tatt ikke den, CE computer vel logsoftmax?
        )

    def forward(self, x):
        return self.clfhead(x)


class Classifier(nn.Module):
    def __init__(self, classes, lat_size, img_size, loss, opti, lr, new=False):
        super().__init__()

        self.head = Head(classes, lat_size)

        if new == True:
            self.enco = Encoder(lat_size, img_size)
        elif new == False:
            loaded = Encoder(lat_size, img_size)
            loaded.load_state_dict(torch.load('enc_trained.pth'))
            self.enco = loaded

        if loss == 'MSE':
            self.crit = nn.MSELoss()
        elif loss == 'CE':
            self.crit = nn.CrossEntropyLoss()

        if opti == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif opti == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        return self.head(self.enco(x))






