import torch
import torch.nn as nn
import torch.optim as optim
from stacked_mnist import DataMode, StackedMNISTData
import torch.utils.data
from verification_net import VerificationNet
import matplotlib.pyplot as plt
import dataload
import visuals
import os
import numpy as np
from tensorflow import keras
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Encoder(nn.Module):

    def __init__(self, n_channels, lat_size):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(n_channels, 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*4*4, lat_size)
        )

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    def __init__(self, n_channels, lat_size):
        super().__init__()

        self.lin = nn.Sequential(
            nn.Linear(lat_size, 8*4*4),
            nn.ReLU()
        )

        self.con = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2, n_channels, 6, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.lin(x)
        x = x.view((x.shape[0], 8, 4, 4))
        return self.con(x)


class AutoEncoder(nn.Module):
    def __init__(self, n_channels, lat_size, loss, lr):
        super().__init__()

        self.encoder = Encoder(n_channels, lat_size)
        self.decoder = Decoder(n_channels, lat_size)

        if loss == 'MSE':
            self.crit = nn.MSELoss()
        elif loss == 'BCE': #If using binary mode
            self.crit = nn.BCELoss()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def main():
    # Get all needed datamodes for the experiments
    mono_float_complete = StackedMNISTData(DataMode.MONO_FLOAT_COMPLETE) # Vanlig MNIST, float, med alle klasser
    mono_float_missing = StackedMNISTData(DataMode.MONO_FLOAT_MISSING) # Vanlig MNIST, float, uten klasse 8

    color_float_complete = StackedMNISTData(DataMode.COLOR_FLOAT_COMPLETE) # SMNIST, float, alle klasser
    color_float_missing = StackedMNISTData(DataMode.COLOR_FLOAT_MISSING) # SMNIST, float, uten 8

    # Get the verification net to be used for quality and coverage. Trained if needed, uses saved model otherwise:
    verification = VerificationNet(force_learn=False)
    verification.train(mono_float_complete, epochs=3)

    # Get training data to a pytorch loader and train the standard AE, or load a saved model. Train and save a good model:
    mfc = dataload.get_pyloader(mono_float_complete, train=True)
    AE = AutoEncoder(1, 100, 'MSE', 0.001)

    load = True
    if load:
        AE.load_state_dict(torch.load('models_ae/autoencoder_1.pth'))
    else:
        AE.train()
        ae_epochs = 25
        tlosses = []
        for epoch in range(ae_epochs):
            trainloss, nb = 0, 0
            for (img, _) in mfc:
                nb += 1
                img = img.float()
                recon = AE(img)
                loss = AE.crit(recon, img)
                trainloss += loss.item()

                AE.optimizer.zero_grad()
                loss.backward()
                AE.optimizer.step()

            tlosses.append(trainloss / nb)
            print('Epoch: %i, Training loss: %.3f' % (epoch + 1, trainloss / nb))

        plt.plot(tlosses, label='training loss')
        plt.title('AutoEncoder learning')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.show()
        plt.close()

        # Save trained AE model. Decoder can be accessed specifically
        torch.save(AE.state_dict(), 'models_ae/autoencoder_1.pth')

    # Check accuracy and predictability for reconstructions, and plot some reconstructions.
    # Also, generate images, get coverage for this and plot some of the images
    with torch.no_grad():
        AE.eval()
        visuals.recons_report_display(mono_float_complete, AE, verification) #This gives recons + pred, acc
        visuals.generate_report_display(100, AE, verification) #This generates and plots + coverage, pred



    # AE as Anomaly detector. Train the autoencoder with missing 8 data:
    mfm_train = dataload.get_pyloader(mono_float_missing, train=True)
    anomaly_AE = AutoEncoder(1, 100, 'MSE', 0.001)

    load = True
    if load:
        anomaly_AE.load_state_dict(torch.load('models_ae/anomaly_autoencoder_1.pth'))
    else:
        anomaly_AE.train()
        anomaly_ae_epochs = 25
        tlosses = []
        for epoch in range(anomaly_ae_epochs):
            trainloss, nb = 0, 0
            for (img, _) in mfm_train:
                nb += 1
                img = img.float()
                recon = anomaly_AE(img)
                loss = anomaly_AE.crit(recon, img)
                trainloss += loss.item()

                anomaly_AE.optimizer.zero_grad()
                loss.backward()
                anomaly_AE.optimizer.step()

            tlosses.append(trainloss / nb)
            print('Epoch: %i, Training loss: %.3f' % (epoch + 1, trainloss / nb))

        plt.plot(tlosses, label='training loss')
        plt.title('Anomaly AutoEncoder learning')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.show()
        plt.close()

        # Save trained AE model. Decoder can be accessed specifically
        torch.save(anomaly_AE.state_dict(), 'models_ae/anomaly_autoencoder_1.pth')

    # Get results for anomaly autoencoder:
    with torch.no_grad():
        anomaly_AE.eval()
        visuals.recons_report_display(mono_float_missing, anomaly_AE, verification)
        visuals.generate_report_display(100, anomaly_AE, verification)


    # Get top-k anomalous examples from test data
    mfm_test = dataload.get_pyloader(mono_float_missing, train=False)

    with torch.no_grad():
        anomaly_AE.eval()
        # Calculate the reconstruction error

        recons = None
        origs = None
        losses = None
        for (img, _) in mfm_test:
            img = img.float()
            recs = anomaly_AE(img)

            #re_loss = anomaly_AE.crit(recs, img)
            #print(re_loss.item())

            # Calculate MSE to get for every image instead of batch mean with item. Check that the calculation is correct
            re_loss = ((img - recs)**2).sum(axis=(1, 2, 3)) / 256

            if losses is None:
                losses = re_loss
            else:
                losses = torch.cat([losses, re_loss])

            if recons is None:
                recons = recs
            else:
                recons = torch.cat([recons, recs])

            if origs is None:
                origs = img
            else:
                origs = torch.cat([origs, img])


        losses = losses.detach().numpy()
        recons = recons.detach().numpy().squeeze()
        origs = origs.detach().numpy().squeeze()

        # Get the N largest indices, plot
        indices = losses.argsort()[-1:-6:-1]
        fig = plt.figure(figsize=(6, 2))

        for i, ind in enumerate(indices):
            ax1 = fig.add_subplot(2, 6, i+1)
            plt.imshow(origs[ind])
            ax2 = fig.add_subplot(2, 6, 6+i+1)
            plt.imshow(recons[ind])
        ax1.set_title('Original', horizontalalignment='center')
        ax2.set_title('Recon', horizontalalignment='center')

        plt.show()
        plt.close()


if __name__ == '__main__':
    main()


