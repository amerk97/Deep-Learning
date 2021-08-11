import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from stacked_mnist import StackedMNISTData, DataMode
from verification_net import VerificationNet
import matplotlib.pyplot as plt
import dataload
import visuals
import os
import numpy as np
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
            nn.Flatten()
        )

        self.z_mean = nn.Linear(8*4*4, lat_size)
        self.log_var = nn.Linear(8*4*4, lat_size)

    def forward(self, x):
        x = self.enc(x)
        z_mean, z_log_var = self.z_mean(x), self.log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded, z_mean, z_log_var

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1))
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z


class Decoder(nn.Module):
    def __init__(self, n_channels, lat_size):
        super().__init__()

        self.lin = nn.Sequential(
            nn.Linear(lat_size, 8 * 4 * 4),
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


class VAE(nn.Module):
    def __init__(self, n_channels, lat_size, lr):
        super().__init__()

        self.encoder = Encoder(n_channels, lat_size)
        self.decoder = Decoder(n_channels, lat_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, tr=False):
        encoded, mean, log_var = self.encoder(x)
        if tr:
            return self.decoder(encoded), mean, log_var

        return self.decoder(encoded)


    @staticmethod
    def loss_function(input, recon, mu, logvar):

        # recon loss:
        BCE = F.binary_cross_entropy(recon, input, reduction='sum')

        # KL-divergence loss:
        KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

        return BCE + KLD


def main():
    # Get data
    mono_binary_complete = StackedMNISTData(DataMode.MONO_BINARY_COMPLETE)  # Vanlig MNIST, float, med alle klasser
    mono_binary_missing = StackedMNISTData(DataMode.MONO_BINARY_MISSING)  # Vanlig MNIST, float, uten klasse 8

    color_binary_complete = StackedMNISTData(DataMode.COLOR_BINARY_COMPLETE)  # SMNIST, float, alle klasser
    color_binary_missing = StackedMNISTData(DataMode.COLOR_BINARY_MISSING)  # SMNIST, float, uten 8

    # Get the verification net to be used for quality and coverage. Trained if needed, uses saved model otherwise:
    verification = VerificationNet(force_learn=False)
    verification.train(mono_binary_complete, epochs=3)

    # Get training data to a pytorch loader and train the standard AE, or load a saved model. Train and save a good model:
    mfc = dataload.get_pyloader(mono_binary_complete, train=True)
    vae = VAE(1, 100, 0.001)

    load = True
    if load:
        vae.load_state_dict(torch.load('models_vae/vae_1.pth'))
    else:
        vae.train()
        vae_epochs = 30
        tlosses = []
        for epoch in range(vae_epochs):
            trainloss, nb = 0, 0
            for (img, _) in mfc:
                nb += 1
                img = img.float()
                recon, mean, log_var = vae(img, tr=True)

                loss = vae.loss_function(img, recon, mean, log_var)
                trainloss += loss

                vae.optimizer.zero_grad()
                loss.backward()
                vae.optimizer.step()

            tlosses.append(trainloss / nb)
            print('Epoch: %i, Training loss: %.3f' % (epoch + 1, trainloss / nb))

        plt.plot(tlosses, label='training loss')
        plt.title('Variational AutoEncoder learning')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.show()
        plt.close()

        # Save trained AE model. Decoder can be accessed specifically
        torch.save(vae.state_dict(), 'models_vae/vae_1.pth')

    # Check accuracy and predictability for reconstructions, and plot some reconstructions.
    # Also, generate images, get coverage for this and plot some of the images
    with torch.no_grad():
        vae.eval()
        visuals.recons_report_display(mono_binary_complete, vae, verification)  # This gives recons + pred, acc
        visuals.generate_report_display(100, vae, verification)  # This generates and plots + coverage, pred

    # AE as Anomaly detector. Train the autoencoder with missing 8 data:
    mfm_train = dataload.get_pyloader(mono_binary_missing, train=True)
    anomaly_vae = VAE(1, 100, 0.001)

    load = True
    if load:
        anomaly_vae.load_state_dict(torch.load('models_vae/anomaly_vae_1.pth'))
    else:
        anomaly_vae.train()
        anomaly_vae_epochs = 30
        tlosses = []
        for epoch in range(anomaly_vae_epochs):
            trainloss, nb = 0, 0
            for (img, _) in mfm_train:
                nb += 1
                img = img.float()
                recon, mean, log_var = anomaly_vae(img, tr=True)
                loss = anomaly_vae.loss_function(img, recon, mean, log_var)
                trainloss += loss

                anomaly_vae.optimizer.zero_grad()
                loss.backward()
                anomaly_vae.optimizer.step()

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
        torch.save(anomaly_vae.state_dict(), 'models_vae/anomaly_vae_1.pth')

    # Get results for anomaly vae:
    with torch.no_grad():
        anomaly_vae.eval()
        visuals.recons_report_display(mono_binary_missing, anomaly_vae, verification)
        visuals.generate_report_display(100, anomaly_vae, verification)

    # Get top-k anomalous examples from test data
    mfm_test = dataload.get_pyloader(mono_binary_missing, train=False)

    with torch.no_grad():
        anomaly_vae.eval()
        # Calculate the reconstruction error

        recons = None
        origs = None
        losses = None
        for (img, _) in mfm_test:
            img = img.float()
            recs = anomaly_vae(img)

            # re_loss = anomaly_vae.crit(recs, img)
            # print(re_loss.item())

            # Calculate MSE to get for every image instead of batch mean with item.
            re_loss = ((img - recs) ** 2).sum(axis=(1, 2, 3)) / 256

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

        # Get the N (6 here) largest indices, plot
        indices = losses.argsort()[-1:-6:-1]
        fig = plt.figure(figsize=(6, 2))

        for i, ind in enumerate(indices):
            ax1 = fig.add_subplot(2, 6, i + 1)
            plt.imshow(origs[ind])
            ax2 = fig.add_subplot(2, 6, 6 + i + 1)
            plt.imshow(recons[ind])
        ax1.set_title('Original', horizontalalignment='center')
        ax2.set_title('Recon', horizontalalignment='center')

        plt.show()
        plt.close()


if __name__ == '__main__':
    main()



