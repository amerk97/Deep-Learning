import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from verification_net import VerificationNet
from stacked_mnist import StackedMNISTData, DataMode
import dataload
import visuals
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Generator(nn.Module):

    def __init__(self, n_channels, noise_size, lr, beta1):
        super().__init__()

        self.noise = noise_size # input layer size

        self.inputlayer = nn.Sequential(
            nn.Linear(noise_size, 8*4*4),
            nn.ReLU()
        )

        self.convs = nn.Sequential(

            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),

            nn.ConvTranspose2d(2, n_channels, 6, stride=2, padding=1),
            nn.Tanh()

            #[1, 28, 28] ut, altså bilder generert
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))
        self.crit = nn.BCELoss()

    def forward(self, x):
        x = self.inputlayer(x).view((x.shape[0], 8, 4, 4))
        return self.convs(x)


class Discriminator(nn.Module):

    def __init__(self, n_channels, lr, beta1):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(n_channels, 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(2, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Flatten(),

            nn.Linear(8*8*2, 1),
            nn.Sigmoid()
            # out  1 value to classify real or fake
        )

        self.crit = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))

    def forward(self, x):
        return self.convs(x)


def report_and_plot(trained_gen, trained_vnet, ep = None):
    #Do the Q&C check, and plot some examples of generated images, such as done in visuals for AE's
    p_tot, cov_tot = 0, 0
    for n in range(10):
        z = torch.Tensor(np.random.randn(256, 100)).float()
        gened = trained_gen(z)  # Gir ut [256, 1, 28, 28]
        gened_tf = np.moveaxis(gened.detach().double().numpy(), 1, -1)

        pred, acc = trained_vnet.check_predictability(data=gened_tf)
        cov = trained_vnet.check_class_coverage(data=gened_tf)

        p_tot += pred * 1 / 10
        cov_tot += cov * 1 / 10

    print('Predictability: %.3f, Coverage: %.3f' % (p_tot, cov_tot))

    # Generate and plot some images:

    z = np.random.randn(25, 100)
    z = torch.Tensor(z).float()
    generated = trained_gen(z)
    generated = generated.squeeze()

    fig = plt.figure(figsize=(5, 5))
    for i, item in enumerate(generated):
        # plt.subplot(2, n_recons, i + 1)
        fig.add_subplot(5, 5, i + 1)
        plt.imshow(item)
    plt.title('Generated images')

    if ep is not None:
        plt.savefig(f'results_gan/gen_epoch_{ep}.png')
    else:
        plt.show()


def main():
    # PARAMETERS
    in_channels = 1
    LR = 0.0002
    noise_size = 100
    b1 = 0.5
    epochs = 20
    load = False

    # Get all needed datamodes for the experiments
    mono_float_complete = StackedMNISTData(DataMode.MONO_FLOAT_COMPLETE)  # Vanlig MNIST, float, med alle klasser
    color_float_complete = StackedMNISTData(DataMode.COLOR_FLOAT_COMPLETE)  # SMNIST, float, alle klasser

    # Get the verification net to be used for quality and coverage. Trained if needed, uses saved model otherwise:
    verification = VerificationNet(force_learn=False)
    verification.train(mono_float_complete, epochs=3)

    # Get training data as a pytorch loader:
    mfc = dataload.get_pyloader(mono_float_complete, train=True)

    # Create Discriminator and Generator.
    generator = Generator(in_channels, noise_size, LR, b1)
    discriminator = Discriminator(in_channels, LR, b1)

    # Train the DCGAN:
    if load:
        generator.load_state_dict(torch.load('models_gan/generator_1.pth'))
        report_and_plot(generator, verification)
    else:
        for e in range(epochs):
            for index, (img, _) in enumerate(mfc):
                img = img.float()
                noise = torch.randn(img.shape[0], noise_size)
                fake = generator(noise)

                # Train discriminator:
                d_real = discriminator(img).reshape(-1)
                loss_dreal = discriminator.crit(d_real, torch.ones_like(d_real))
                d_fake = discriminator(fake).reshape(-1)
                loss_dfake = discriminator.crit(d_fake, torch.zeros_like(d_fake))

                loss_disc = (loss_dfake+loss_dreal) / 2

                discriminator.zero_grad()
                loss_disc.backward(retain_graph = True)
                discriminator.optimizer.step()

                # Train generator:
                out = discriminator(fake).reshape(-1)
                loss_gen = generator.crit(out, torch.ones_like(out))
                generator.zero_grad()
                loss_gen.backward()
                generator.optimizer.step()

                #Print loss now and then:
                if index % 100 == 0:
                    print('Epoch: %i, Batch: %i, Loss D: %.3f, Loss G: %.3f' % (e + 1, index+1, loss_disc, loss_gen))

            #Save some generated images for the trained epoch. Furthermore, get the Q&C values for the epoch.
            with torch.no_grad():
                generator.eval()
                report_and_plot(generator, verification, e)

            generator.train()

        # Save the generator model once finished training:
        torch.save(generator.state_dict(), 'models_gan/generator_1.pth')


if __name__ == '__main__':
    main()