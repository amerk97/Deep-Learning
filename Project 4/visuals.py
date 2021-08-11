import matplotlib.pyplot as plt
from verification_net import VerificationNet
import dataload
import numpy as np
import torch
import matplotlib.image as mpimg

def recons_report_display(ds, trained_AE, trained_vn):
    # Check predictability and accuracy of reconstructions using test data
    imgs, labs = ds.get_full_data_set(training=False)
    imgs = np.moveaxis(imgs, -1, 1)
    imgs = torch.Tensor(imgs).float()
    recons = trained_AE(imgs)
    #print(recons.shape)

    recons = np.moveaxis(recons.detach().double().numpy(), 1, -1)

    pred, acc = trained_vn.check_predictability(data=recons, correct_labels=labs)
    print('Predictability: %.3f, Accuracy: %.3f' % (pred, acc))

    # Run some images thorugh trained model and plot recons them using test data
    imgs, labs = ds.get_random_batch(training=False, batch_size=6)
    imgs = np.moveaxis(imgs, -1, 1)
    imgs = torch.Tensor(imgs).float()

    oris = []
    res = []
    recon = trained_AE(imgs)
    n=0
    for re in recon:
        oris.append(imgs[n].detach().numpy().squeeze())
        res.append(re.detach().numpy().squeeze())
        n+=1

    fig = plt.figure(figsize=(6, 2))
    for i, item in enumerate(oris):
        #plt.subplot(2, n_recons, i + 1)
        ax1 = fig.add_subplot(2, 6, i+1)
        plt.imshow(item)
    ax1.set_title('Original', horizontalalignment='center')

    for i, item in enumerate(res):
        #plt.subplot(2, n_recons, 10 + i + 1)
        ax2 = fig.add_subplot(2, 6, 6+i+1)
        plt.imshow(item)
    ax2.set_title('Reconstructed', horizontalalignment = 'center')

    plt.show()
    plt.close()


def generate_report_display(lat_size, trained_AE, trained_vn):
    # Check predictability and coverage for the generator:
    # Generate a "large" number of examples and run like batches

    p_tot, cov_tot = 0, 0
    for n in range(10):
        z = torch.Tensor(np.random.randn(256, lat_size)).float()
        gened = trained_AE.decoder(z) #Gir ut [256, 1, 28, 28]
        gened_tf = np.moveaxis(gened.detach().double().numpy(), 1, -1)

        pred, acc = trained_vn.check_predictability(data=gened_tf)
        cov = trained_vn.check_class_coverage(data=gened_tf)

        p_tot += pred * 1/10
        cov_tot += cov * 1/10

    print('Predictability: %.3f, Coverage: %.3f' % (p_tot, cov_tot))

    # Generate and plot some images:

    z = np.random.randn(9, lat_size)
    z = torch.Tensor(z).float()
    generated = trained_AE.decoder(z)
    generated = generated.squeeze()

    fig = plt.figure(figsize=(3, 3))
    for i, item in enumerate(generated):
        #plt.subplot(2, n_recons, i + 1)
        fig.add_subplot(3, 3, i+1)
        plt.imshow(item)
    plt.show()
    plt.title('Generated images')