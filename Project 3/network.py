import torch
import matplotlib.pyplot as plt
import dataloader
import models
from torch.utils import data
import configparser
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import seaborn as sn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_params():
    config = configparser.ConfigParser(empty_lines_in_values=False)
    try:
        config.read("config.txt")
    except Exception:
        print("Configuration file does not exist")
        print("Exiting...")
        exit()

    globals, autoencoder, classifier = {}, {}, {}

    for section in config.sections():
        d = {}

        for item in config.items(section):
            try:
                d[item[0]] = eval(item[1])
            except Exception as e:
                d[item[0]] = item[1]

        if (section == "GLOBAL"):
            globals = d
        elif (section == "AUTOENCODER"):
            autoencoder = d
        elif (section == "CLASSIFIER"):
            classifier = d

    return globals, autoencoder, classifier


def main():
    if os.path.exists('enc_trained.pth'):
        os.remove('enc_trained.pth')

    GLOB, AE, CLF = get_params()

#___TRAIN AUTOENCODER
    DataLoad = dataloader.Loader(GLOB['dataset'], GLOB['d1_frac'], GLOB['d2_trainfrac'], GLOB['d2_valfrac'])

    auto_encoder = models.AutoEncoder(GLOB['latent'], DataLoad.imgsize, AE['loss'], AE['optim'], AE['lr'])
    ae_train = DataLoad.load_D1_train
    ae_val = DataLoad.load_D1_val

    #TSNE 1:
    if GLOB['clusters']:
        d = data.DataLoader(dataset=DataLoad.D1, batch_size=1, shuffle=True)
        dat = []
        la = []
        i=0
        for (img, lab) in d:
            if i==100:
                break
            i+=1
            img = torch.flatten(img, start_dim=1)
            lat = auto_encoder.encoder(img)[0].detach().numpy()
            dat.append(lat)
            la.append(lab)

        m = TSNE(n_components=2)
        tsne_data = m.fit_transform(dat)
        tsne_data = np.vstack((tsne_data.T, la)).T
        tsne_df = pd.DataFrame(data=tsne_data, columns=('dim1', 'dim2', 'labels'))

        sn.FacetGrid(tsne_df, hue='labels', height=6, aspect=1.5).map(plt.scatter, 'dim1', 'dim2')
        plt.show()
        plt.close()


    ae_epochs = AE['epochs']
    tlosses = []
    vlosses = []
    for epoch in range(ae_epochs):
        auto_encoder.train()
        trainloss, valloss, nb, nbt = 0, 0, 0, 0
        for (img, _) in ae_train:
            nb+=1
            img = img.reshape(-1, DataLoad.imgsize)
            recon = auto_encoder(img)
            loss = auto_encoder.crit(recon, img)
            trainloss += loss.item()

            auto_encoder.optimizer.zero_grad()
            loss.backward()
            auto_encoder.optimizer.step()

        #print('Epoch ' + str(epoch+1) + ', Train loss: ' + str(trainloss/nb))
        tlosses.append(trainloss/nb)
        nbt=nb
        nb=0
        with torch.no_grad():
            auto_encoder.eval()
            for (img, _) in ae_val:
                nb+=1
                img = img.reshape(-1, DataLoad.imgsize)
                recon = auto_encoder(img)
                vloss = auto_encoder.crit(recon, img)
                valloss += vloss.item()

            vlosses.append((valloss / nb))
            #print('Epoch ' + str(epoch + 1) + ', Val loss: ' + str(valloss / nb))
        print('Epoch:%i, Training loss:%.3f Validation Loss: %.3f' % (epoch + 1, trainloss / nbt, valloss / nb))

    plt.plot(tlosses, label='training loss')
    plt.plot(vlosses, label = 'Val loss')
    plt.title('Autoencoder learning')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.show()
    plt.close()

    #Save Encoder module weights for SSL classifier
    encoder = auto_encoder.encoder
    torch.save(encoder.state_dict(), 'enc_trained.pth')

    #TSNE2:
    if GLOB['clusters']:
        d = data.DataLoader(dataset=DataLoad.D1, batch_size=1, shuffle=True)
        dat = []
        la = []
        i=0
        for (img, lab) in d:
            if i==100:
                break
            i+=1
            img = torch.flatten(img, start_dim=1)
            lat = auto_encoder.encoder(img)[0].detach().numpy()
            dat.append(lat)
            la.append(lab)

        m = TSNE(n_components=2)
        tsne_data = m.fit_transform(dat)
        tsne_data = np.vstack((tsne_data.T, la)).T
        tsne_df = pd.DataFrame(data=tsne_data, columns=('dim1', 'dim2', 'labels'))

        sn.FacetGrid(tsne_df, hue='labels', height=6, aspect=1.5).map(plt.scatter, 'dim1', 'dim2')
        plt.show()
        plt.close()

    #Run chosen number of random images in trained AE. Show the recons vs originals:
    n_recons = AE['recons']
    container = []
    re_rand = data.DataLoader(dataset=DataLoad.D2_test, batch_size=n_recons, shuffle=True) #Legg inn N fra config

    for (orig, _) in re_rand:
        orig = torch.flatten(orig, start_dim=1)
        recon = auto_encoder(orig)
        container.append((orig, recon))

    fig = plt.figure(figsize=(n_recons, 2))
    origs = container[0][0].detach().numpy()
    recons = container[1][1].detach().numpy()

    for i, item in enumerate(origs):
        #plt.subplot(2, n_recons, i + 1)
        ax1 = fig.add_subplot(2, n_recons, i+1)
        item = item.reshape(-1, DataLoad.h, DataLoad.w)
        plt.imshow(item[0])
    ax1.set_title('Original', horizontalalignment='center')

    for i, item in enumerate(recons):
        #plt.subplot(2, n_recons, 10 + i + 1)
        ax2 = fig.add_subplot(2, n_recons, n_recons+i+1)
        item = item.reshape(-1, DataLoad.h, DataLoad.w)
        plt.imshow(item[0])
    ax2.set_title('Reconstructed', horizontalalignment = 'center')

    plt.show()
    plt.close()




#___TRAIN CLASSIFIER - SEMI SUPERVISED:

    # Get data for both classifiers
    clf_train = DataLoad.load_D2_train
    clf_val = DataLoad.load_D2_val
    clf_test = DataLoad.load_D2_test

    clf1 = models.Classifier(DataLoad.classes, GLOB['latent'], DataLoad.imgsize, CLF['loss'], CLF['optim'], CLF['lr'], new=False)

    frz = CLF['freeze']

    clf1_epochs = CLF['epochs']
    cl1_trainloss = []
    cl1_valloss = []
    cl1_tacc = []
    cl1_vacc = []
    for epoch in range(clf1_epochs):
        clf1.train()

        if frz:
            for par in clf1.enco.parameters():
                par.requires_grad = False

        trainloss_c1 = 0
        valloss_c1 = 0
        tcorr = 0
        vcorr = 0
        tot = 0
        nb=0
        nbt=0
        total=0
        for imgs, labs in clf_train:
            nb+=1
            imgs = torch.flatten(imgs, start_dim=1)
            preds = clf1(imgs)

            _, predicted = torch.max(preds.data, 1)
            tot += labs.size(0)
            tcorr += (predicted == labs).sum().item()

            loss = clf1.crit(preds, labs)
            trainloss_c1 += loss.item()

            clf1.optimizer.zero_grad()
            loss.backward()
            clf1.optimizer.step()

        #print('Epoch ' + str(epoch + 1) + ', Training Loss: ' + str(trainloss_c1/nb))
        cl1_trainloss.append((trainloss_c1 / nb))
        #print('Epoch ' + str(epoch + 1) + ' Train Accuracy: ' + str(100*tcorr/tot))
        cl1_tacc.append((100*tcorr/tot))

        total = tot
        nbt = nb
        tot = 0
        nb=0
        with torch.no_grad():
            clf1.eval()
            for (imgs, labs) in clf_val:
                nb+=1
                imgs = torch.flatten(imgs, start_dim=1)
                preds = clf1(imgs)

                _, predicted = torch.max(preds.data, 1)
                tot += labs.size(0)
                vcorr += (predicted == labs).sum().item()

                vloss = clf1.crit(preds, labs)
                valloss_c1 += vloss.item()

            cl1_valloss.append((valloss_c1 / nb))
            #print('Epoch ' + str(epoch + 1) + ', Val Loss: ' + str(valloss_c1 / nb))
            #print('Epoch ' + str(epoch + 1) + ', Val Accuracy: ' + str(100 * vcorr / tot))
            print('Epoch:%i, Training loss:%.3f Validation Loss: %.3f Training Acc: %.3f, Validation Acc %.3f' % (epoch+1, trainloss_c1/nbt, valloss_c1/nb, tcorr/total, vcorr/tot))
            cl1_vacc.append((100 * vcorr / tot))

    plt.plot(cl1_trainloss, label='training loss')
    plt.plot(cl1_valloss, label='Val loss')
    plt.title('SSL Classifier learning')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.show()
    plt.close()

    plt.plot(cl1_tacc, label='Train accuracy')
    plt.plot(cl1_vacc, label='Val accuracy')
    plt.title('SSL Classifier learning')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    plt.close()

    #Teste med test dataen - trenger ikke plottes

    testcorr = 0
    total = 0
    with torch.no_grad():
        clf1.eval()
        for imgs, labs in clf_test:
            imgs = torch.flatten(imgs, start_dim=1)
            preds = clf1(imgs)

            _, predicted = torch.max(preds.data, 1)
            total += labs.size(0)
            testcorr += (predicted == labs).sum().item()

    print('Semi supervised - Test set Accuracy: ' + str(100 * testcorr / total))

    #TSNE3:
    if GLOB['clusters']:
        d = data.DataLoader(dataset=DataLoad.D1, batch_size=1, shuffle=True)
        dat = []
        la = []
        i=0
        for (img, lab) in d:
            if i==100:
                break
            i+=1
            img = torch.flatten(img, start_dim=1)
            lat = clf1.enco(img)[0].detach().numpy()
            dat.append(lat)
            la.append(lab)

        m = TSNE(n_components=2)
        tsne_data = m.fit_transform(dat)
        tsne_data = np.vstack((tsne_data.T, la)).T
        tsne_df = pd.DataFrame(data=tsne_data, columns=('dim1', 'dim2', 'labels'))

        sn.FacetGrid(tsne_df, hue='labels', height=6, aspect=1.5).map(plt.scatter, 'dim1', 'dim2')
        plt.show()
        plt.close()



#___TRAIN CLASSIFIER - FULL SUPERVISED:

    clf2 = models.Classifier(DataLoad.classes, GLOB['latent'], DataLoad.imgsize, CLF['loss'], CLF['optim'], CLF['lr'], new=True)

    clf2_epochs = CLF['epochs']
    cl2_trainloss = []
    cl2_valloss = []
    cl2_tacc = []
    cl2_vacc = []

    for epoch in range(clf2_epochs):
        clf2.train()
        nb=0
        trainloss_c2, valloss_c2 = 0,0
        tcorr = 0
        vcorr = 0
        tot = 0
        nbt = 0
        total = 0

        for imgs, labs in clf_train:
            nb+=1
            imgs = imgs.reshape(-1, DataLoad.imgsize)
            preds = clf2(imgs)

            _, predicted = torch.max(preds.data, 1)
            tot += labs.size(0)
            tcorr += (predicted == labs).sum().item()

            loss = clf2.crit(preds, labs)
            trainloss_c2 += loss.item()

            clf2.optimizer.zero_grad()
            loss.backward()
            clf2.optimizer.step()

        cl2_trainloss.append(trainloss_c2/nb)
        #print('Epoch ' + str(epoch + 1) + ', training loss: ' + str(trainloss_c2/nb))
        #print('Epoch ' + str(epoch + 1) + ', Train Accuracy: ' + str(100 * tcorr / tot))
        cl2_tacc.append((100 * tcorr / tot))
        total = tot
        tot = 0
        nbt = nb
        nb=0
        with torch.no_grad():
            clf2.eval()
            for (imgs, labs) in clf_val:
                nb+=1
                imgs = torch.flatten(imgs, start_dim=1)
                preds = clf2(imgs)

                _, predicted = torch.max(preds.data, 1)
                tot += labs.size(0)
                vcorr += (predicted == labs).sum().item()

                vloss = clf2.crit(preds, labs)
                valloss_c2 += vloss.item()

            cl2_valloss.append((valloss_c2 / nb))
            print('Epoch:%i, Training loss:%.3f Validation Loss: %.3f Training Acc: %.3f, Validation Acc %.3f' % (epoch+1, trainloss_c2/nbt, valloss_c2/nb, tcorr/total, vcorr/tot))
            #print('Epoch ' + str(epoch + 1) + ', Val loss: ' + str(valloss_c2 / nb))
            #print('Epoch ' + str(epoch + 1) + ', Val Accuracy: ' + str(100 * vcorr / tot))
            cl2_vacc.append((100 * vcorr / tot))

    plt.plot(cl2_trainloss, label='training loss')
    plt.plot(cl2_valloss, label='Val loss')
    plt.title('Supervised Classifier learning')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.show()
    plt.close()

    plt.plot(cl2_tacc, label='Train accuracy')
    plt.plot(cl2_vacc, label='Val accuracy')
    plt.title('Supervised Classifier learning')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    plt.close()

    # Teste med test dataen - trenger ikke plottes

    testcorr = 0
    total = 0
    with torch.no_grad():
        clf1.eval()
        for imgs, labs in clf_test:
            imgs = torch.flatten(imgs, start_dim=1)
            preds = clf1(imgs)

            _, predicted = torch.max(preds.data, 1)
            total += labs.size(0)
            testcorr += (predicted == labs).sum().item()

    print('Supervised classifier - Test set Accuracy: ' + str(100 * testcorr / total))

    #Comparative plot accuracies vs epochs:
    plt.plot(cl1_tacc, label='Semi-supervised, Training')
    plt.plot(cl1_vacc, label='Semi-supervised, Validation')
    plt.plot(cl2_tacc, label='Supervised Train')
    plt.plot(cl2_vacc, label='Supervised Validation')
    plt.title('Comparative Classifier learning')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    plt.close()


main()




