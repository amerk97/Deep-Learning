import numpy as np
import matplotlib.pyplot as plt
# from random import random, seed

class Layer:
    def __init__(self, input_size, output_size, activation='sigmoid', lr=0.001):
        self.input = input_size
        self.output = output_size
        self.weights = np.random.randn(self.input, self.output)*np.sqrt(2/self.input) #0<vals<1
        self.bias = np.zeros((1, self.output)) #start at 0

        if(activation == 'relu'):
            self.act = self.relu
            self.dact =self.drelu
        elif(activation == 'sigmoid'):
            self.act = self.sigmoid
            self.dact = self.dsigmoid
        elif(activation == 'tanh'):
            self.act = self.tanh
            self.dact = self.dtanh
        elif(activation == 'linear'):
            self.act = self.linear
            self.dact = self.dlinear

        self.lr = lr

    def forward(self, input):
        self.x = input
        self.z = np.dot(self.x, self.weights)+self.bias #z = xW + b for particular layer
        self.a = self.act(self.z) #activate
        print(self.a)
        return self.a

    def update(self, weight_gradient, bias_gradient):
        self.weights -= self.lr * weight_gradient
        self.bias -= self.lr*bias_gradient

    #Activation functions
    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))

    def relu(self, x):
        return np.maximum(x,0)

    def tanh(self, x):
        return np.tanh(x)

    def linear(self, x):
        return x

    #Derivatives of activation functions
    def dsigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def drelu(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def dtanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def dlinear(self, x):
        return 1


class Network:

    def __init__(self, images, labels, loss, soft=True, n_layers=None):
        self.images = images
        self.labels = labels
        self.soft = soft

        #Defining layers and nodes and the activation functions on each layer, and learning rate
        #self.layers = []

        self.layer1 = Layer(12*12, 100, activation='sigmoid')
        self.layer2 = Layer(100, 4, activation='sigmoid')
        #self.layer3 = Layer(800, 400, activation= 'sigmoid')
        #self.layer4 = Layer(400, 200, activation= 'sigmoid')
        #self.layer5 = Layer(200, 4, activation='relu', lr = 0.08)

        self.layers = [self.layer1, self.layer2]

        #Choose Loss function
        if(loss == 'CE'):
            self.loss = self.crossentropy
            self.dloss = self.dcrossentropy
        elif(loss == 'MSE'):
            self.loss = self.MSE
            self.dloss = self.dMSE


    def train(self, epochs, vdata, vlabel, testdata, testlabel, verbose=False, wrt = None):
        training_loss = []
        val_loss = []
        t_acc = []
        v_acc = []

        #Train in epochs, go through all training images each time then backprop
        for e in range(epochs):
            correct = 0
            tot = 0
            if verbose: print('EPOCH: ', e)

            for i in range(len(self.images)):
                image = self.images[i]
                label = self.labels[i]

                if verbose: print('Input data: \n', image)

                y = self.softmax(self.forward(image)).flatten()

                if verbose: print('target: ', label, 'Output: ', y)

                if(np.argmax(y) == np.argmax(label)):
                    correct += 1
                tot += 1
                print(y)
                loss = self.loss(label, y)
                if verbose: print('Loss: ', loss, '\n')

                training_loss_temp = list()
                training_loss_temp.append(loss)

                # softmax step for J is run here, then rest of backprop with the method (Z!)
                J_L_S = self.dloss(label, y)
                J_S = self.dsoftmax(y)
                J_L_Z = np.dot(J_L_S.T, J_S)
                self.backpropagation(J_L_Z)
            print('Training accuracy this epoch:', correct/tot)
            training_loss.append(np.mean(training_loss_temp))
            t_acc.append(correct/tot)

            c = 0
            t = 0
            for i in range(len(vdata)):
                img = vdata[i]
                lab = vlabel[i]
                yy = self.softmax(self.forward(img)).flatten()

                if (np.argmax(yy) == np.argmax(lab)):
                    c += 1
                t += 1

                loss = self.loss(lab, yy)
                temp = list()
                temp.append(loss)
            val_loss.append(np.mean(temp))
            print('Validation accuracy this epoch:', c / t)
            v_acc.append(c/t)

        #Plot training and val losses
        plt.plot(training_loss, label = 'training')
        plt.plot(val_loss, label = 'validation')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        #Plot accuracies:
        plt.plot(t_acc, label= 'Training accuracy')
        plt.plot(v_acc, label = 'validation accuracy')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.show()

        #Test data and print results.
        tcorr = 0
        ttot = 0
        for i in range(len(testdata)):
            im = testdata[i]
            lb = testlabel[i]
            pred = self.softmax(self.forward(im)).flatten()

            if (np.argmax(pred) == np.argmax(lb)):
                tcorr += 1
            ttot += 1

            loss = self.loss(lb, pred)
            tmp = list()
            tmp.append(loss)
        testloss = np.mean(temp)
        print('\n')
        print('Testing accuracy of final model:', tcorr / ttot)
        print('Averaged test loss: ', testloss)


    def forward(self, forward):
        #Forward pass all layers in the network, use forward method from Layer class
        #if self.layers:
        for layer in self.layers:
            forward = layer.forward(forward) #loop all layers, with layerwise method from Layer class
        return forward #a

    def backpropagation(self, J_L_Z):
        #continue rest of propagation
        for i in range(len(self.layers)-1, -1, -1):

            J_sum = self.layers[i].dact(self.layers[i].z)*np.identity(self.layers[i].z.shape[1])
            J_Z_W = np.outer(self.layers[i].x, np.diag(J_sum))
            J_L_W = J_L_Z*J_Z_W
            J_L_B = J_L_Z

            self.layers[i].update(J_L_W, J_L_B)

            J_Z_Y = np.dot(J_sum, self.layers[i].weights.T)
            J_L_Z = np.dot(J_L_Z, J_Z_Y)

        return

    #Loss functions and its derivatives. MSE and CE are supported
    def MSE(self, target, pred):
        return np.square(np.subtract(target, pred)).mean()

    def dMSE(self, target, pred):
        return target-pred

    def crossentropy(self, targets, predictions):
        return -1 * np.sum(targets * np.log(predictions))

    def dcrossentropy(self, targets, predictions):
        return np.where(predictions != 0, -targets / predictions, 0.0)
    #Cross entropies are used as suggested in announcement

    #The softmax function and the derivative
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def dsoftmax(self, x):
        s = x.reshape(4,1)

        out = np.zeros((len(s),len(s)))
        for i in range(len(s)):
            for j in range(len(s)):
                if(j == i):
                    out[i][j] = s[i]-s[i]*s[i]
                else:
                    out[i][j] = -s[i]*s[j]
        return out

#Test and run

#train network
trainimg = np.load('train.npy')
trainlab = np.load('train_labels.npy')

valimg = np.load('val.npy')
vallab = np.load('vallabels.npy')

testimg = np.load('testing.npy')
testlab = np.load('testlabels.npy')

network = Network(trainimg, trainlab, 'CE')
network.train(10, valimg, vallab, testimg, testlab)








