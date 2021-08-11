import numpy as np
import matplotlib.pyplot as plt


#Dense Layers
class LayerD:
    def __init__(self, input_size, output_size, activation='sigmoid', lr=0.001):
        self.input = input_size
        self.output = output_size
        self.weights = np.random.randn(self.input, self.output)*np.sqrt(2/self.input) #0<vals<1
        self.bias = np.zeros((1, self.output)) #start at 0
        self.lr = lr

        self.ac = Activation()

        if(activation == 'relu'):
            self.act = self.ac.relu
            self.dact = self.ac.drelu
        elif(activation == 'sigmoid'):
            self.act = self.ac.sigmoid
            self.dact = self.ac.dsigmoid
        elif(activation == 'tanh'):
            self.act = self.ac.tanh
            self.dact = self.ac.dtanh
        elif(activation == 'linear'):
            self.act = self.ac.linear
            self.dact = self.ac.dlinear
        elif(activation == 'elu'):
            self.act = self.ac.elu
            self.dact = self.ac.delu
        elif(activation == 'selu'):
            self.act = self.ac.selu
            self.dact = self.ac.dselu

    def forward(self, input):
        self.x = input
        self.z = np.dot(self.x, self.weights)+self.bias #z = xW + b for particular layer
        self.a = self.act(self.z) #activate
        return self.a

    def update(self, weight_gradient, bias_gradient):
        self.weights -= self.lr * weight_gradient
        self.bias -= self.lr*bias_gradient


#Convolutional layers
class LayerC:
    def __init__(self, input_c, input_size, output_size, activation='sigmoid', lr=0.001, stride=1, mode='valid', n_kernels=2, kd1=2, kd2=2):
        self.input = input_c
        self.in_size = input_size
        self.output = output_size
        self.kstack = np.random.randn(n_kernels, input_c, kd1, kd2)*np.sqrt(2 /n_kernels)

        self.stride = stride
        self.mode = mode

        self.lr = lr
        self.ac = Activation()

        if (activation == 'relu'):
            self.act = self.ac.relu
            self.dact = self.ac.drelu
        elif (activation == 'sigmoid'):
            self.act = self.ac.sigmoid
            self.dact = self.ac.dsigmoid
        elif (activation == 'tanh'):
            self.act = self.ac.tanh
            self.dact = self.ac.dtanh
        elif (activation == 'linear'):
            self.act = self.ac.linear
            self.dact = self.ac.dlinear
        elif (activation == 'elu'):
            self.act = self.ac.elu
            self.dact = self.ac.delu
        elif (activation == 'selu'):
            self.act = self.ac.selu
            self.dact = self.ac.dselu

    def forward(self, a_prev, verbose=True):
        #a_prev = a_prev.reshape(10,10)
        #Få dimensjonene, og sjekker om input er single channel
        if a_prev.ndim == 2:
            (n_h_prev, n_w_prev) = a_prev.shape
            n_c_prev = 1
        else:
            (n_h_prev, n_w_prev, n_c_prev) = a_prev.shape

        (n_c, n_c_prev, f, f) = self.kstack.shape

        #Padding og modes
        if (self.mode == 'valid'):
            pad = 0
            a_prev_pad = a_prev
            n_h = int((n_h_prev-f) / self.stride) + 1
            n_w = int((n_w_prev-f) / self.stride) + 1
            #print(n_h)

        elif (self.mode == 'full'): #?***
            pad = int(np.ceil(((self.stride - 1) * (n_h_prev+f-1) - self.stride + f) / 2))
            #print(pad)
            a_prev_pad = self.padding(a_prev, pad)
            n_h = int((n_h_prev - f + 2 * pad) / self.stride) + 1
            n_w = int((n_w_prev - f + 2 * pad) / self.stride) + 1
            #print(n_h)

        elif (self.mode == 'same'):
            pad = int(np.ceil(((self.stride-1)*n_h_prev-self.stride+f)/2))
            #print(pad)
            a_prev_pad = self.padding(a_prev, pad)
            n_h = int((n_h_prev - f + 2 * pad) / self.stride) + 1
            n_w = int((n_w_prev - f + 2 * pad) / self.stride) + 1
            #print(n_h)

        #Tom array for outputen, med rett dimensjoner
        z = np.zeros((n_h, n_w, n_c))

        #Looper og conver. og aktivere hver outputt.
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    vert_start = h * self.stride
                    vert_end = vert_start + f
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + f

                    if(a_prev.ndim == 2):
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end]
                    else:
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    z[h, w, c] = self.conv(a_slice_prev, self.kstack[c, ...])
                    z[h, w, c] = self.act(z[h, w, c])

        self.cache = (a_prev, self.kstack, pad, self.stride) #til backprop evt

        print(z)

        if verbose: print(self.kstack)

        return z

    def padding(self, input, pad):
        self.x = input
        if(input.ndim == 2):
            self.x_pad = np.pad(self.x, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
        else:
            self.x_pad = np.pad(self.x, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
        return self.x_pad

    def conv(self, a_prev_part, W):
        s = np.multiply(a_prev_part, W)
        self.z = np.sum(s)
        return self.z

    def update(self, k_grad):
        self.kstack -= self.lr * k_grad
        return


#Activation functions and derivatives (containing a d as first letter)
class Activation:

    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))

    def relu(self, x):
        return np.maximum(x,0)

    def tanh(self, x):
        return np.tanh(x)

    def linear(self, x):
        return x

    def elu(self, x):
        alp = 0.5 #kan tilpasses
        x[x > 0] = x
        x[x < 0] = alp*(np.exp(x)-1)
        return x

    def selu(self, x):
        alp = 1.6732 #Samme med alfa og sånt her, litt usiker på om dette blir rett
        gam = 1.0507
        x[x > 0] = gam*x
        x[x <= 0] = gam*(alp*np.exp(x)-alp)
        return x

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

    def delu(self, x):
        alp = 0.5 #Må samsvare med elu
        x[x > 0] = 1
        x[x <= 0] = self.elu(x) + alp
        return x

    def dselu(self, x):
        alp = 1.6732
        gam = 1.0507
        x[x > 0] = gam*1
        x[x <= 0] = gam*alp*np.exp(x)
        return x


#Network class
class Network:

    def __init__(self, images, labels, loss, soft=True, n_layers=None):
        self.images = images
        self.labels = labels
        self.soft = soft

        self.layer1 = LayerC(1, 100, 100, activation = 'sigmoid', stride = 1, mode = 'full', n_kernels = 2, kd1 = 3, kd2 = 3)
        #self.layer1 = Layerd(12*12, 400, activation='sigmoid')
        self.layer2 = LayerD(200, 4, activation='sigmoid')
        #self.layer3 = LayerD(800, 400, activation= 'sigmoid')
        #self.layer4 = LayerD(400, 200, activation= 'sigmoid')
        #self.layer5 = LayerD(200, 4, activation='relu', lr = 0.08)

        self.layers = [self.layer1, self.layer2]

        if(loss == 'CE'):
            self.loss = self.crossentropy
            self.dloss = self.dcrossentropy
        elif(loss == 'MSE'):
            self.loss = self.MSE
            self.dloss = self.dMSE


    def train(self, epochs, vdata, vlabel, testdata, testlabel, verbose=False):
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
                image = self.images[i]#.reshape(10,10) #reshape added for conv
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


### Plots, training, val, test (i.e. learning curves)
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
        #legge inn if elns. som velger om man skal bruke dense eller conv forward. Vet ikke om det egt trengs med denne måte!
        #Kobling mellom dense og conv


        for layer in self.layers:
            if forward.ndim == 3:
                forward = forward.flatten()
            forward = layer.forward(forward) #loop all layers, with layerwise method from Layer class

        #if self.layers:
        #for layer in self.layers:
         #   forward = layer.forward(forward) #loop all layers, with layerwise method from Layer class
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




#Test conv forward: _________________________
#bilde = np.abs(np.random.randint(2, size=(3, 3,2) ))
bi = np.load('train.npy')
b1 = bi[0].reshape(10,10)
#plt.plot()
#plt.imshow(b1)
#plt.show()

o = LayerC(1, 2, 4, activation='sigmoid', lr=0.001, stride=1, mode='same', n_kernels=2, kd1=3, kd2=3)
o.forward(b1, verbose=True)
#Test med laplacian kernel

#Test conv backprop: __________________

#trainimg = np.load('train.npy')
#trainlab = np.load('train_labels.npy')

#valimg = np.load('val.npy')
#vallab = np.load('vallabels.npy')

#testimg = np.load('testing.npy')
#testlab = np.load('testlabels.npy')

#network = Network(trainimg, trainlab, 'CE')
#network.train(3, valimg, vallab, testimg, testlab)



#train network: _____________
#trainimg = np.load('train.npy')
#trainlab = np.load('train_labels.npy')

#valimg = np.load('val.npy')
#vallab = np.load('vallabels.npy')

#testimg = np.load('testing.npy')
#testlab = np.load('testlabels.npy')

#network = Network(trainimg, trainlab, 'CE')
#network.train(10, valimg, vallab, testimg, testlab)








