import numpy as np
import matplotlib.pyplot as plt
import math


class Generator:

    def __init__(self, dim, noise, train, val, test):
        #Check that the inputs make sense and are within bounds
        if noise > 100 or noise < 0:
            raise Exception("Noise must be between 0-100%.")
        else:
            self.noise = noise

        if (train + val + test) != 100:
            raise Exception("Data splits must add up to 100%.")
        else:
            self.train = train
            self.val = val
            self.test = test

        if dim < 10 or dim > 50:
            raise Exception("Image is NxN where N is between 10 and 50.")
        else:
            self.dim = dim

        return

    def rectangle(self):
        frame = np.zeros((self.dim, self.dim))

        #Logic with randomized variation
        #Starting points in first quartile
        i = int(np.random.choice(np.arange(0, math.floor(self.dim/2)), 1, replace=False))
        j = int(np.random.choice(np.arange(0, math.floor(self.dim/2)), 1, replace=False))

        #Max length and width
        h = int(np.random.choice(np.arange(3, self.dim-1-j), 1, replace=False))
        w = int(np.random.choice(np.arange(3, self.dim-1-i), 1, replace=False))

        i1 = i
        j1 = j
        for y in range(0, h):
            for x in range(0, w):
                frame[j1, i1] = 1
                i1+=1

            i1 = i
            j1+=1

        i2 = i+1
        j2 = j+1
        for y in range(0, h-2):
            for x in range(0, w-2):
                frame[j2, i2] = 0
                i2+=1

            i2 = i+1
            j2+=1

        #Through the logic the shapes, sizes and placements are randomized - could for sure have been more elegant.

        #Add noise as specified by user
        n_pix = int((self.noise / 100) * self.dim * self.dim)
        frame = np.array(frame, dtype=bool)
        for i in range(n_pix):
            x = np.random.choice(range(0, self.dim), 1, replace=False)
            y = np.random.choice(range(0, self.dim), 1, replace=False)
            frame[x, y] = ~frame[x, y]

        #Visualize
        #plt.plot()
        #plt.imshow(frame)
        #plt.show()

        return frame.astype(dtype=int)

    def circle(self):
        frame = np.zeros((self.dim, self.dim))

        #Randomizing sizes and centers
        ci, cj = np.random.choice(range(2, (self.dim-2)), 1, replace=False), np.random.choice(range(2, (self.dim-2)), 1, replace=False)

        r = 1000
        while r >= min(ci, cj, self.dim-ci-1, self.dim-cj-1):
            r = np.random.choice(range(1, math.floor((self.dim-1)/2)))

        for angle in range(0, 360):
            x = (r+1) * math.sin(math.radians(angle)) + ci
            y = (r+1) * math.cos(math.radians(angle)) + cj
            frame[int(np.round(y))][int(np.round(x))] = 1

        #Add noise based on specified parameter
        n_pix = int((self.noise/100)*self.dim*self.dim)
        frame = np.array(frame, dtype=bool)
        for i in range(n_pix):
            x = np.random.choice(range(0,self.dim), 1, replace=False)
            y = np.random.choice(range(0,self.dim), 1, replace=False)
            frame[x, y] = ~frame[x, y]

        #Plot the 2D-array
        #plt.plot()
        #plt.imshow(frame)
        #plt.show()

        return frame.astype(dtype=int)

    def hbars(self):
        frame = np.zeros((self.dim, self.dim)).astype(dtype=int)

        #Somewhat randomized hbars:
        start = int(np.random.choice(range(0, 5), 1, replace=False))
        step = int(np.random.choice(np.arange(2, 6), 1, replace=False))
        size = int(np.random.choice(np.arange(self.dim-5, self.dim), 1, replace=False))
        for i in range(start, size, step):
            frame[i, :] = 1

        #Add noise
        n_pix = int((self.noise / 100) * self.dim * self.dim)
        frame = np.array(frame, dtype=bool)
        for i in range(n_pix):
            x = np.random.choice(range(0, self.dim), 1, replace=False)
            y = np.random.choice(range(0, self.dim), 1, replace=False)
            frame[x, y] = ~frame[x, y]

        #Visualization
        #plt.plot()
        #plt.imshow(frame)
        #plt.show()

        return frame.astype(dtype=int)

    def vbars(self):
        frame = np.zeros((self.dim, self.dim))

        # Somewhat randomized vbars:
        start = int(np.random.choice(range(0, 5), 1, replace=False))
        step = int(np.random.choice(np.arange(2, 6), 1, replace=False))
        size = int(np.random.choice(np.arange(self.dim - 5, self.dim), 1, replace=False))
        for i in range(start, size, step):
            frame[:, i] = 1

        # Add noise
        n_pix = int((self.noise / 100) * self.dim * self.dim)
        frame = np.array(frame, dtype=bool)
        for i in range(n_pix):
            x = np.random.choice(range(0, self.dim), 1, replace=False)
            y = np.random.choice(range(0, self.dim), 1, replace=False)
            frame[x, y] = ~frame[x, y]

        # Visualization
        #plt.plot()
        #plt.imshow(frame)
        #plt.show()

        return frame.astype(dtype=int)

    def generate(self, quantity):
        #Generate a user specified dataset-quantity. They are split into train-val-test as specified.
        #three npy files are created for each dataset. Another three label-files are created for each corresponding datafile.
        figs = ['r', 'c', 'h', 'v'] #Shapes

        #Quantities for each
        train = int(np.ceil(quantity*(self.train/100)))
        val = int(np.floor(quantity*(self.val/100)))
        test = int(np.floor(quantity*(self.test/100)))

        #To add for each file as data and labels are generated - a label and data container
        images= np.zeros((train, self.dim*self.dim))
        imlab = np.zeros((train,4))

        vali = np.zeros((val, self.dim*self.dim))
        vallab = np.zeros((val,4))

        testi = np.zeros((test, self.dim * self.dim))
        testlab = np.zeros((test,4))

        print('qty of train, val, test:', train,val,test)

        #Generating and saving to files - it is randomized in uniform distribution.
        for i in range(0, train):
            f = np.random.choice(figs)
            if f == "r":
                td = self.rectangle()
            if f == "c":
                td = self.circle()
            if f == "h":
                td = self.hbars()
            if f == "v":
                td = self.vbars()

            images[i] = td.flatten()
            imlab[i][self.onehot(f)] = 1
        np.save('train.npy', images)
        np.save('train_labels.npy', imlab)

        for i in range(0, val):
            f = np.random.choice(figs)
            if f == "r":
                vd = self.rectangle()
            if f == "c":
                vd = self.circle()
            if f == "h":
                vd =  self.hbars()
            if f == "v":
                vd = self.vbars()

            vali[i] = vd.flatten()
            vallab[i][self.onehot(f)] = 1
        np.save('val.npy', vali)
        np.save('vallabels.npy', vallab)

        for i in range(0, test):
            f = np.random.choice(figs)
            if f == "r":
                ted = self.rectangle()
            if f == "c":
                ted = self.circle()
            if f == "h":
                ted = self.hbars()
            if f == "v":
                ted = self.vbars()

            testi[i] = ted.flatten()
            testlab[i][self.onehot(f)] = 1
        np.save('testing.npy', testi)
        np.save('testlabels.npy', testlab)

        return

    def onehot(self, gg):
        #Method for one hot encoding of labels - we get a [X, X, X, X] where the coresponding index is 1 to mark label

        if(gg =='r'): #rectangle
            return 0
        if(gg =='c'): #circle
            return 1
        if(gg=='h'): #hbar
            return 2
        if(gg=='v'): #vbar
            return 3

    def vis(self, img):
        plt.plot()
        plt.imshow(img)
        plt.show()
        return None


#Tests: Generator initialization:
o = Generator(12, 1, 70, 10, 20)

#Each shape method:
#o.circle()
#o.hbars()
#o.vbars()
#o.rectangle()

#Generating the images and files:
o.generate(100)