#### Test file, ignore this####

import numpy as np
import matplotlib.pyplot as plt
images = np.load('testing.npy')
labels = np.load('testlabels.npy')

#for i in range(10):
    #plt.imshow(images[i].reshape(25,-1))
    #print(labels[i])
    #plt.show()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def dsoftmax(x):
    s = softmax(x)
    out = np.zeros((len(s),len(s)))
    for i in range(len(s)):
        for j in range(len(s)):
            if(j == i):
                out[i][j] = s[i]-s[i]*s[i]
            else:
                out[i][j] = -s[i]*s[j]
    return out

pp = [1,2,3,4]
print(softmax(pp))
print(dsoftmax(pp))