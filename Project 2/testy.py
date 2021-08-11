import numpy as np

bi = np.load('train.npy')
b1 = bi[0].reshape(10,10)

print(b1)

#self.x_pad = np.pad(self.x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

b = np.pad(b1, ((1, 1), (1, 1), (0,0)), 'constant', constant_values=0)

print(b)