import numpy as np

def kernel_middle(size):
    k = np.zeros(size)
    k[int((size[0]-1)/2), :] = np.ones(shape=(size[1]))
    return k

def pad(vects, fsize):
    hsize = int((fsize-1)/2)
    output = np.zeros(shape=(vects.shape[0], vects.shape[1] + 2*hsize, vects.shape[2]))
    output[:, hsize:-hsize, :] = vects
    return output
