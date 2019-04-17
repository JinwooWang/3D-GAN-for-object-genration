import sys
import os
import scipy.ndimage as nd
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def plot_voxels(voxels):
    x,y, z = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c= 'red')
    #plt.savefig("1.png")
    plt.show()

def get_voxel(path):
    voxels = np.pad(io.loadmat(path)['instance'],(1,1),'constant',constant_values=(0,0))
    voxels = nd.zoom(voxels, (1,1,1), mode='constant', order=0)
    return voxels



if __name__ == '__main__':
    #path = sys.argv[1]
    path = 'volumetric_data/chair/30/train/chair_000000182_1.mat'
    volume = get_voxel(path)
    plot_voxels(volume)
