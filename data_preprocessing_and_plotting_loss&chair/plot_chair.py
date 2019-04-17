import numpy as np
import os
import numpy
from skimage import measure
from stl import mesh
import stl
# Plot out the meshed object
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from stl import mesh
from skimage import measure


def plot_chair(filename):
    samples=np.load("%d" % filename)
    chair = samples[10]
    chair=chair.reshape([32,32,32])
    chair=chair.round()

    """
    #scatter version

    x,y, z = chair.nonzero()
    #print x
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c= 'red')
    #plt.savefig("1.png")
    plt.show()
    """
    vertices, faces = measure.marching_cubes_classic(chair,level=0.5)
    chair = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            chair.vectors[i][j] = vertices[f[j],:]


    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(chair.vectors, facecolors = 'r'))
    scale = chair.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    plt.show()


if __name__ == "__main__":
    plot_chair(15000)
