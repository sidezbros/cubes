import math

import numpy as np
from matplotlib import pyplot as plt


def all_rotations(polycube):
    """
    Calculates all rotations of a polycube.

    Adapted from https://stackoverflow.com/questions/33190042/how-to-calculate-all-24-rotations-of-3d-array.
    This function computes all 24 rotations around each of the axis x,y,z. It uses numpy operations to do this, to avoid unecessary copies.
    The function returns a generator, to avoid computing all rotations if they are not needed.

    Parameters:
    polycube (np.array): 3D Numpy byte array where 1 values indicate polycube positions

    Returns:
    generator(np.array): Yields new rotations of this cube about all axes

    """

    def single_axis_rotation(polycube, axes):
        """Yield four rotations of the given 3d array in the plane spanned by the given axes.
        For example, a rotation in axes (0,1) is a rotation around axis 2"""
        for i in range(4):
            yield np.rot90(polycube, i, axes)

    # 4 rotations about axis 0
    yield from single_axis_rotation(polycube, (1, 2))

    # rotate 180 about axis 1, 4 rotations about axis 0
    yield from single_axis_rotation(np.rot90(polycube, 2, axes=(0, 2)), (1, 2))

    # rotate 90 or 270 about axis 1, 8 rotations about axis 2
    yield from single_axis_rotation(np.rot90(polycube, axes=(0, 2)), (0, 1))
    yield from single_axis_rotation(np.rot90(polycube, -1, axes=(0, 2)), (0, 1))

    # rotate about axis 2, 8 rotations about axis 1
    yield from single_axis_rotation(np.rot90(polycube, axes=(0, 1)), (0, 2))
    yield from single_axis_rotation(np.rot90(polycube, -1, axes=(0, 1)), (0, 2))

testcube = np.array(
    [
     [[1, 1],
      [1, 0]],
     [[0, 0],
      [1, 0]],
     [[0, 0],
      [1, 0]]
     ])
testcube2 = np.array(
    [
     [[1, 1],
      [0, 1]],
     [[0, 0],
      [0, 1]],
     [[0, 0],
      [0, 1]]
     ])


def find_centroid(polycube, n):
    centroid = []
    for axis in ((1, 2), (2, 0), (1, 0)):
        s = np.sum(polycube, axis=axis)
        m = 0.0
        for i in range(len(s)):
            m += (i + 0.5) * s[i]
        centroid.append(m / n)
    return centroid

def find_second_moment(polycube, centroid, threshold):
    moments = []
    for axis in ((1, 2), (2, 0), (1, 0)):
        axis_index = len(moments)
        slice_index = int(centroid[axis_index] // 1)
        slice_amt = centroid[axis_index] % 1
        s = np.sum(polycube, axis=axis)
        m = 0.0
        for i in range(len(s)):
            if i != slice_index or slice_amt < threshold or (1-slice_amt) < threshold:
                m += (i + 0.5 - centroid[axis_index]) ** 2 * s[i]
            else:
                m += (0.5 * slice_amt) ** 2 * (s[i] * slice_amt)
                m += (0.5 * (1 - slice_amt)) ** 2 * (s[i] * (1 - slice_amt))
        moments.append(int(round(m, 10)*10**10))
    return moments


def find_chirality_old(polycube):
    balance = []
    for axis in ((1, 2), (0, 2), (0, 1)):
        s = np.sum(polycube, axis=axis)
        val = int(np.heaviside(np.sum(s[:len(s) // 2]) - np.sum(s[len(s) % 2 + len(s) // 2:]), 0.5)*2 - 1)
        if val == 0:
            val = 1
        balance.append(val)

    return balance[0]*balance[1]*balance[2]


def find_chirality(polycube):
    for rotation in all_rotations(polycube):
        if np.array_equal(np.flipud(polycube), rotation):
            return 0
    return 1

def rle(polycube):
    n = np.sum(polycube)
    encoded_cube = []
    encoded_cube.extend(np.sort(polycube.shape))

    centroid = find_centroid(polycube, n)
    moment = find_second_moment(polycube, centroid, 10**-5)
    encoded_cube.extend(np.sort(moment))
    chirality = find_chirality(polycube)
    encoded_cube.append(chirality)
    #render_shape(polycube)
    return tuple(encoded_cube)

def render_shape(polycube):
    colors = np.empty(polycube.shape, dtype=object)
    colors[:] = '#FFD65DC0'

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(polycube, facecolors=colors, edgecolor='k', linewidth=0.1)

    ax.set_xlim([0, max(polycube.shape)])
    ax.set_ylim([0, max(polycube.shape)])
    ax.set_zlim([0, max(polycube.shape)])
    plt.axis("off")
    plt.show()

#print(rle(testcube))
#print(rle(np.flipud(testcube)))
