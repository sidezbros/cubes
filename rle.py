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
testcube3 = np.array([[[0,1,0], [1,1,1],[1,0,1]]])
testcube4 = np.array([[[0,0,1], [1,1,1],[1,1,0]]])
testcube5 = np.array([[[0,0,0], [1,0,0],[0,0,0]], [[0,1,0], [1,1,1],[1,0,1]],[[0,0,0], [0,0,1],[0,0,0]]])
testcube6 = np.array([[[0,0,0], [1,0,0],[0,0,0]], [[0,0,1], [1,1,1],[1,1,0]],[[0,0,0], [0,0,1],[0,0,0]]])

order_cube = np.array([[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]]])

corner_cube = np.ones((2, 2, 2))
corner_cube[0,0,0] = 0

sevenprob1 = np.array([[[0,1],
  [0,1],
  [0,1]],
 [[0,0],
  [1,1],
  [0,0]],
 [[0,0],
  [1,1],
  [0,0]]])

sevenprob2 = np.array([[[0,1],
  [0,1],
  [0,0]],
 [[0,0],
  [1,1],
  [1,0]],
 [[0,1],
  [0,1],
  [0,0]]])

zcube1 = np.array([[[0, 1, 1],
                    [1, 1, 0]]])
zcube2 = np.array([[[0, 1],
                    [1, 1],
                    [1, 0]]])

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

def generate_distance_cube(n):
    cube = np.empty((n, n, n), dtype=int)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                cube[x, y, z] = (x**2+y**2+z**2)**2  # distance is already squared
    return cube  # returns r^2 for each distance

def calc_all_corners(distcube, polycube):
    corners = np.empty((2, 2, 2), dtype=int)
    flipped = np.rot90(polycube, 2, (0, 2))
    for i in range(4):
        cube = np.rot90(polycube, i, (0, 1))
        flipcube = np.rot90(flipped, i-1, (0, 1))
        corners[(i+i//2) % 2, i//2, 0] = np.sum(distcube[:cube.shape[0], :cube.shape[1], :cube.shape[2]]*cube)
        corners[(i+i//2) % 2, i//2, 1] = np.sum(distcube[:flipcube.shape[0], :flipcube.shape[1], :flipcube.shape[2]]*flipcube)
    return corners

def orient_polycube(polycube):
    #render_shape(polycube)
    n = np.sum(polycube)
    scalecube = np.array([[[1, 10],[10, 1000]],[[10, 1000],[1000, 10000]]])
    distcube = generate_distance_cube(n)
    corners = calc_all_corners(distcube, polycube)
    sums = []

    all_orientations = list(all_rotations(corners))
    for i in range(24):
        sums.append(np.sum(all_orientations[i] * scalecube))
    sums = np.array(sums)
    min_index = np.where(sums == sums.min())[0]
    #print(sums[min_index])
    cubes = list(all_rotations(polycube))
    moments = []
    for index in min_index:
        moments.append(find_second_moment(cubes[index], find_centroid(cubes[index], n), 10**-5))
        render_shape(cubes[index])
    moments = np.array(moments)
    #print(min_index)
    #print(moments)

    indexes_to_keep = (np.where((moments[:, 0] == moments.min())))[0]
    print(len(indexes_to_keep))
    if len(indexes_to_keep) == 1:
        return cubes[min_index[indexes_to_keep[0]]]
    #min_index = min_index[indexes_to_keep]
    #moments = moments[indexes_to_keep]

    #indexes_to_keep = (np.where((moments[:, 2] == moments.max())))[0]

    sums = []
    for index in indexes_to_keep:

        corners3 = (calc_all_corners(distcube, cubes[min_index[index]]))
        sums.append(np.sum(corners3 * scalecube))
    return cubes[min_index[indexes_to_keep[np.argmin(sums)]]]
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

print('zcube1')
(orient_polycube(zcube1))
print('zcube2')
(orient_polycube(zcube2))
#orient_polycube(sevenprob2)


