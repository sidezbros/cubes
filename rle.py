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

def generate_distance_cube(n):
    cube = np.empty((n, n, n), dtype=int)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                cube[x, y, z] = (x**2+y**2+z**2)**2  # distance is already squared
    return cube  # returns r^2 for each distance

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


def find_balance(polycube):
    balance = []
    for axis in ((1, 2), (0, 2), (0, 1)):
        s = np.sum(polycube, axis=axis)
        #val = int(np.heaviside(, 0.5)*2 - 1)
        balance.append(abs(np.sum(s[:len(s) // 2]) - np.sum(s[len(s) % 2 + len(s) // 2:])))

    return balance

def diag_sum(polycube):
    sums = []
    for axis in range(3):
        plane = np.sum(polycube, axis=axis)
        plane2 = np.flip(np.sum(polycube, axis=axis))
        sums.append(np.trace(plane)+np.trace(plane2))
    return sums


def find_chirality(polycube):
    for rotation in all_rotations(polycube):
        if np.array_equal(np.flipud(polycube), rotation):
            return 0
    return 1

def calc_all_corners(distcube, polycube):
    corners = np.empty((2, 2, 2), dtype=int)
    flipped = np.rot90(polycube, 2, (0, 2))
    for i in range(4):
        cube = np.rot90(polycube, i, (0, 1))
        flipcube = np.rot90(flipped, i-1, (0, 1))
        corners[(i+i//2) % 2, i//2, 0] = np.sum(distcube[:cube.shape[0], :cube.shape[1], :cube.shape[2]]*cube)
        corners[(i+i//2) % 2, i//2, 1] = np.sum(distcube[:flipcube.shape[0], :flipcube.shape[1], :flipcube.shape[2]]*flipcube)
    return corners

def relate_corners(corners):
    relation = np.empty((8,4), dtype=int)
    for x in range(2):
        for y in range(2):
            for z in range(2):
                index = x+y*2+z*4
                relation[index, 0] = corners[x, y, z]
                relation[index, 1:] = np.sort([corners[(x+1) % 2, y, z], corners[x, (y+1) % 2, z], corners[x, y, (z+1) % 2]])

    return np.sort(relation.view('i,i,i,i'), order=['f0', 'f1', 'f2', 'f3'], axis=0).view(np.int)

def relate_corners_test(corners):
    relation = np.empty((8,2), dtype=int)
    for x in range(2):
        for y in range(2):
            for z in range(2):
                index = x+y*2+z*4
                relation[index, 0] = corners[x, y, z]
                relation[index, 1] = corners[(x+1) % 2, y, z] * corners[x, (y+1) % 2, z] * corners[x, y, (z+1) % 2]

    return np.sort(relation.view('i,i'), order=['f0', 'f1'], axis=0).view(np.int)
def all_corners(cube):
    corners = [[]]*8
    flipped = np.rot90(cube, 2, (0, 2))
    for i in range(4):
        corners[i] = np.rot90(cube, i, (0, 1))
        corners[i+4] = np.rot90(flipped, i-1, (0, 1))
    return corners

def rle(polycube):
    n = np.sum(polycube)
    encoded_cube = []
    encoded_cube.extend(np.sort(polycube.shape))
    corner_vals = []
    gen_cube = generate_distance_cube(n)
    #for distcube in all_corners(gen_cube[:polycube.shape[0], :polycube.shape[1], :polycube.shape[2]]):
    #    corner_vals.append(np.sum(distcube * polycube))
    corner_vals = calc_all_corners(gen_cube, polycube)
    relation = relate_corners_test(corner_vals)
    encoded_cube.extend(np.sort(corner_vals.flatten()))
    #centroid = find_centroid(polycube, n)
    #moment = find_second_moment(polycube, centroid, 10**-5)
    #encoded_cube.extend(np.sort(moment))
    #print(np.linalg.norm(polycube))
    chirality = find_chirality(polycube)
    encoded_cube.append(chirality)
    #encoded_cube.extend(np.sort(diag_sum(polycube)))
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

#for cube in all_corners(corner_cube):
#    for x in range(2):
#        for y in range(2):
#            for z in range(2):
#                if cube[x, y, z] == 0:
#                    print((x, y, z))
#    render_shape(cube)
    #print(cube)
#print(order_cube[0,1,0])
#print(rle(testcube3))
#print(rle(np.rot90(testcube3, 2, (0, 2))))
#print(rle(np.flipud(testcube)))
#print(generate_distance_cube(3))

