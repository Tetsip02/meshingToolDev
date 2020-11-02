import meshio
import numpy as np
import pickle

########################################################################
nDivisions = [10, 10, 10]

# with open("./all_points.txt", "wb") as fp:   #Pickling
#     pickle.dump(all_points, fp)
# with open("./rootIndex.txt", "wb") as fp:   #Pickling
#     pickle.dump(rootIndex, fp)
# with open("./DS.txt", "wb") as fp:   #Pickling
#     pickle.dump(DS, fp)
# with open("./TS.txt", "wb") as fp:   #Pickling
#     pickle.dump(TS, fp)
# with open("./LS.txt", "wb") as fp:   #Pickling
#     pickle.dump(LS, fp)
# with open("./RS.txt", "wb") as fp:   #Pickling
#     pickle.dump(RS, fp)
# with open("./FS.txt", "wb") as fp:   #Pickling
#     pickle.dump(FS, fp)
# with open("./BS.txt", "wb") as fp:   #Pickling
#     pickle.dump(BS, fp)
# with open("./level0_mesh.txt", "wb") as fp:   #Pickling
#     pickle.dump(level0_mesh, fp)
# with open("./level1_mesh.txt", "wb") as fp:   #Pickling
#     pickle.dump(level1_mesh, fp)
# with open("./level2_mesh.txt", "wb") as fp:   #Pickling
#     pickle.dump(level2_mesh, fp)
# with open("./level3_mesh.txt", "wb") as fp:   #Pickling
#     pickle.dump(level3_mesh, fp)
# with open("./level0_targets.txt", "wb") as fp:   #Pickling
#     pickle.dump(level0_targets, fp)
# with open("./level1_targets.txt", "wb") as fp:   #Pickling
#     pickle.dump(level1_targets, fp)
# with open("./level2_targets.txt", "wb") as fp:   #Pickling
#     pickle.dump(level2_targets, fp)
with open("./all_points.txt", "rb") as fp:   # Unpickling
    all_points = pickle.load(fp)
with open("./rootIndex.txt", "rb") as fp:   # Unpickling
    rootIndex = pickle.load(fp)
with open("./DS.txt", "rb") as fp:   # Unpickling
    DS = pickle.load(fp)
with open("./TS.txt", "rb") as fp:   # Unpickling
    TS = pickle.load(fp)
with open("./LS.txt", "rb") as fp:   # Unpickling
    LS = pickle.load(fp)
with open("./RS.txt", "rb") as fp:   # Unpickling
    RS = pickle.load(fp)
with open("./FS.txt", "rb") as fp:   # Unpickling
    FS = pickle.load(fp)
with open("./BS.txt", "rb") as fp:   # Unpickling
    BS = pickle.load(fp)
with open("./level0_mesh.txt", "rb") as fp:   # Unpickling
    level0_mesh = pickle.load(fp)
with open("./level1_mesh.txt", "rb") as fp:   # Unpickling
    level1_mesh = pickle.load(fp)
with open("./level2_mesh.txt", "rb") as fp:   # Unpickling
    level2_mesh = pickle.load(fp)
with open("./level3_mesh.txt", "rb") as fp:   # Unpickling
    level3_mesh = pickle.load(fp)
with open("./level0_targets.txt", "rb") as fp:   # Unpickling
    level0_targets = pickle.load(fp)
with open("./level1_targets.txt", "rb") as fp:   # Unpickling
    level1_targets = pickle.load(fp)
with open("./level2_targets.txt", "rb") as fp:   # Unpickling
    level2_targets = pickle.load(fp)
#################################################################

def hex_contains(hex_points, x): # points need to satisfy certain order
    # n1 = np.cross(hex_points[1] - hex_points[0], hex_points[2] - hex_points[0])
    n1 = np.array([0, 0, 1])
    x1 = np.dot(x - hex_points[0], n1)
    if x1 < 0:
        return False
    # n2 = np.cross(hex_points[7] - hex_points[4], hex_points[6] - hex_points[4])
    n2 = np.array([0, 0, -1])
    x2 = np.dot(x - hex_points[4], n2)
    if x2 < 0:
        return False
    # n3 = np.cross(hex_points[3] - hex_points[0], hex_points[7] - hex_points[0])
    n3 = np.array([1, 0, 0])
    x3 = np.dot(x - hex_points[0], n3)
    if x3 < 0:
        return False
    # n4 = np.cross(hex_points[5] - hex_points[1], hex_points[6] - hex_points[1])
    n4 = np.array([-1, 0, 0])
    x4 = np.dot(x - hex_points[1], n4)
    if x4 < 0:
        return False
    # n5 = np.cross(hex_points[4] - hex_points[0], hex_points[5] - hex_points[0])
    n5 = np.array([0, 1, 0])
    x5 = np.dot(x - hex_points[0], n5)
    if x5 < 0:
        return False
    # n6 = np.cross(hex_points[2] - hex_points[3], hex_points[6] - hex_points[3])
    n6 = np.array([0, -1, 0])
    x6 = np.dot(x - hex_points[3], n6)
    if x6 < 0:
        return False
    return True

# variables:
# nDivisions
# all_points
# rootIndex
# DS
# TS
# LS
# RS
# FS
# BS
# level0_mesh
# level1_mesh
# level2_mesh
# level3_mesh
# level0_targets
# level1_targets
# level2_targets

#########################################################
octreeMap = {'U': [[4, "H"], [5, "H"], [6, "H"], [7, "H"], [0, "U"], [1, "U"], [2, "U"], [3, "U"]],
             'B': [[3, "H"], [2, "H"], [1, "B"], [0, "B"], [7, "H"], [6, "H"], [5, "B"], [4, "B"]],
             'R': [[1, "H"], [0, "R"], [3, "R"], [2, "H"], [5, "H"], [4, "R"], [7, "R"], [6, "H"]],
             'D': [[4, "D"], [5, "D"], [6, "D"], [7, "D"], [0, "H"], [1, "H"], [2, "H"], [3, "H"]],
             'F': [[3, "F"], [2, "F"], [1, "H"], [0, "H"], [7, "F"], [6, "F"], [5, "H"], [4, "H"]],
             'L': [[1, "L"], [0, "H"], [3, "H"], [2, "L"], [5, "L"], [4, "H"], [7, "H"], [6, "L"]]}
# octreeMap2 = np.array([[[4, "H"], [3, "H"], [1, "H"], [4, "D"], [3, "F"], [1, "L"]],
#                        [[5, "H"], [2, "H"], [0, "R"], [5, "D"], [2, "F"], [0, "H"]],
#                        [[6, "H"], [1, "B"], [3, "R"], [6, "D"], [1, "H"], [3, "H"]],
#                        [[7, "H"], [0, "B"], [2, "H"], [7, "D"], [0, "H"], [2, "L"]],
#                        [[0, "U"], [7, "H"], [5, "H"], [0, "H"], [7, "F"], [5, "L"]],
#                        [[1, "U"], [6, "H"], [4, "R"], [1, "H"], [6, "F"], [4, "H"]],
#                        [[2, "U"], [5, "B"], [7, "R"], [2, "H"], [5, "H"], [7, "H"]],
#                        [[3, "U"], [4, "B"], [6, "H"], [3, "H"], [4, "H"], [6, "L"]]])
octreeMap2 = [[[4, "H"], [3, "H"], [1, "H"], [4, "D"], [3, "F"], [1, "L"]],
                       [[5, "H"], [2, "H"], [0, "R"], [5, "D"], [2, "F"], [0, "H"]],
                       [[6, "H"], [1, "B"], [3, "R"], [6, "D"], [1, "H"], [3, "H"]],
                       [[7, "H"], [0, "B"], [2, "H"], [7, "D"], [0, "H"], [2, "L"]],
                       [[0, "U"], [7, "H"], [5, "H"], [0, "H"], [7, "F"], [5, "L"]],
                       [[1, "U"], [6, "H"], [4, "R"], [1, "H"], [6, "F"], [4, "H"]],
                       [[2, "U"], [5, "B"], [7, "R"], [2, "H"], [5, "H"], [7, "H"]],
                       [[3, "U"], [4, "B"], [6, "H"], [3, "H"], [4, "H"], [6, "L"]]]
####################################################################

def root_neighbour(target_root, direction):
    # target_root: index of root voxel within rootIndex
    # direction: {"U", "B", "R", "D", "F", "L"}
    # returns: index of neighbour within rootIndex as interger
    if direction == "U":
        if target_root in TS:
            return -1 # out of bounds
        return target_root + nDivisions[0] * nDivisions[1]
    if direction == "B":
        if target_root in BS:
            return -1 # out of bounds
        return target_root + nDivisions[0]
    if direction == "R":
        if target_root in RS:
            return -1 # out of bounds
        return target_root + 1
    if direction == "D":
        if target_root in DS:
            return -1 # out of bounds
        return target_root - nDivisions[0] * nDivisions[1]
    if direction == "F":
        if target_root in FS:
            return -1 # out of bounds
        return target_root - nDivisions[0]
    if direction == "L":
        if target_root in FS:
            return -1 # out of bounds
        return target_root - 1

def all_root_neighbours(target_root):
    # target_root: index of root voxel within rootIndex
    # returns: list; contains indices of all neighbours within rootIndex
    neighbours = [-1, -1, -1, -1, -1, -1]
    if target_root not in TS:
        # neighbours.append(target_root + nDivisions[0] * nDivisions[1]) # upper neighbour
        neighbours[0] = target_root + nDivisions[0] * nDivisions[1]
    if target_root not in BS:
        neighbours[1] = target_root + nDivisions[0] # back neighbour
    if target_root not in RS:
        neighbours[2] = target_root + 1 # right neighbour
    if target_root not in DS:
        neighbours[3] = target_root - nDivisions[0] * nDivisions[1] # down neighbour
    if target_root not in FS:
        neighbours[4] = target_root - nDivisions[0] # front neighbour
    if target_root not in LS:
        neighbours[5] = target_root - 1 # left neighbour
    return neighbours

def level1_neighbour(target, direction):
    # target: cell index within level1_mesh
    # direction: {"U", "B", "R", "D", "F", "L"}
    # return neighbour index within level1_mesh
    target_octant = target % 8
    target_parent = level0_targets[target // 8]
    map = octreeMap[direction][target_octant]
    neighbour_octant = map[0]
    if map[1] == "H":
        return np.argwhere(level0_targets == target_parent).ravel() * 8 + neighbour_octant
    else:
        neighbour_parent = root_neighbour(target_parent, map[1])
        if neighbour_parent == -1:
            return -1
        return np.argwhere(level0_targets == neighbour_parent).ravel() * 8 + neighbour_octant


def all_level1_neighbours(target):
    # target: cell index within level1_mesh
    # return level 1 neighbours in all orthogonal directions {"U", "B", "R", "D", "F", "L"}
    target_octant = target % 8
    target_parent = level0_targets[target // 8]
    map = octreeMap2[target_octant]
    neighs = []
    for neighbour_octant, direction in map:
        if direction == "H":
            neighs.extend(np.argwhere(level0_targets == target_parent).ravel() * 8 + neighbour_octant)
        else:
            neigh_parent = root_neighbour(target_parent, direction)
            if neigh_parent == -1:
                neighs.extend(-1)
            else:
                neighs.extend(np.argwhere(level0_targets == neigh_parent).ravel() * 8 + neighbour_octant)
    return neighs

def level2_neighbour(target, direction):
    # target: cell index within level1_mesh
    # direction: {"U", "B", "R", "D", "F", "L"}
    # return neighbour index within level2_mesh
    target_octant = target % 8
    target_parent = level1_targets[target // 8]
    map = octreeMap[direction][target_octant]
    neighbour_octant = map[0]
    if map[1] == "H":
        return np.argwhere(level1_targets == target_parent).ravel() * 8 + neighbour_octant
    else:
        neighbour_parent = level1_neighbour(target_parent, map[1])
        if neighbour_parent == -1:
            return -1
        return np.argwhere(level1_targets == neighbour_parent).ravel() * 8 + neighbour_octant

def all_level2_neighbours(target):
    # target: cell index within level1_mesh
    # return level 2 neighbours in all orthogonal directions {"U", "B", "R", "D", "F", "L"}
    target_octant = target % 8
    target_parent = level1_targets[target // 8]
    map = octreeMap2[target_octant]
    neighs = []
    for neighbour_octant, direction in map:
        if direction == "H":
            neighs.extend(np.argwhere(level1_targets == target_parent).ravel() * 8 + neighbour_octant)
        else:
            neigh_parent = level1_neighbour(target_parent, direction)
            if neigh_parent == -1:
                neighs.extend(-1)
            else:
                neighs.extend(np.argwhere(level1_targets == neigh_parent).ravel() * 8 + neighbour_octant)
    return neighs

def level3_neighbour(target, direction):
    # target: cell index within level1_mesh
    # direction: {"U", "B", "R", "D", "F", "L"}
    # return neighbour index within level3_mesh
    target_octant = target % 8
    target_parent = level2_targets[target // 8]
    map = octreeMap[direction][target_octant]
    neighbour_octant = map[0]
    if map[1] == "H":
        return np.argwhere(level2_targets == target_parent).ravel() * 8 + neighbour_octant
    else:
        neighbour_parent = level2_neighbour(target_parent, map[1])
        if neighbour_parent == -1:
            return -1
        return np.argwhere(level2_targets == neighbour_parent).ravel() * 8 + neighbour_octant

def all_level3_neighbours(target):
    # target: cell index within level1_mesh
    # return level 3 neighbours in all orthogonal directions {"U", "B", "R", "D", "F", "L"}
    target_octant = target % 8
    target_parent = level2_targets[target // 8]
    map = octreeMap2[target_octant]
    neighs = []
    for neighbour_octant, direction in map:
        if direction == "H":
            neighs.extend(np.argwhere(level2_targets == target_parent).ravel() * 8 + neighbour_octant)
        else:
            neigh_parent = level2_neighbour(target_parent, direction)
            if neigh_parent == -1:
                neighs.extend(-1)
            else:
                neighs.extend(np.argwhere(level2_targets == neigh_parent).ravel() * 8 + neighbour_octant)
    return neighs

# print((all_level3_neighbours(29)))
x = all_level3_neighbours(30)

# level3_neighbours = all_level3_neighbours(start_cell)

#l3 targets[29, 30, 124, 25]


voxel1 = [("hexahedron", level3_mesh[[30]])]
mesh = meshio.Mesh(points = all_points, cells = voxel1)
meshio.write("./l3_cell30.vtk", mesh, file_format="vtk", binary=False)
voxel1 = [("hexahedron", level3_mesh[x])]
mesh = meshio.Mesh(points = all_points, cells = voxel1)
meshio.write("./l3_cell30_neigh.vtk", mesh, file_format="vtk", binary=False)
