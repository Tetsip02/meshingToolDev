import meshio
import numpy as np
# import math
from helpers import *
from sympy import * # symbols, Eq, solve, Matrix


surface_triangles, surface_quads, surface_points = read_surface("./testGeometries/sphere.obj")
d1 = 0.1
d2 = 0.12
level1, level2, level3 = build_ref_mesh_points(surface_points, surface_triangles, surface_quads, d1, d2, 1)
nSurfacePoints = len(surface_points)
level1_triangles = surface_triangles
level1_quads = surface_quads
level2_triangles = level1_triangles + nSurfacePoints
level2_quads = level1_quads + nSurfacePoints
level3_triangles = level2_triangles + nSurfacePoints
level3_quads = level2_quads + nSurfacePoints
allPoints = np.concatenate((level1, level2, level3))

faceTable = np.load('./dataOutput/faceTable.npy', allow_pickle=True)
vertexTable = np.load('./dataOutput/vertexTable.npy', allow_pickle=True)
level2VertexTable = np.load('./dataOutput/level2VertexTable.npy', allow_pickle=True)
level2FaceTable = np.load('./dataOutput/level2FaceTable.npy', allow_pickle=True)

# print(faceTable[0:10, :])
# print(faceTable.shape)
# print(vertexTable.shape)

#######################################

# build r_zeta
r_zeta = (vertexTable[:, [4, 5, 6]] * d1 + level2VertexTable[:, [4, 5, 6]] * d2) / (d1 + d2)
# print("r_zeta", r_zeta[0:6, :])

######################################

#build r_zeta_zeta
r_zeta_zeta = (level2VertexTable[:, [4, 5, 6]] - vertexTable[:, [4, 5, 6]]) / (d1 + d2)
# print("r_zeta_zeta", r_zeta_zeta[0:6, :])

#######################################

#determine valence of each vertex
valence = np.ndarray((nSurfacePoints, 1), dtype = object) # number of points to be used in evaluating derivatives
maxNeighbours = max(vertexTable[:, 7])
if maxNeighbours < 6:
    directNeighbours = np.ndarray((nSurfacePoints, 6), dtype = object) # entry is -1 if directNeighbours < 6
    nDirectNeighbours = 6
else:
    directNeighbours = np.ndarray((nSurfacePoints, maxNeighbours), dtype = object) # entry is -1 if directNeighbours < maxNeighbours
    nDirectNeighbours = maxNeighbours
diagonals = np.ndarray((nSurfacePoints, 8), dtype = object)
nDiagonals = -np.ones((nSurfacePoints, 1), dtype = int)
print(nDiagonals)


############################################################

for vertex in range(nSurfacePoints):
    # initialize neighbours for vertex
    directNeighboursTemp = np.array([], dtype=np.int)
    diagonalsTemp = np.array([], dtype=np.int)

    nAttFaces = vertexTable[vertex, 7]
    # direct-valence = 5 or higher
    if nAttFaces > 4:
        for f in range(nAttFaces):
            fIndex = vertexTable[vertex, f + 8]
            if faceTable[fIndex, 1] == 3:
                directNeighboursTemp = np.append(directNeighboursTemp, faceTable[fIndex, [2, 3, 4]])
            elif faceTable[fIndex , 1] == 4:
                pointPos = np.where(faceTable[fIndex, [2, 3, 4, 5]] == vertex)
                pointPos = pointPos[0][0]
                directNeighboursTemp = np.append(directNeighboursTemp, [faceTable[fIndex, ((pointPos - 1) % 4) + 2], faceTable[fIndex, ((pointPos + 1) % 4) + 2]])
        # remove duplicate points ##
        directNeighboursTemp = np.unique(directNeighboursTemp)
        # remove P itself from list of neighbours
        directNeighboursTemp = np.delete(directNeighboursTemp, np.where(directNeighboursTemp == vertex))

        #set total valence for vertex
        valence[vertex] = len(directNeighboursTemp)
        # print(valence[vertex])
        emptyEntries = nDirectNeighbours - len(directNeighboursTemp)
        if emptyEntries > 0:
            for i in range(emptyEntries):
                directNeighboursTemp = np.append(directNeighboursTemp, -1)

        # print(directNeighboursTemp)
        directNeighbours[vertex, :] = directNeighboursTemp

    elif nAttFaces == 4:
        for f in range(nAttFaces):
            fIndex = vertexTable[vertex, f + 8]
            if faceTable[fIndex, 1] == 4:
                pointPos = np.where(faceTable[fIndex, [2, 3, 4, 5]] == vertex)
                pointPos = pointPos[0][0]
                directNeighboursTemp = np.append(directNeighboursTemp, [faceTable[fIndex, ((pointPos - 1) % 4) + 2], faceTable[fIndex, ((pointPos + 1) % 4) + 2]])
                diagonalsTemp = np.append(diagonalsTemp, faceTable[fIndex, ((pointPos + 2) % 4) + 2])
            elif faceTable[fIndex , 1] == 3:
                directNeighboursTemp = np.append(directNeighboursTemp, faceTable[fIndex, [2, 3, 4]])
                # get faces that share two vertices with faceTable[fIndex , 1]
                for i, diagCell in enumerate(faceTable[:, [2, 3, 4, 5]]):
                    if diagCell[3] == -1:
                        diagCell = np.delete(diagCell, 3)
                    mask = np.in1d(diagCell, faceTable[fIndex, [2, 3, 4, 5]]) # check if elements in arg1 are present in arg2, returns vector of length arg1
                    if np.sum(mask) == 2 and vertex not in diagCell:
                        diagonalsTemp = np.append(diagonalsTemp, diagCell)
        # remove duplicate points
        directNeighboursTemp = np.unique(directNeighboursTemp)
        # remove P itself from list of neighbours
        directNeighboursTemp = np.delete(directNeighboursTemp, np.where(directNeighboursTemp == vertex))
        # remove direct neighbours from list of diagonals
        diagonalsTemp = np.setdiff1d(diagonalsTemp, directNeighboursTemp)

        #set total valence for vertex
        valence[vertex] = 4
        #set number of diagonals for vertex
        nDiagonals[vertex] = len(diagonalsTemp)

        for i in range(4, nDirectNeighbours):
            directNeighboursTemp = np.append(directNeighboursTemp, -1)

        if len(diagonalsTemp) < 8:
            for i in range(8 -len(diagonalsTemp)):
                diagonalsTemp = np.append(diagonalsTemp, -1)

        directNeighbours[vertex, :] = directNeighboursTemp
        diagonals[vertex, :] = diagonalsTemp

    elif nAttFaces == 3:?


print(valence[0:50])
print(directNeighbours[0:50, :])
print(diagonals[0:50, :])

###########################################################
