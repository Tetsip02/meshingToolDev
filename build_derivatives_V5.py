import meshio
import numpy as np
# import math
from helpers import *
# import json
import pickle

np.set_printoptions(precision=16)

surface_triangles, surface_quads, surface_points = read_surface("./testGeometries/sphere.obj")
# d1 = 0.1
# d2 = 0.12
# level1, level2, level3 = build_ref_mesh_points(surface_points, surface_triangles, surface_quads, d1, d2, 1)
# nSurfacePoints = len(surface_points)
# level1_triangles = surface_triangles
# level1_quads = surface_quads
# level2_triangles = level1_triangles + nSurfacePoints
# level2_quads = level1_quads + nSurfacePoints
# level3_triangles = level2_triangles + nSurfacePoints
# level3_quads = level2_quads + nSurfacePoints
# allPoints = np.concatenate((level1, level2, level3))

##########################################################################################

with open("./dataOutput/V2_triFaceIndices.txt", "rb") as fp:   # Unpickling
    triFaceIndices = pickle.load(fp)
with open("./dataOutput/V2_quadFaceIndices.txt", "rb") as fp:   # Unpickling
    quadFaceIndices = pickle.load(fp)
with open("./dataOutput/V2_directValence.txt", "rb") as fp:   # Unpickling
    directValence = pickle.load(fp)
with open("./dataOutput/V2_directNeighbours.txt", "rb") as fp:   # Unpickling
    directNeighbours = pickle.load(fp)
with open("./dataOutput/V2_simpleDiagonals.txt", "rb") as fp:   # Unpickling
    simpleDiagonals = pickle.load(fp)
with open("./dataOutput/V2_interDiagonals.txt", "rb") as fp:   # Unpickling
    interDiagonals = pickle.load(fp)

######################################################################

nPoints = len(surface_points)
nTriFaces = len(surface_triangles)
nQuadFaces = len(surface_quads)
nFaces = nTriFaces + nQuadFaces

######################################

# build face normals
triNormal = np.ndarray((nTriFaces, 3), dtype = np.float64)
quadNormal = np.ndarray((nQuadFaces, 3), dtype = np.float64)
for i, triFace in enumerate(surface_triangles):
    coor = surface_points[triFace, :]
    triNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
for i, quadFace in enumerate(surface_quads):
    coor = surface_points[quadFace, :]
    quadNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])

# build vertex normals
vertexNormal = np.zeros([nPoints, 3], dtype = object)
for vertex, (triNeighbours, quadNeighbours) in enumerate(zip(triFaceIndices, quadFaceIndices)):
    vertexNormal[vertex] += triNormal[triNeighbours].sum(axis = 0) + quadNormal[quadNeighbours].sum(axis = 0)

#########################################################

# create normalized normals
triNormal_norm = triNormal / np.linalg.norm(triNormal, axis = 1)[: , None]

quadNormal_norm = quadNormal / np.linalg.norm(quadNormal, axis = 1)[: , None]

##########################################################

# extrude points / build level 2
print(surface_points.shape)
nextLevel = np.zeros(surface_points.shape)
d = 0.05

triFaceTest = np.array([3, 4, 5])
print("point 3", surface_points[3])
print("point 4", surface_points[4])
print("point 5", surface_points[5])
print("normal 3", triNormal_norm[3])
print("normal 4", triNormal_norm[4])
print("normal 5", triNormal_norm[5])
coor = surface_points[triFaceTest]
print(coor)
n = triNormal_norm[triFaceTest]
print(n)
next = coor + d * n
print(next)
print(nextLevel[triFace])
nextLevel[triFace] = next
print(nextLevel[0:10])

# flip == 0
# for triFace in surface_triangles:
#     print(triFace)
    # coor = surface_points[triFace, :]
    # print(coor)
    # print(triNormal_norm[triFace])
    # nextCoor = coor + d * coor
    # print()


####################################################

# extrude points / build level 2
# print(surface_points.shape)


def extrudePoints(surfacePoints, triangle_cells, quad_cells, t, flip = 0):
    ##initialize layer 1 points##
    l1_points = np.empty(surfacePoints.shape)
    ##get layer 1 points##
    for cell in triangle_cells:
        cellPoints = np.array([surfacePoints[cell[0]], surfacePoints[cell[1]], surfacePoints[cell[2]]])
        cellNormal = getNormals(cellPoints)
        if flip == 1:
            cellNormal = cellNormal * (-1)
        l1_cellPoints = cellPoints + t * cellNormal
        l1_points[cell[0]] = l1_cellPoints[0]
        l1_points[cell[1]] = l1_cellPoints[1]
        l1_points[cell[2]] = l1_cellPoints[2]
    for cell in quad_cells:
        cellPoints = np.array([surfacePoints[cell[0]], surfacePoints[cell[1]], surfacePoints[cell[2]], surfacePoints[cell[3]]])
        cellNormal = getNormals(cellPoints)
        if flip == 1:
            cellNormal = cellNormal * (-1)
        l1_cellPoints = cellPoints + t * cellNormal
        l1_points[cell[0]] = l1_cellPoints[0]
        l1_points[cell[1]] = l1_cellPoints[1]
        l1_points[cell[2]] = l1_cellPoints[2]
        l1_points[cell[3]] = l1_cellPoints[3]

    return l1_points
