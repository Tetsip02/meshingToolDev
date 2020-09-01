import meshio
import numpy as np
# import math
from helpers import *

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

# # print(level2)
# print(level2[0, :])
# print(level2[0, :].shape)
# testPoint = level2[0, :]
# test2Point = allPoints[nSurfacePoints, :]
# print(test2Point)
# testSet = np.empty([3, 3])
# print(testSet)
# testSet[0, :] = allPoints[0, :]
# testSet[1, :] = allPoints[nSurfacePoints, :]
# testSet[2, :] = allPoints[nSurfacePoints * 2, :]
# # getVertexNormal()
# print(testSet)

#build face normal table
# | face index | quad or tri (4 for quad, 3 for tri) | point indices (4 entries, if triangle then last entry is -1) | face normal, not normalized (3 entries) |
# [nFaces, 9]
# print(len(level1_triangles))
nFaces = len(level1_triangles) + len(level1_quads)
# print(nFaces)
faceTable = np.ndarray((nFaces,9),dtype = object)
# faceTable = np.empty([nFaces, 9])
for i in range(nFaces):
    faceTable[i, 0] = int(i)

for i, cell in enumerate(level1_triangles):
    faceTable[i, 1] = 3
    # print(faceTable[i, 2:4])
    faceTable[i, 2:5] = cell
    faceTable[i, 5] = -1
    pointCoor = np.array([level1[cell[0]], level1[cell[1]], level1[cell[2]]])
    # print(pointCoor)
    # cellNormal = getAbsNormals(pointCoor)
    # print(cellNormal)
    faceTable[i, 6:9] = getAbsNormals(pointCoor)

for i, cell in enumerate(level1_quads):
    faceTable[i + len(level1_triangles), 1] = 4
    faceTable[i + len(level1_triangles), 2:6] = cell
    pointCoor = np.array([level1[cell[0]], level1[cell[1]], level1[cell[2]], level1[cell[3]]])
    faceTable[i + len(level1_triangles), 6:9] = getAbsNormals(pointCoor)


#build vertex table:
# | point index (entry 0) | point coordinates (entries 1,2,3) | number of faces attached (entry 4) | face indices of attached faces (entries 5, 6, 7...) |

vertexIndex = np.ndarray((nSurfacePoints, 1), dtype = object)
for i in range(nSurfacePoints):
    vertexIndex[i] = i

# print(vertexIndex)
# print(vertexIndex.shape)
vertexTable = np.append(vertexIndex, level1, axis=1)
# print(vertexTable)
# print(vertexTable.shape)

neighbourFaces = []
nNeighbourFaces = np.ndarray((nSurfacePoints, 1), dtype = object)
for P in range(nSurfacePoints):
    neighbourFacesTemp = []
    for j in range(nFaces):
        if P in faceTable[j, 2:6]:
            neighbourFacesTemp.append(j)
        # print(neighbourFacesTemp)
    nNeighbourFaces[P] = len(neighbourFacesTemp)
    neighbourFaces.append(neighbourFacesTemp)
# print(nNeighbourFaces)
# print(neighbourFaces)
vertexTable = np.append(vertexTable, nNeighbourFaces, axis=1)
# print(vertexTable)

maxNeighbours = len(max(neighbourFaces, key=len))
print(maxNeighbours)
for i, neighbours in enumerate(neighbourFaces):
    j = len(neighbours)
    k = maxNeighbours - j
    if k > 0:
        for l in range(k):
            neighbourFaces[i].append(-1)
neighbourFaces = np.array(neighbourFaces)
vertexTable = np.append(vertexTable, neighbourFaces, axis=1)

#####################################

#########################

# P = 0
# neighbourFacesTemp = []
# neighbourFaces = []
# nNeighbourFaces = np.ndarray((1, 1), dtype = object)
# for j in range(nFaces):
#     if P in faceTable[j, 2:6]:
#         neighbourFacesTemp.append(j)
# print(len(neighbourFacesTemp))
# neighbourFaces.append(neighbourFacesTemp)
# neighbourFaces.append([0, 338, 340, 343, 344, 1])
# MaxNeighbours = 6
# print(len(neighbourFaces))
# print(len(neighbourFaces[1]))
# # for i in range(len(neighbourFaces)):
# #     j = len(neighbourFaces[i])
# #     if (MaxNeighbours - j) > 0:
# #         for k in range(MaxNeighbours - j):
# #             neighbourFaces[i].append(-1)
# for i, neighbours in enumerate(neighbourFaces):
#     j = len(neighbours)
#     if (MaxNeighbours - j) > 0:
#         for k in range(MaxNeighbours - j):
#             neighbourFaces[i].append(-1)
# print(neighbourFaces)
# neighbourFaces = np.array(neighbourFaces)
# print(neighbourFaces)
# print(neighbourFaces.shape)

##############################
