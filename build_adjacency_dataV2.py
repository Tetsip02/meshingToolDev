import meshio
import numpy as np
# import math
from helpers import *

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

######################################################################

nPoints = len(surface_points)
nTriFaces = len(surface_triangles)
nQuadFaces = len(surface_quads)
nFaces = nTriFaces + nQuadFaces


# ######################################
#
# # build faceIndices for each point
# triFaceIndices = [] # [nPoints, number_of_attached_triangle_faces]
# quadFaceIndices = [] # [nPoints, number_of_attached_quad_faces]
# directValence = [] # [nPoints, 1]
# directNeighbours = []
# for index in range(nPoints):
#     # get face indices for each point split into quad and tri lists
#     mask1 = np.isin(surface_triangles, index)
#     triFaceIndices.append(np.nonzero(mask1)[0])
#     mask2 = np.isin(surface_quads, index)
#     quadFaceIndices.append(np.nonzero(mask2)[0])
#
#     # get total number of attached faces for each point = direct valence
#     indexValence = len(np.nonzero(mask1)[0]) + len(np.nonzero(mask2)[0])
#     directValence.append(indexValence)
#
#     # get relevant neighbour nodes used later for calculating derivatives
#     # initialize direct neighbours for point index
#     indexNeighbours = []
#
#     if indexValence > 4:
#         # add all points of attached triangles
#         for faceIndex in np.nonzero(mask1)[0]:
#             indexNeighbours.extend(surface_triangles[faceIndex , :])
#
#         # add all points except diagonal from attached quads
#         for faceIndex, pos in zip(np.nonzero(mask2)[0], np.nonzero(mask2)[1]):
#             face = surface_quads[faceIndex, :]
#             indexNeighbours.append(surface_quads[faceIndex, (pos - 1) % 4])
#             indexNeighbours.append(surface_quads[faceIndex, (pos + 1) % 4])
#
#         # remove duplicate points ##
#         indexNeighbours = list(set(indexNeighbours))
#         # remove point itself from list of neighbours
#         try:
#             indexNeighbours.remove(index)
#         except ValueError:
#             pass
#
#     directNeighbours.append(indexNeighbours)
#
# # print(directNeighbours)
# # print(len(directNeighbours))
#
#
# ##########################################

directNeighbours = []
diagonalsTri = []
diagonalsQuad = []
simpleDiagonals = []
interDiagonals = []

index = 6
mask1 = np.isin(surface_triangles, index)
# triFaceIndices.append(np.nonzero(mask1)[0])
print(np.nonzero(mask1))
mask2 = np.isin(surface_quads, index)
# quadFaceIndices.append(np.nonzero(mask2)[0])
print(np.nonzero(mask2))
print(surface_triangles[2, :])
print(surface_quads[[1, 4, 18], :])
directValence = len(np.nonzero(mask1)[0]) + len(np.nonzero(mask2)[0])
print(directValence)

if directValence == 4:
    # add all points of attached triangles
    for faceIndex, pos in zip(np.nonzero(mask1)[0], np.nonzero(mask1)[1]):
        directNeighbours.extend(surface_triangles[faceIndex , :])

    # find faces that share two nodes with attached triangles, if thats a triangle add to diagonals, if quad add mean of diagonals, skip for now

    # seperate nodes of attached quad faces into direct neighbours and diagonals
    for faceIndex, pos in zip(np.nonzero(mask2)[0], np.nonzero(mask2)[1]):
        directNeighbours.append(surface_quads[faceIndex, (pos - 1) % 4])
        directNeighbours.append(surface_quads[faceIndex, (pos + 1) % 4])
        simpleDiagonals.append(surface_quads[faceIndex, (pos + 2) % 4])

print(directNeighbours)
print(simpleDiagonals)

    # for f in range(nAttFaces):
    #     fIndex = vertexTable[vertex, f + 8]
    #     if faceTable[fIndex, 1] == 4:
    #         pointPos = np.where(faceTable[fIndex, [2, 3, 4, 5]] == vertex)
    #         pointPos = pointPos[0][0]
    #         directNeighboursTemp = np.append(directNeighboursTemp, [faceTable[fIndex, ((pointPos - 1) % 4) + 2], faceTable[fIndex, ((pointPos + 1) % 4) + 2]])
    #         diagonalsTemp = np.append(diagonalsTemp, faceTable[fIndex, ((pointPos + 2) % 4) + 2])
    #     elif faceTable[fIndex , 1] == 3:
    #         directNeighboursTemp = np.append(directNeighboursTemp, faceTable[fIndex, [2, 3, 4]])
    #         # get faces that share two vertices with faceTable[fIndex , 1]
    #         for i, diagCell in enumerate(faceTable[:, [2, 3, 4, 5]]):
    #             if diagCell[3] == -1:
    #                 diagCell = np.delete(diagCell, 3)
    #             mask = np.in1d(diagCell, faceTable[fIndex, [2, 3, 4, 5]]) # check if elements in arg1 are present in arg2, returns vector of length arg1
    #             if np.sum(mask) == 2 and vertex not in diagCell:
    #                 diagonalsTemp = np.append(diagonalsTemp, diagCell)


    # # remove duplicate points
    # directNeighboursTemp = np.unique(directNeighboursTemp)
    # # remove P itself from list of neighbours
    # directNeighboursTemp = np.delete(directNeighboursTemp, np.where(directNeighboursTemp == vertex))
    # # remove direct neighbours from list of diagonals
    # diagonalsTemp = np.setdiff1d(diagonalsTemp, directNeighboursTemp)
    #
    # #set total valence for vertex
    # valence[vertex] = 4
    # #set number of diagonals for vertex
    # nDiagonals[vertex] = len(diagonalsTemp)
    #
    # for i in range(4, nDirectNeighbours):
    #     directNeighboursTemp = np.append(directNeighboursTemp, -1)
    #
    # if len(diagonalsTemp) < 8:
    #     for i in range(8 -len(diagonalsTemp)):
    #         diagonalsTemp = np.append(diagonalsTemp, -1)
    #
    # directNeighbours[vertex, :] = directNeighboursTemp
    # diagonals[vertex, :] = diagonalsTemp




    # mylist = list(set(directNeighbours))
    # print(mylist)
    # test = np.array(directNeighbours, dtype = object)
    # print(directNeighbours)
    # directNeighbours = np.ndarray.flatten(directNeighbours)

    # for f in range(nAttFaces):
    #     fIndex = vertexTable[vertex, f + 8]
    #     if faceTable[fIndex, 1] == 3:
    #         directNeighboursTemp = np.append(directNeighboursTemp, faceTable[fIndex, [2, 3, 4]])
    #     elif faceTable[fIndex , 1] == 4:
    #         pointPos = np.where(faceTable[fIndex, [2, 3, 4, 5]] == vertex)
    #         pointPos = pointPos[0][0]
    #         directNeighboursTemp = np.append(directNeighboursTemp, [faceTable[fIndex, ((pointPos - 1) % 4) + 2], faceTable[fIndex, ((pointPos + 1) % 4) + 2]])
    # # remove duplicate points ##
    # directNeighboursTemp = np.unique(directNeighboursTemp)
    # # remove P itself from list of neighbours
    # directNeighboursTemp = np.delete(directNeighboursTemp, np.where(directNeighboursTemp == vertex))
    #
    #
    # valence[vertex] = len(directNeighboursTemp)
    #
    # emptyEntries = nDirectNeighbours - len(directNeighboursTemp)
    # if emptyEntries > 0:
    #     for i in range(emptyEntries):
    #         directNeighboursTemp = np.append(directNeighboursTemp, -1)
    #
    # directNeighbours[vertex, :] = directNeighboursTemp


# print(directValence[0:20])


# nTriIndices = [len(x) for x in triFaceIndices]
# nQuadIndices = [len(x) for x in quadFaceIndices]
# nTriIndices = np.array([len(x) for x in triFaceIndices])
# nQuadIndices = np.array([len(x) for x in quadFaceIndices])

# directValence = nTriIndices + nQuadIndices


# vertex = 0
# dVal = directValence[vertex]
# nAttFaces = directValence[vertex]

# if nAttFaces > 4:
#     for f in range(nAttFaces):
#         fIndex = vertexTable[vertex, f + 8]
#         if faceTable[fIndex, 1] == 3:
#             directNeighboursTemp = np.append(directNeighboursTemp, faceTable[fIndex, [2, 3, 4]])
#         elif faceTable[fIndex , 1] == 4:
#             pointPos = np.where(faceTable[fIndex, [2, 3, 4, 5]] == vertex)
#             pointPos = pointPos[0][0]
#             directNeighboursTemp = np.append(directNeighboursTemp, [faceTable[fIndex, ((pointPos - 1) % 4) + 2], faceTable[fIndex, ((pointPos + 1) % 4) + 2]])
#     # remove duplicate points ##
#     directNeighboursTemp = np.unique(directNeighboursTemp)
#     # remove P itself from list of neighbours
#     directNeighboursTemp = np.delete(directNeighboursTemp, np.where(directNeighboursTemp == vertex))
#
#     #set total valence for vertex
#     valence[vertex] = len(directNeighboursTemp)
#     # print(valence[vertex])
#     emptyEntries = nDirectNeighbours - len(directNeighboursTemp)
#     if emptyEntries > 0:
#         for i in range(emptyEntries):
#             directNeighboursTemp = np.append(directNeighboursTemp, -1)
#
#     # print(directNeighboursTemp)
#     directNeighbours[vertex, :] = directNeighboursTemp

############ build neighbour data:########################

#######################################

# #determine valence of each vertex
# valence = np.ndarray((nSurfacePoints, 1), dtype = object) # number of points to be used in evaluating derivatives
# maxNeighbours = max(vertexTable[:, 7])
# if maxNeighbours < 6:
#     directNeighbours = np.ndarray((nSurfacePoints, 6), dtype = object) # entry is -1 if directNeighbours < 6
#     nDirectNeighbours = 6
# else:
#     directNeighbours = np.ndarray((nSurfacePoints, maxNeighbours), dtype = object) # entry is -1 if directNeighbours < maxNeighbours
#     nDirectNeighbours = maxNeighbours
# diagonals = np.ndarray((nSurfacePoints, 8), dtype = object)
# nDiagonals = -np.ones((nSurfacePoints, 1), dtype = int)
# # print(nDiagonals)
#
#
# ############################################################
#
# nAttFaces = vertexTable[vertex, 7]
# # direct-valence = 5 or higher
# elif nAttFaces == 4:
#     for f in range(nAttFaces):
#         fIndex = vertexTable[vertex, f + 8]
#         if faceTable[fIndex, 1] == 4:
#             pointPos = np.where(faceTable[fIndex, [2, 3, 4, 5]] == vertex)
#             pointPos = pointPos[0][0]
#             directNeighboursTemp = np.append(directNeighboursTemp, [faceTable[fIndex, ((pointPos - 1) % 4) + 2], faceTable[fIndex, ((pointPos + 1) % 4) + 2]])
#             diagonalsTemp = np.append(diagonalsTemp, faceTable[fIndex, ((pointPos + 2) % 4) + 2])
#         elif faceTable[fIndex , 1] == 3:
#             directNeighboursTemp = np.append(directNeighboursTemp, faceTable[fIndex, [2, 3, 4]])
#             # get faces that share two vertices with faceTable[fIndex , 1]
#             for i, diagCell in enumerate(faceTable[:, [2, 3, 4, 5]]):
#                 if diagCell[3] == -1:
#                     diagCell = np.delete(diagCell, 3)
#                 mask = np.in1d(diagCell, faceTable[fIndex, [2, 3, 4, 5]]) # check if elements in arg1 are present in arg2, returns vector of length arg1
#                 if np.sum(mask) == 2 and vertex not in diagCell:
#                     diagonalsTemp = np.append(diagonalsTemp, diagCell)
#     # remove duplicate points
#     directNeighboursTemp = np.unique(directNeighboursTemp)
#     # remove P itself from list of neighbours
#     directNeighboursTemp = np.delete(directNeighboursTemp, np.where(directNeighboursTemp == vertex))
#     # remove direct neighbours from list of diagonals
#     diagonalsTemp = np.setdiff1d(diagonalsTemp, directNeighboursTemp)
#
#     #set total valence for vertex
#     valence[vertex] = 4
#     #set number of diagonals for vertex
#     nDiagonals[vertex] = len(diagonalsTemp)
#
#     for i in range(4, nDirectNeighbours):
#         directNeighboursTemp = np.append(directNeighboursTemp, -1)
#
#     if len(diagonalsTemp) < 8:
#         for i in range(8 -len(diagonalsTemp)):
#             diagonalsTemp = np.append(diagonalsTemp, -1)
#
#     directNeighbours[vertex, :] = directNeighboursTemp
#     diagonals[vertex, :] = diagonalsTemp
#
# ##########################################################


# # build Face Normals
#
# triFaceNormals = np.ndarray((nTriFaces, 3), dtype = object)
#
#
# face0 = surface_triangles[0, :]
# # print(surface_triangles[0, :])
# pointCoor = np.array([surface_points[face0[0]], surface_points[face0[1]], surface_points[face0[2]]])
# testNormal = getAbsNormals(pointCoor)
# print(testNormal)




# def getAbsNormals(shape):
#     n = np.cross(shape[1,:] - shape[0,:], shape[2,:] - shape[0,:])
#
#     return n



#######################################################

# #build face normal table
# # | face index | quad or tri (4 for quad, 3 for tri) | point indices (4 entries, if triangle then last entry is -1) | face normal, not normalized (3 entries) |
#
# nFaces = len(level1_triangles) + len(level1_quads)
# faceTable = np.ndarray((nFaces,9),dtype = object)
# for i in range(nFaces):
#     faceTable[i, 0] = int(i)
#
# for i, cell in enumerate(level1_triangles):
#     faceTable[i, 1] = 3
#     faceTable[i, 2:5] = cell
#     faceTable[i, 5] = -1
#     pointCoor = np.array([level1[cell[0]], level1[cell[1]], level1[cell[2]]])
#
#     faceTable[i, 6:9] = getAbsNormals(pointCoor)
#
# for i, cell in enumerate(level1_quads):
#     faceTable[i + len(level1_triangles), 1] = 4
#     faceTable[i + len(level1_triangles), 2:6] = cell
#     pointCoor = np.array([level1[cell[0]], level1[cell[1]], level1[cell[2]], level1[cell[3]]])
#     faceTable[i + len(level1_triangles), 6:9] = getAbsNormals(pointCoor)
#
# #############################################################################
#
# #build vertex table:
# # | point index (entry 0) | point coordinates (entries 1,2,3) | number of faces attached (entry 4) | face indices of attached faces (entries 5, 6, 7...) |
#
# vertexIndex = np.ndarray((nSurfacePoints, 1), dtype = object)
# for i in range(nSurfacePoints):
#     vertexIndex[i] = i
#
# vertexTable = np.append(vertexIndex, level1, axis=1)
#
# neighbourFaces = []
# nNeighbourFaces = np.ndarray((nSurfacePoints, 1), dtype = object)
# for P in range(nSurfacePoints):
#     neighbourFacesTemp = []
#     for j in range(nFaces):
#         if P in faceTable[j, 2:6]:
#             neighbourFacesTemp.append(j)
#
#     nNeighbourFaces[P] = len(neighbourFacesTemp)
#     neighbourFaces.append(neighbourFacesTemp)
#
# vertexTable = np.append(vertexTable, nNeighbourFaces, axis=1)
#
# maxNeighbours = len(max(neighbourFaces, key=len))
#
# for i, neighbours in enumerate(neighbourFaces):
#     j = len(neighbours)
#     k = maxNeighbours - j
#     if k > 0:
#         for l in range(k):
#             neighbourFaces[i].append(-1)
# neighbourFaces = np.array(neighbourFaces)
# vertexTable = np.append(vertexTable, neighbourFaces, axis=1)
#
# ################################################################
#
# # add vertex normals to to vertex table
# allVNormals = np.zeros((nSurfacePoints, 3), dtype = object)
# for v in range(nSurfacePoints):
#     vNormal = np.zeros((1, 3), dtype = object)
#
#     nF = vertexTable[v, 4]
#
#     for f in range(nF):
#         vNormal += faceTable[vertexTable[v, 5 + f], 6:9]
#
#     norm = np.linalg.norm(vNormal)
#     vNormal = vNormal / norm
#
#     allVNormals[v, :] = vNormal
# vertexTable = np.insert(vertexTable, [4], allVNormals, axis=1)
#
# #####################################
#
# #build level 2 faceTable | not complete, need to update face normals
# level2FaceTable = faceTable.copy()
# level2FaceTable[:, 0] += nFaces
# for f in range(nFaces):
#     if level2FaceTable[f, 1] == 3:
#         level2FaceTable[f, [2, 3, 4]] += nSurfacePoints
#     elif level2FaceTable[f, 1] == 4:
#         level2FaceTable[f, [2, 3, 4, 5]] += nSurfacePoints
#
# ######################################
#
# #build level2 vertex table | incomplete, need to update vertex normals
# level2VertexTable = vertexTable.copy()
# level2VertexTable[:, 0] += nSurfacePoints
# for v in range(nSurfacePoints):
#     level2VertexTable[v, 8:(level2VertexTable[v, 7] - maxNeighbours)] += nFaces
#     level2VertexTable[v, [1, 2, 3]] = level2[v, :]
#
# #######################################
#
# #update level 2 face table normals
# for f in range(nFaces):
#     if faceTable[f, 1] == 3:
#         pointCoor = np.array([level2[faceTable[f, 2]], level2[faceTable[f, 3]], level2[faceTable[f, 4]]])
#     elif faceTable[f, 1] == 4:
#         pointCoor = np.array([level2[faceTable[f, 2]], level2[faceTable[f, 3]], level2[faceTable[f, 4]], level2[faceTable[f, 5]]])
#     level2FaceTable[f, [6, 7, 8]] = getAbsNormals(pointCoor)
#
# #######################################
#
# # update level2 vertex table vertex normals
# for v in range(nSurfacePoints):
#     vNormal = np.zeros((1, 3), dtype = object)
#     nF = vertexTable[v, 7]
#     for f in range(nF):
#         vNormal += level2FaceTable[vertexTable[v, 8 + f], 6:9]
#     norm = np.linalg.norm(vNormal)
#     vNormal = vNormal / norm
#     level2VertexTable[v, [4, 5, 6]] = vNormal


#######################################

# np.save('./dataOutput/faceTable.npy', faceTable)
# np.save('./dataOutput/level2FaceTable.npy', level2FaceTable)
# np.save('./dataOutput/vertexTable.npy', vertexTable)
# np.save('./dataOutput/level2VertexTable.npy', level2VertexTable)
