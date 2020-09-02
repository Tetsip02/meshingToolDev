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
else:
    directNeighbours = np.ndarray((nSurfacePoints, maxNeighbours), dtype = object) # entry is -1 if directNeighbours < maxNeighbours
diagonals = np.ndarray((nSurfacePoints, 8), dtype = object)
# print(maxNeighbours)


############################################################

for vertex in range(nSurfacePoints):
    # initialize neighbours for vertex
    directNeighboursTemp = np.array([], dtype=np.int)

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

###########################################################

# nerighbours for 5 or more attached faces
directNeighboursTemp = np.array([], dtype=np.int)
vertex = 0
nAttFaces = vertexTable[vertex, 7]
if nAttFaces > 4: # cases where val => 5
    valence[vertex] = nAttFaces
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
    # remove P itself from list of neighbours ##
    #     Pindex = np.where(neighbours == P)
    directNeighboursTemp = np.delete(directNeighboursTemp, np.where(directNeighboursTemp == vertex))

# neighbours and diagonals for 4 attached faces
directNeighboursTemp = np.array([], dtype=np.int)
diagonalsTemp = np.array([], dtype=np.int)
vertex = 3
nAttFaces = vertexTable[vertex, 7]
print(nAttFaces)
if nAttFaces == 4: # cases where val => 5
    # valence[vertex] = nAttFaces
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

# neighbours for 3 attached faces
directNeighboursTemp = np.array([], dtype=np.int)
# diagonalsTemp = np.array([], dtype=np.int)
vertex = 980
nAttFaces = vertexTable[vertex, 7]
print(nAttFaces)
if nAttFaces == 3: # cases where val => 5
    # valence[vertex] = nAttFaces
    for f in range(nAttFaces):
        fIndex = vertexTable[vertex, f + 8]
        if faceTable[fIndex, 1] == 4:
            directNeighboursTemp = np.append(directNeighboursTemp, faceTable[fIndex, [2, 3, 4, 5]])
        elif faceTable[fIndex , 1] == 3:
            directNeighboursTemp = np.append(directNeighboursTemp, faceTable[fIndex, [2, 3, 4]])
            # get cells that share two vertices with faceTable[fIndex , [2, 3, 4]]
            # for cell2 in triangle_cells:
            for i, diagCell in enumerate(faceTable[:, [2, 3, 4, 5]]):
                if diagCell[3] == -1:
                    diagCell = np.delete(diagCell, 3)
                mask = np.in1d(diagCell, faceTable[fIndex, [2, 3, 4, 5]]) # check if elements in arg1 are present in arg2, returns vector of length arg1
                if np.sum(mask) == 2:
                    directNeighboursTemp = np.append(directNeighboursTemp, diagCell)

    # # remove duplicate points
    directNeighboursTemp = np.unique(directNeighboursTemp)
    # remove P itself from list of neighbours
    directNeighboursTemp = np.delete(directNeighboursTemp, np.where(directNeighboursTemp == vertex))

############################################################




# nerighbours for 5 or more attached faces
# directNeighboursTemp = np.array([], dtype=np.int)
# vertex = 0
# nAttFaces = vertexTable[vertex, 7]
# if nAttFaces > 4: # cases where val => 5
#     valence[vertex] = nAttFaces
#     for f in range(nAttFaces):
#         fIndex = vertexTable[vertex, f + 8]
#         if faceTable[fIndex, 1] == 3:
#             directNeighboursTemp = np.append(directNeighboursTemp, faceTable[fIndex, [2, 3, 4]])
#         elif faceTable[fIndex , 1] == 4:
#             pointPos = np.where(faceTable[fIndex, [2, 3, 4, 5]] == vertex)
#             print("face ", fIndex, "pointPosition", pointPos)
#             pointPos = pointPos[0][0]
#             print("face ", fIndex, "pointPosition", pointPos)
#             directNeighboursTemp = np.append(directNeighboursTemp, [faceTable[fIndex, ((pointPos - 1) % 4) + 2], faceTable[fIndex, ((pointPos + 1) % 4) + 2]])
#             print("directNeighboursTemp ", directNeighboursTemp)
#             # directNeighboursTemp = np.append(directNeighboursTemp, [faceTable[fIndex, [2, 3, 4, 5][(cellIndex-1)%4], faceTable[fIndex, [2, 3, 4, 5][(cellIndex+1)%4]])
#     # remove duplicate points ##
#     directNeighboursTemp = np.unique(directNeighboursTemp)
#     # remove P itself from list of neighbours ##
#     #     Pindex = np.where(neighbours == P)
#     directNeighboursTemp = np.delete(directNeighboursTemp, np.where(directNeighboursTemp == vertex))
#
# print("vertex zero ", vertexTable[0, :]) # 5 neighbour faces: 0 338 340 343 344 -1
# print("face 0 (tri) ", faceTable[0, [2, 3, 4, 5]])
# print("face 338 (quad) ", faceTable[338, [2, 3, 4, 5]])
# print("face 340 (quad) ", faceTable[340, [2, 3, 4, 5]])
# print("face 343 (quad) ", faceTable[343, [2, 3, 4, 5]])
# print("face 344 (quad) ", faceTable[344, [2, 3, 4, 5]])
# print("directNeighboursTemp ", directNeighboursTemp)

# # neighbours and diagonals for 4 attached faces
# directNeighboursTemp = np.array([], dtype=np.int)
# diagonalsTemp = np.array([], dtype=np.int)
# vertex = 3
# nAttFaces = vertexTable[vertex, 7]
# print(nAttFaces)
# if nAttFaces == 4: # cases where val => 5
#     # valence[vertex] = nAttFaces
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
#                     # print("diagCell", diagCell)
#                     diagonalsTemp = np.append(diagonalsTemp, diagCell)
#     # remove duplicate points
#     directNeighboursTemp = np.unique(directNeighboursTemp)
#     # remove P itself from list of neighbours
#     directNeighboursTemp = np.delete(directNeighboursTemp, np.where(directNeighboursTemp == vertex))
#     # remove direct neighbours from list of diagonals
#     diagonalsTemp = np.setdiff1d(diagonalsTemp, directNeighboursTemp)
# # print("directNeighboursTemp", directNeighboursTemp)
# # print("diagonalsTemp", diagonalsTemp)
# print("vertex 1 ", vertexTable[1, [1, 2, 3]])
# print("vertex 5 ", vertexTable[5, [1, 2, 3]])
# print("vertex 4 ", vertexTable[4, [1, 2, 3]])
# print("vertex 213 ", vertexTable[213, [1, 2, 3]])
# print("vertex 15 ", vertexTable[15, [1, 2, 3]])
# print("vertex 214 ", vertexTable[214, [1, 2, 3]])
# print("vertex 0 ", vertexTable[0, [1, 2, 3]])
# print("vertex 6 ", vertexTable[6, [1, 2, 3]])
# print("vertex 7 ", vertexTable[7, [1, 2, 3]])
# print("vertex 216 ", vertexTable[216, [1, 2, 3]])
#
#
# print("vertex three ", vertexTable[3, :]) # 4 neighbour faces: 1 318 319 338 -1 -1
# print("face 1 (tri) ", faceTable[1, [2, 3, 4, 5]])
# print("face 318 (quad) ", faceTable[318, [2, 3, 4, 5]])
# print("face 319 (quad) ", faceTable[319, [2, 3, 4, 5]])
# print("face 338 (quad) ", faceTable[338, [2, 3, 4, 5]])
# print("directNeighboursTemp ", directNeighboursTemp)
# print("diagonalsTemp ", diagonalsTemp)

# # neighbours for 3 attached faces
# directNeighboursTemp = np.array([], dtype=np.int)
# # diagonalsTemp = np.array([], dtype=np.int)
# vertex = 980
# nAttFaces = vertexTable[vertex, 7]
# print(nAttFaces)
# if nAttFaces == 3: # cases where val => 5
#     # valence[vertex] = nAttFaces
#     for f in range(nAttFaces):
#         fIndex = vertexTable[vertex, f + 8]
#         if faceTable[fIndex, 1] == 4:
#             directNeighboursTemp = np.append(directNeighboursTemp, faceTable[fIndex, [2, 3, 4, 5]])
#         elif faceTable[fIndex , 1] == 3:
#             directNeighboursTemp = np.append(directNeighboursTemp, faceTable[fIndex, [2, 3, 4]])
#             # get cells that share two vertices with faceTable[fIndex , [2, 3, 4]]
#             # for cell2 in triangle_cells:
#             for i, diagCell in enumerate(faceTable[:, [2, 3, 4, 5]]):
#                 if diagCell[3] == -1:
#                     diagCell = np.delete(diagCell, 3)
#                 mask = np.in1d(diagCell, faceTable[fIndex, [2, 3, 4, 5]]) # check if elements in arg1 are present in arg2, returns vector of length arg1
#                 if np.sum(mask) == 2:
#                     directNeighboursTemp = np.append(directNeighboursTemp, diagCell)
#
#     # # remove duplicate points
#     directNeighboursTemp = np.unique(directNeighboursTemp)
#     # remove P itself from list of neighbours
#     directNeighboursTemp = np.delete(directNeighboursTemp, np.where(directNeighboursTemp == vertex))
#     # # remove direct neighbours from list of diagonals
#     # diagonalsTemp = np.setdiff1d(diagonalsTemp, directNeighboursTemp)
#
# print("vertex 980 ", vertexTable[980, :]) # 3 neighbour faces: 995 1019 1070 -1 -1 -1
# print("face 995 (quad) ", faceTable[995, [2, 3, 4, 5]])
# print("face 1019 (quad) ", faceTable[1019, [2, 3, 4, 5]])
# print("face 1070 (quad) ", faceTable[1070, [2, 3, 4, 5]])
# print("directNeighboursTemp ", directNeighboursTemp)
#
# print("vertex 715 ", vertexTable[715, [1, 2, 3]])
# print("vertex 817 ", vertexTable[817, [1, 2, 3]])
# print("vertex 979 ", vertexTable[979, [1, 2, 3]])
# print("vertex 993 ", vertexTable[993, [1, 2, 3]])
# print("vertex 994 ", vertexTable[994, [1, 2, 3]])
# print("vertex 995 ", vertexTable[995, [1, 2, 3]])






######################################


# ## for M >= 5##
# P1 = 10 # for P=10 n_neighbours = 5 (2xtri, 3xquad)
# v = 0
# for cell in triangle_cells:
#     if P1 in cell:
#         v+=1
# for cell in quad_cells:
#     if P1 in cell:
#         v+=1
# print("valence", v)
# print(" ")
# r_xi = np.zeros([3, 1])
# r_xi_1 = np.zeros([3, v])
# r_0 = surfacePoints[P1]
# # print(r_xi)
# print("P1", r_0)
# # P1neighbours, P1diagonals = getNeighboursV4(P1, triangle_cells, quad_cells)
# P1neighbours = getNeighbours(P1, triangle_cells, quad_cells)
# print("neighbours", P1neighbours)
# # print("diagonals", P1diagonals)
# # print("xi", r_xi_1)
# print("neighbour A", surfacePoints[8])
# print("neighbour B", surfacePoints[9])
# print("neighbour C", surfacePoints[11])
# print("neighbour D", surfacePoints[220])
# print("neighbour E ", surfacePoints[221])
#
# # get x_xi
# for i, P in enumerate(P1neighbours):
#     r_xi_1[:, i] = surfacePoints[P]
# print("r_xi_1", r_xi_1)
#
# cos_theta_m = np.zeros([v, 1])
# for i in range(v):
#     cos_theta_m[i] = np.cos((i * 2 * np.pi) / v)
# print("cos_theta_m", cos_theta_m, "sum: ", np.sum(cos_theta_m))
#
# r_xi = (2 / v) * np.dot(r_xi_1, cos_theta_m)
# print("r_xi: ", r_xi)
