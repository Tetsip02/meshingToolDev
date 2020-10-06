import meshio
import numpy as np
# import math
# from ../helpers import *
# import json
import pickle

def read_surface(surfaceFile):
    ## write triangle cells, quad cells and surface points in numpy arrays ##
    surfaceMesh = meshio.read(surfaceFile)

    ##get cells##
    triangle_cells = np.array([])
    quad_cells = np.array([])
    for cell in surfaceMesh.cells:
        if cell.type == "triangle":
            if triangle_cells.size == 0:
                triangle_cells = cell.data
            else:
                triangle_cells = np.concatenate((triangle_cells, cell.data))
            # print(cell.data)
        elif cell.type == "quad":
            if quad_cells.size == 0:
                quad_cells = cell.data
            else:
                quad_cells = np.concatenate((quad_cells, cell.data))

    ##get surface points##
    surfacePoints = surfaceMesh.points

    return triangle_cells, quad_cells, surfacePoints

def get_layer_normals(points, triangles, quads, tri_neighbours, quad_neighbours):
    # build face normals
    triNormal = np.ndarray((nTriFaces, 3), dtype = np.float64)
    quadNormal = np.ndarray((nQuadFaces, 3), dtype = np.float64)
    for i, triFace in enumerate(triangles):
        coor = points[triFace, :]
        triNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
    for i, quadFace in enumerate(quads):
        coor = surface_points[quadFace, :]
        quadNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
    # build vertex normals
    vertexNormal = np.zeros([points.shape[0], 3], dtype = np.float64)
    for vertex, (triNeighbours, quadNeighbours) in enumerate(zip(tri_neighbours, quad_neighbours)):
        vertexNormal[vertex] += triNormal[triNeighbours].sum(axis = 0) + quadNormal[quadNeighbours].sum(axis = 0)
    # normalize
    vertexNormal_norm = vertexNormal / np.linalg.norm(vertexNormal, axis = 1)[: , None]

    return vertexNormal, vertexNormal_norm

surface_triangles_ini, surface_quads_ini, surface_points = read_surface("../testGeometries/sphere.obj")

######################################################################

#change orientation
surface_triangles = surface_triangles_ini.copy()
# surface_triangles[:,[1, 2]] = surface_triangles[:,[2, 1]]
surface_quads = surface_quads_ini.copy()
# surface_quads[:,[1, 3]] = surface_quads[:,[3, 1]]

######################################

nPoints = len(surface_points)
nTriFaces = len(surface_triangles)
nQuadFaces = len(surface_quads)
nFaces = nTriFaces + nQuadFaces

######################################

level2_triangles = surface_triangles + nPoints
level2_quads = surface_quads + nPoints
level3_triangles = level2_triangles + nPoints
level3_quads = level2_quads + nPoints

########################################################################


# build faceIndices for each point
triFaceIndices = [] # [nPoints, number_of_attached_triangle_faces]
quadFaceIndices = [] # [nPoints, number_of_attached_quad_faces]
directValence = [] # [nPoints, 1]
directNeighbours = []
for index in range(nPoints):
    # get face indices for each point split into quad and tri lists
    mask1 = np.isin(surface_triangles, index)
    triFaceIndices.append(np.nonzero(mask1)[0])
    mask2 = np.isin(surface_quads, index)
    quadFaceIndices.append(np.nonzero(mask2)[0])

    # get total number of attached faces for each point = direct valence
    indexValence = len(np.nonzero(mask1)[0]) + len(np.nonzero(mask2)[0])
    directValence.append(indexValence)

    # # initialize direct neighbours for point index
    indexNeighbours = []

    # add all points of attached triangles
    for faceIndex in np.nonzero(mask1)[0]:
        indexNeighbours.extend(surface_triangles[faceIndex , :])

    # add all points except diagonal from attached quads
    for faceIndex, pos in zip(np.nonzero(mask2)[0], np.nonzero(mask2)[1]):
        face = surface_quads[faceIndex, :]
        indexNeighbours.append(surface_quads[faceIndex, (pos - 1) % 4])
        indexNeighbours.append(surface_quads[faceIndex, (pos + 1) % 4])

    # remove duplicate points ##
    indexNeighbours = list(set(indexNeighbours))
    # remove point itself from list of neighbours
    try:
        indexNeighbours.remove(index)
    except ValueError:
        pass

    directNeighbours.append(indexNeighbours)

##################################################

nLayers = 15
GR = 1.3
firstLayerHeight = 0.005
# nSmoothingIt = 2
layerHeight = np.arange(nLayers + 1)
layerHeight = firstLayerHeight * GR ** layerHeight
level1 = surface_points.copy()
points = level1.copy()
level1_normals, level1_normals_norm = get_layer_normals(level1, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices)
# level2 = level1 + layerHeight[0] * level1_normals_norm
for layer in range(nLayers):
    print(layer)
    level1_normals, level1_normals_norm = get_layer_normals(level1, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices)
    level2 = level1 + layerHeight[layer] * level1_normals_norm
    level2_normals, level2_normals_norm = get_layer_normals(level2, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices)
    d = np.ndarray((nPoints, 3), dtype = np.float64) * 0
    k = np.ones(nPoints, dtype = np.int) * 3 # 3 for corners, 2 for ridges, 1 for smooth vertex
    for vertex in range(nPoints):
        N = level2_normals_norm[directNeighbours[vertex]]
        W = np.linalg.norm(level2_normals[directNeighbours[vertex]], axis = 1)[: , None] * np.eye(directValence[vertex]) #may need to normalize weights
        a = np.dot(N, level2[vertex])
        A = np.transpose(N) @ W @ N
        b = np.transpose(N) @ W @ a
        # x = np.linalg.solve(A, b)
        lam, e = np.linalg.eig(A)
        order = np.argsort(-lam)
        # d = np.zeros(3)[: , None]
        # print(k[vertex])
        for i in range(k[vertex]):
            d[vertex][: , None] += (e[:, order[i]][: , None] @ (np.transpose(b[: , None]) @ e[:, order[i]][: , None]))/lam[order[i]]

    points = np.append(points, d, axis = 0)
    level1 = d



################################################################
# build mesh
voxels = []
for layer in range(nLayers):
    layerVoxel = [["wedge", np.concatenate((surface_triangles + (layer * nPoints), surface_triangles + ((layer + 1) * nPoints)),axis=1)], ["hexahedron", np.concatenate((surface_quads + (layer * nPoints), surface_quads + ((layer + 1) * nPoints)),axis=1)]]
    voxels.extend(layerVoxel)
# print(points[:, 0, :].size)
# nMeshPoints = nPoints + nLayers * nPoints
# print(nMeshPoints)
# points2 = np.reshape(points, (nMeshPoints, 3), order='C')
mesh = meshio.Mesh(points = points, cells = voxels)
meshio.write("./output/testMesh.vtk", mesh, file_format="vtk", binary=False)
# meshio.write("./output/ref_mesh_test2.msh", V5_mesh, file_format="gmsh22", binary=False)
###############################################################

## get normals
# normals, normals_norm = get_layer_normals(surface_points, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices)

# vertex = 0
# N = normals_norm[directNeighbours[vertex]]
# W = np.linalg.norm(normals[directNeighbours[vertex]], axis = 1)[: , None] * np.eye(directValence[vertex]) #may need to normalize weights
# a = np.dot(N, surface_points[vertex])
# A = np.transpose(N) @ W @ N
# b = np.transpose(N) @ W @ a
# # print("A", A)
# # print("b", b)
# x = np.linalg.solve(A, b)
# # print("v", surface_points[vertex])
# # print("x", x)
#
# lam, e = np.linalg.eig(A)
#
# order = np.argsort(-lam)
# k = np.ones(nPoints, dtype = np.int) * 3 # 3 for corners, 2 for ridges, 1 for smooth vertex
# d = np.zeros(3)[: , None]
# print(k[vertex])
# for i in range(k[vertex]):
#     d += (e[:, order[i]][: , None] @ (np.transpose(b[: , None]) @ e[:, order[i]][: , None]))/lam[order[i]]
# print(d)

#########################################################

# d = np.ndarray((nPoints, 3), dtype = np.float64) * 0
# k = np.ones(nPoints, dtype = np.int) * 1 # 3 for corners, 2 for ridges, 1 for smooth vertex
# for vertex in range(nPoints):
#     N = normals_norm[directNeighbours[vertex]]
#     W = np.linalg.norm(normals[directNeighbours[vertex]], axis = 1)[: , None] * np.eye(directValence[vertex]) #may need to normalize weights
#     a = np.dot(N, surface_points[vertex])
#     A = np.transpose(N) @ W @ N
#     b = np.transpose(N) @ W @ a
#     # x = np.linalg.solve(A, b)
#     lam, e = np.linalg.eig(A)
#     order = np.argsort(-lam)
#     # d = np.zeros(3)[: , None]
#     # print(k[vertex])
#     for i in range(k[vertex]):
#         d[vertex][: , None] += (e[:, order[i]][: , None] @ (np.transpose(b[: , None]) @ e[:, order[i]][: , None]))/lam[order[i]]

###################################################################
