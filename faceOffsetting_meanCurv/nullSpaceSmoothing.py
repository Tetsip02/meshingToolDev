import meshio
import numpy as np
# import math
# from ../helpers import *
# import json
import pickle
from scipy.linalg import null_space


#determinant of matrix a
def det(a):
    return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]

#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

#dot product of vectors a and b
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#cross product of vectors a and b
def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)

#area of polygon poly
def area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)





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

def get_smoothed_normals(points, triangles, quads, tri_neighbours, quad_neighbours, nIt):
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

    # vertexNormal = vertexNormal / np.linalg.norm(vertexNormal, axis = 1)[: , None]
    triNormal_norm = np.linalg.norm(triNormal, axis = 1)
    quadNormal_norm = np.linalg.norm(quadNormal, axis = 1)
    # print(tri_norm.shape)

    # print(vertexNormal[0])
    for _ in range(nIt):
        for i, triFace in enumerate(triangles):
            triNormal[i, :] = np.mean((vertexNormal[triFace]), axis = 0)
            triNormal[i, :] *= triNormal_norm[i] / np.linalg.norm(triNormal[i, :])
        for i, quadFace in enumerate(quads):
            quadNormal[i, :] = np.mean((vertexNormal[quadFace]), axis = 0)
            quadNormal[i, :] *= quadNormal_norm[i] / np.linalg.norm(quadNormal[i, :])
        vertexNormal *= 0
        for vertex, (triNeighbours, quadNeighbours) in enumerate(zip(tri_neighbours, quad_neighbours)):
            vertexNormal[vertex] += triNormal[triNeighbours].sum(axis = 0) + quadNormal[quadNeighbours].sum(axis = 0)
        # print(vertexNormal[0])

    # normalize
    vertexNormal_norm = vertexNormal / np.linalg.norm(vertexNormal, axis = 1)[: , None]

    return vertexNormal, vertexNormal_norm

def centroid(vertexes):
     _x_list = [vertex [0] for vertex in vertexes]
     _y_list = [vertex [1] for vertex in vertexes]
     _z_list = [vertex [2] for vertex in vertexes]
     _len = len(vertexes)
     _x = sum(_x_list) / _len
     _y = sum(_y_list) / _len
     _z = sum(_z_list) / _len
     return(_x, _y, _z)


surface_triangles_ini, surface_quads_ini, surface_points = read_surface("../testGeometries/sphere.obj")

######################################################################

#change orientation
surface_triangles = surface_triangles_ini.copy()
surface_triangles[:,[1, 2]] = surface_triangles[:,[2, 1]]
surface_quads = surface_quads_ini.copy()
surface_quads[:,[1, 3]] = surface_quads[:,[3, 1]]

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

############################################

# get face weights
vertex = 0
print(directValence[vertex])
print(triFaceIndices[vertex])
print(quadFaceIndices[vertex])
print(directNeighbours[vertex])

#########################################

nLayers = 1
GR = 1.3
firstLayerHeight = 0.005
nNormalSmoothingIt = 7
layerHeight = np.arange(nLayers + 1)
layerHeight = firstLayerHeight * GR ** layerHeight
level1 = surface_points.copy()
points = level1.copy()
level1_normals, level1_normals_norm = get_smoothed_normals(level1, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices,nNormalSmoothingIt)
# level2 = level1 + layerHeight[0] * level1_normals_norm
for layer in range(nLayers):
    print(layer)
    level1_normals, level1_normals_norm = get_smoothed_normals(level1, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices, nNormalSmoothingIt)
    level2 = level1 + layerHeight[layer] * level1_normals_norm
    level2_normals, level2_normals_norm = get_smoothed_normals(level2, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices, nNormalSmoothingIt)
    d = np.ndarray((nPoints, 3), dtype = np.float64) * 0
    k = np.ones(nPoints, dtype = np.int) * 1 # 3 for corners, 2 for ridges, 1 for smooth vertex
    k_real = np.ones(nPoints, dtype = np.int) * 1
    # if layer == 0:
    #     k *= 3
    for vertex in range(nPoints):
        N = level2_normals_norm[directNeighbours[vertex]]


        W = np.linalg.norm(level2_normals[directNeighbours[vertex]], axis = 1)[: , None] * np.eye(directValence[vertex]) #may need to normalize weights
        # weight1 = triFaceIndices[vertex]

        a = np.dot(N, level2[vertex])
        A = np.transpose(N) @ W @ N

        b = np.transpose(N) @ W @ a
        # x = np.linalg.solve(A, b)
        lam, e = np.linalg.eig(A)
        order = np.argsort(-lam)
        # d = np.zeros(3)[: , None]
        # print(k[vertex])
        for i in range(k[vertex]):
            # d[vertex][: , None] += (e[:, order[i]][: , None] @ (np.transpose(b[: , None]) @ e[:, order[i]][: , None]))/lam[order[i]]
            # print(b[: , None].shape)
            # print(e[:, order[i]][: , None].shape)
            # d[vertex][: , None] += np.transpose(( np.transpose(e[:, order[i]][: , None]) @ ( b[: , None] @ np.transpose(e[:, order[i]][: , None]) ) ) / lam[order[i]])
            d[vertex][: , None] += ( (np.transpose(e[:, order[i]][: , None]) @ b[: , None]) * e[:, order[i]][: , None] ) / lam[order[i]]

        # print("A", A)
        # print("N(A)", null_space(A, rcond=1))

        # T = null_space(A, rcond=1)
        # if k_real[vertex] == 2:
        #     T = T[:, 0]
        # elif k_real[vertex] == 1:
        #     T = T[:, [0, 1]]
        # elif k_real == 3:
        #     T *= 0
        # # print(T.shape)
        # c = centroid()



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
