import meshio
import numpy as np
# import math
from helpers import *
# import json
import pickle

np.set_printoptions(precision=16)

surface_triangles_ini, surface_quads_ini, surface_points = read_surface("./testGeometries/sphere.obj")
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
vertexNormal = np.zeros([nPoints, 3], dtype = np.float64)
for vertex, (triNeighbours, quadNeighbours) in enumerate(zip(triFaceIndices, quadFaceIndices)):
    vertexNormal[vertex] += triNormal[triNeighbours].sum(axis = 0) + quadNormal[quadNeighbours].sum(axis = 0)

#########################################################

# create normalized normals
triNormal_norm = triNormal / np.linalg.norm(triNormal, axis = 1)[: , None]

quadNormal_norm = quadNormal / np.linalg.norm(quadNormal, axis = 1)[: , None]

vertexNormal_norm = vertexNormal / np.linalg.norm(vertexNormal, axis = 1)[: , None]

###################################################

# reference mesh extrusion by vertex normals
d = 0.05
nextLevel = surface_points + d * vertexNormal_norm
# print(nextLevel)

################################################

# temporary reference mesh
# level1 = surface_points.copy()
d1 = 0.05
d2 = 0.07
level2 = surface_points + d1 * vertexNormal_norm
level2_triNormal = np.ndarray((nTriFaces, 3), dtype = np.float64)
level2_quadNormal = np.ndarray((nQuadFaces, 3), dtype = np.float64)
level2_vertexNormal = np.zeros([nPoints, 3], dtype = np.float64)
for i, triFace in enumerate(surface_triangles):
    coor = level2[triFace, :]
    level2_triNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
for i, quadFace in enumerate(surface_quads):
    coor = level2[quadFace, :]
    level2_quadNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
for vertex, (triNeighbours, quadNeighbours) in enumerate(zip(triFaceIndices, quadFaceIndices)):
    level2_vertexNormal[vertex] += level2_triNormal[triNeighbours].sum(axis = 0) + level2_quadNormal[quadNeighbours].sum(axis = 0)
level2_vertexNormal_norm = level2_vertexNormal / np.linalg.norm(vertexNormal, axis = 1)[: , None]
level3 = level2 + d2 * level2_vertexNormal_norm

##############################################################

level2_triangles = surface_triangles + nPoints
level2_quads = surface_quads + nPoints
level3_triangles = level2_triangles + nPoints
level3_quads = level2_quads + nPoints
refMesh_points = np.concatenate((surface_points, level2, level3))
voxels1 = [["wedge", np.concatenate((surface_triangles, level2_triangles),axis=1)], ["hexahedron", np.concatenate((surface_quads, level2_quads),axis=1)]]
voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)], ["hexahedron", np.concatenate((level2_quads, level3_quads),axis=1)]]
voxels1and2 = voxels1
voxels1and2.extend(voxels2)
V5_mesh = meshio.Mesh(points = refMesh_points, cells = voxels1and2)
meshio.write("./output/V5_mesh.vtk", V5_mesh, file_format="vtk", binary=False)
# meshio.write("./output/ref_mesh_test2.msh", V5_mesh, file_format="gmsh22", binary=False)

###############################################################

# xi_trig_mat = []
# eta_trig_mat = []
# for val in directValence:
#     if val > 4:
#         xi_trig_mat.append(np.transpose(np.cos((np.arange(val) * 2 * np.pi) / val)[:, None]))
#     elif val == 4
# print(xi_trig_mat[1])

###########################################################

## get zeta derivatives
#get level2 vertex normals:
zeta = (vertexNormal * d1 + level2_vertexNormal * d2) / (d1 + d2)
zeta_zeta = (1 * (level2_vertexNormal - vertexNormal)) / (d1 + d2) ## double check if the 2 multiplier if correct
# print(zeta[0:10])
# print(zeta_zeta[0:10])
#
# r_zeta = (vertexTable[:, [4, 5, 6]] * d1 + level2VertexTable[:, [4, 5, 6]] * d2) / (d1 + d2)
# r_zeta_zeta = (level2VertexTable[:, [4, 5, 6]] - vertexTable[:, [4, 5, 6]]) / (d1 + d2)


#############################################################
## this can probably be sped up a lot by using masks and not looping through each individually

xi = np.ndarray((nPoints, 3), dtype = np.float64)
eta = np.ndarray((nPoints, 3), dtype = np.float64)
xi_xi = np.ndarray((nPoints, 3), dtype = np.float64)
eta_eta = np.ndarray((nPoints, 3), dtype = np.float64)
xi_eta = np.ndarray((nPoints, 3), dtype = np.float64)


for vertex, val in enumerate(directValence):
    if val > 4:
        neigh = level2[directNeighbours[vertex]]
        trig_base = np.arange(val)
        xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / val)[:, None])
        eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / val)[:, None])
        xi_xi_trig = 4 * np.square(xi_trig) - 1
        eta_eta_trig = 4 * np.square(eta_trig) - 1
        xi_eta_trig = xi_trig * eta_trig
        xi[vertex, :] = (2 / val) * np.dot(xi_trig, neigh)
        eta[vertex, :] = (2 / val) * np.dot(eta_trig, neigh)
        xi_xi[vertex, :] = (2 / val) * np.dot(xi_xi_trig, neigh)
        eta_eta[vertex, :] = (2 / val) * np.dot(eta_eta_trig, neigh)
        xi_eta[vertex, :] = (8 / val) * np.dot(xi_eta_trig, neigh)
    elif val == 3:
        neigh = level2[directNeighbours[vertex]]
        if interDiagonals[vertex]:
            interNeigh = np.mean(np.array(level2[interDiagonals[vertex], :]), axis = 1)
            neigh = np.concatenate((neigh, interNeigh))
        M = neigh.shape[0]
        trig_base = np.arange(M)
        xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / M)[:, None])
        eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / M)[:, None])
        xi_xi_trig = 4 * np.square(xi_trig) - 1
        eta_eta_trig = 4 * np.square(eta_trig) - 1
        xi_eta_trig = xi_trig * eta_trig
        xi[vertex, :] = (2 / M) * np.dot(xi_trig, neigh)
        eta[vertex, :] = (2 / M) * np.dot(eta_trig, neigh)
        xi_xi[vertex, :] = (2 / M) * np.dot(xi_xi_trig, neigh)
        eta_eta[vertex, :] = (2 / M) * np.dot(eta_eta_trig, neigh)
        xi_eta[vertex, :] = (8 / M) * np.dot(xi_eta_trig, neigh)
    elif val == 4:
        neigh = level2[directNeighbours[vertex]]
        neigh_hat = neigh.copy()
        if simpleDiagonals[vertex]:
            simpleNeigh = level2[simpleDiagonals[vertex]]
            neigh_hat = np.concatenate((neigh_hat, simpleNeigh))
        if interDiagonals[vertex]:
            interNeigh = np.mean(np.array(level2[interDiagonals[vertex], :]), axis = 1)
            neigh_hat = np.concatenate((neigh_hat, interNeigh))
        xi[vertex, :] = (2 / val) * (neigh[0] - neigh[2])
        eta[vertex, :] = (2 / val) * (neigh[1] - neigh[3])
        xi_xi[vertex, :] = neigh[0] + neigh[2]
        xi_xi[vertex, :] = neigh[1] + neigh[3]
        M = 8
        trig_base = np.arange(M)
        xi_eta_trig = np.cos(((trig_base + 0.5) * 2 * np.pi) / M) * np.sin(((trig_base + 0.5) * 2 * np.pi) / M)
        xi_eta[vertex, :] = (2 / M) * np.dot(xi_eta_trig, neigh_hat)

############################################################
# build all tensors and updated points

g_11 = xi[: , 0] ** 2 + xi[: , 1] ** 2 + xi[: , 2] ** 2
g_22 = eta[: , 0] ** 2 + eta[: , 1] ** 2 + eta[: , 2] ** 2
g_33 = zeta[: , 0] ** 2 + zeta[: , 1] ** 2 + zeta[: , 2] ** 2
g_12 = xi[: , 0] * eta[: , 0] + xi[: , 1] * eta[: , 1] + xi[: , 2] * eta[: , 2]
g = xi[: , 0] * (eta[: , 1] * zeta[: , 2] - eta[: , 2] * zeta[: , 1]) - xi[: , 1] * (eta[: , 0] * zeta[: , 2] - eta[: , 2] * zeta[: , 0]) + xi[: , 2] * (eta[: , 0] * zeta[: , 1] - eta[: , 1] * zeta[: , 0])
g_square = g ** 2
c1 = (g_22 * g_33) / g_square
c2 = (g_11 * g_33) / g_square
c3 = -2 * ((g_12 * g_33) / g_square)
c4 = (g_11 * g_22 - g_12 ** 2) / g_square
level2_it1 = (c1[: , None] * xi_xi + c2[: , None] * eta_eta + c3[: , None] * xi_eta + c4[: , None] * zeta_zeta) / (2 * (c1[: , None] + c2[: , None]))
# points_it1 = np.concatenate((level1, level2_it1))
# points_it1 = points_it1.astype('float64')
# points_it0 = np.concatenate((level1, level2))

# print(g_square)
# print(g_11[0])
# print(g_22[0])
# print(g_33[0])
# print(g_12[0])
# print(g[0])
# print(g_square[1])
# print(c1[1])
# print(c2[1])
# print(c3[1])
# print(c4[1])
# print(xi_xi[1])
# print(eta_eta[1])
# print(xi_eta[1])
# print(zeta_zeta[1])

v = 3
# gDisp = np.array([[xi[0, 0], xi[0, 1], xi[0, 2]], [eta[0, 0], eta[0, 1], eta[0, 2]], [zeta[0, 0], zeta[0, 1], zeta[0, 2]]])
gDisp = np.array([[xi[v, 0], xi[v, 1], xi[v, 2]], [eta[v, 0], eta[v, 1], eta[v, 2]], [zeta[v, 0], zeta[v, 1], zeta[v, 2]]])
print(g[v])
print("pause")
print(gDisp)
print(np.linalg.det(gDisp))

##############################################################

layer1_points = np.concatenate((surface_points, level2_it1))
# layer1_points = layer1_points.astype('float64')
voxels1_it1 = [["wedge", np.concatenate((surface_triangles, level2_triangles),axis=1)], ["hexahedron", np.concatenate((surface_quads, level2_quads),axis=1)]]
voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)], ["hexahedron", np.concatenate((level2_quads, level3_quads),axis=1)]]
# voxels1and2 = voxels1
# voxels1and2.extend(voxels2)
level2SurfaceV = [["triangle", surface_triangles], ["quad", surface_quads]]
level2Surface = meshio.Mesh(points = level2_it1, cells = level2SurfaceV)
V5_mesh_layer1 = meshio.Mesh(points = layer1_points, cells = voxels1_it1)
meshio.write("./output/V5_mesh_layer1.vtk", V5_mesh_layer1, file_format="vtk", binary=False)
meshio.write("./output/V5_mesh_layer1.msh", V5_mesh_layer1, file_format="gmsh22", binary=False)
meshio.write("./output/V5_mesh_level2Surface.vtk", level2Surface, file_format="vtk", binary=False)

#############################################################
#iteration 2
# level2 = surface_points + d1 * vertexNormal_norm
level2_triNormal2 = np.ndarray((nTriFaces, 3), dtype = np.float64)
level2_quadNormal2 = np.ndarray((nQuadFaces, 3), dtype = np.float64)
level2_vertexNormal2 = np.zeros([nPoints, 3], dtype = np.float64)
for i, triFace in enumerate(surface_triangles):
    coor = level2_it1[triFace, :]
    level2_triNormal2[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
for i, quadFace in enumerate(surface_quads):
    coor = level2_it1[quadFace, :]
    level2_quadNormal2[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
for vertex, (triNeighbours, quadNeighbours) in enumerate(zip(triFaceIndices, quadFaceIndices)):
    level2_vertexNormal2[vertex] += level2_triNormal[triNeighbours].sum(axis = 0) + level2_quadNormal[quadNeighbours].sum(axis = 0)
level2_vertexNormal_norm2 = level2_vertexNormal / np.linalg.norm(vertexNormal, axis = 1)[: , None]
level3_it1 = level2_it1 + d2 * level2_vertexNormal_norm2

zeta = (vertexNormal * d1 + level2_vertexNormal2 * d2) / (d1 + d2)
zeta_zeta = (1 * (level2_vertexNormal2 - vertexNormal)) / (d1 + d2)

xi = np.ndarray((nPoints, 3), dtype = np.float64)
eta = np.ndarray((nPoints, 3), dtype = np.float64)
xi_xi = np.ndarray((nPoints, 3), dtype = np.float64)
eta_eta = np.ndarray((nPoints, 3), dtype = np.float64)
xi_eta = np.ndarray((nPoints, 3), dtype = np.float64)


for vertex, val in enumerate(directValence):
    if val > 4:
        neigh = level2_it1[directNeighbours[vertex]]
        trig_base = np.arange(val)
        xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / val)[:, None])
        eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / val)[:, None])
        xi_xi_trig = 4 * np.square(xi_trig) - 1
        eta_eta_trig = 4 * np.square(eta_trig) - 1
        xi_eta_trig = xi_trig * eta_trig
        xi[vertex, :] = (2 / val) * np.dot(xi_trig, neigh)
        eta[vertex, :] = (2 / val) * np.dot(eta_trig, neigh)
        xi_xi[vertex, :] = (2 / val) * np.dot(xi_xi_trig, neigh)
        eta_eta[vertex, :] = (2 / val) * np.dot(eta_eta_trig, neigh)
        xi_eta[vertex, :] = (8 / val) * np.dot(xi_eta_trig, neigh)
    elif val == 3:
        neigh = level2_it1[directNeighbours[vertex]]
        if interDiagonals[vertex]:
            interNeigh = np.mean(np.array(level2_it1[interDiagonals[vertex], :]), axis = 1)
            neigh = np.concatenate((neigh, interNeigh))
        M = neigh.shape[0]
        trig_base = np.arange(M)
        xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / M)[:, None])
        eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / M)[:, None])
        xi_xi_trig = 4 * np.square(xi_trig) - 1
        eta_eta_trig = 4 * np.square(eta_trig) - 1
        xi_eta_trig = xi_trig * eta_trig
        xi[vertex, :] = (2 / M) * np.dot(xi_trig, neigh)
        eta[vertex, :] = (2 / M) * np.dot(eta_trig, neigh)
        xi_xi[vertex, :] = (2 / M) * np.dot(xi_xi_trig, neigh)
        eta_eta[vertex, :] = (2 / M) * np.dot(eta_eta_trig, neigh)
        xi_eta[vertex, :] = (8 / M) * np.dot(xi_eta_trig, neigh)
    elif val == 4:
        neigh = level2_it1[directNeighbours[vertex]]
        neigh_hat = neigh.copy()
        if simpleDiagonals[vertex]:
            simpleNeigh = level2_it1[simpleDiagonals[vertex]]
            neigh_hat = np.concatenate((neigh_hat, simpleNeigh))
        if interDiagonals[vertex]:
            interNeigh = np.mean(np.array(level2_it1[interDiagonals[vertex], :]), axis = 1)
            neigh_hat = np.concatenate((neigh_hat, interNeigh))
        eta[vertex, :] = (2 / val) * (neigh[0] - neigh[2])
        eta[vertex, :] = (2 / val) * (neigh[1] - neigh[3])
        xi_xi[vertex, :] = neigh[0] + neigh[2]
        xi_xi[vertex, :] = neigh[1] + neigh[3]
        M = 8
        trig_base = np.arange(M)
        xi_eta_trig = np.cos(((trig_base + 0.5) * 2 * np.pi) / M) * np.sin(((trig_base + 0.5) * 2 * np.pi) / M)
        xi_eta[vertex, :] = (2 / M) * np.dot(xi_eta_trig, neigh_hat)

g_11 = xi[: , 0] ** 2 + xi[: , 1] ** 2 + xi[: , 2] ** 2
g_22 = eta[: , 0] ** 2 + eta[: , 1] ** 2 + eta[: , 2] ** 2
g_33 = zeta[: , 0] ** 2 + zeta[: , 1] ** 2 + zeta[: , 2] ** 2
g_12 = xi[: , 0] * eta[: , 0] + xi[: , 1] * eta[: , 1] + xi[: , 2] * eta[: , 2]
g = xi[: , 0] * (eta[: , 1] * zeta[: , 2] - eta[: , 2] * zeta[: , 1]) - xi[: , 1] * (eta[: , 0] * zeta[: , 2] - eta[: , 2] * zeta[: , 0]) + xi[: , 2] * (eta[: , 0] * zeta[: , 1] - eta[: , 1] * zeta[: , 0])
g_square = g ** 2
c1 = (g_22 * g_33) / g_square
c2 = (g_11 * g_33) / g_square
c3 = -2 * ((g_12 * g_33) / g_square)
c4 = (g_11 * g_22 - g_12 ** 2) / g_square
level2_it2 = (c1[: , None] * xi_xi + c2[: , None] * eta_eta + c3[: , None] * xi_eta + c4[: , None] * zeta_zeta) / (2 * (c1[: , None] + c2[: , None]))

layer1_points2 = np.concatenate((surface_points, level2_it2))
# layer1_points = layer1_points.astype('float64')
voxels1_it2 = [["wedge", np.concatenate((surface_triangles, level2_triangles),axis=1)], ["hexahedron", np.concatenate((surface_quads, level2_quads),axis=1)]]
# voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)], ["hexahedron", np.concatenate((level2_quads, level3_quads),axis=1)]]
# voxels1and2 = voxels1
# voxels1and2.extend(voxels2)
# level2SurfaceV = [["triangle", surface_triangles], ["quad", surface_quads]]
# level2Surface = meshio.Mesh(points = level2_it1, cells = level2SurfaceV)
V5_mesh_layer1_it2 = meshio.Mesh(points = layer1_points2, cells = voxels1_it2)
meshio.write("./output/V5_mesh_layer1_it2.vtk", V5_mesh_layer1_it2, file_format="vtk", binary=False)
# meshio.write("./output/V5_mesh_layer1.msh", V5_mesh_layer1, file_format="gmsh22", binary=False)
# meshio.write("./output/V5_mesh_level2Surface.vtk", level2Surface, file_format="vtk", binary=False)
###############################################################
# control functions
# CFxi = np.ndarray((nPoints, 3), dtype = np.float64)
# CFeta = np.ndarray((nPoints, 3), dtype = np.float64)
# CFxi_xi = np.ndarray((nPoints, 3), dtype = np.float64)
# CFeta_eta = np.ndarray((nPoints, 3), dtype = np.float64)
# CFxi_eta = np.ndarray((nPoints, 3), dtype = np.float64)
#
# for vertex, val in enumerate(directValence):
#     if val > 4:
#         neigh = surface_points[directNeighbours[vertex]]
#         trig_base = np.arange(val)
#         xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / val)[:, None])
#         eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / val)[:, None])
#         xi_xi_trig = 4 * np.square(xi_trig) - 1
#         eta_eta_trig = 4 * np.square(eta_trig) - 1
#         xi_eta_trig = xi_trig * eta_trig
#         CFxi[vertex, :] = (2 / val) * np.dot(xi_trig, neigh)
#         CFeta[vertex, :] = (2 / val) * np.dot(eta_trig, neigh)
#         CFxi_xi[vertex, :] = (2 / val) * np.dot(xi_xi_trig, neigh)
#         CFeta_eta[vertex, :] = (2 / val) * np.dot(eta_eta_trig, neigh)
#         CFxi_eta[vertex, :] = (8 / val) * np.dot(xi_eta_trig, neigh)
#     elif val == 3:
#         neigh = surface_points[directNeighbours[vertex]]
#         if interDiagonals[vertex]:
#             interNeigh = np.mean(np.array(surface_points[interDiagonals[vertex], :]), axis = 1)
#             neigh = np.concatenate((neigh, interNeigh))
#         M = neigh.shape[0]
#         trig_base = np.arange(M)
#         xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / M)[:, None])
#         eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / M)[:, None])
#         xi_xi_trig = 4 * np.square(xi_trig) - 1
#         eta_eta_trig = 4 * np.square(eta_trig) - 1
#         xi_eta_trig = xi_trig * eta_trig
#         CFxi[vertex, :] = (2 / M) * np.dot(xi_trig, neigh)
#         CFeta[vertex, :] = (2 / M) * np.dot(eta_trig, neigh)
#         CFxi_xi[vertex, :] = (2 / M) * np.dot(xi_xi_trig, neigh)
#         CFeta_eta[vertex, :] = (2 / M) * np.dot(eta_eta_trig, neigh)
#         CFxi_eta[vertex, :] = (8 / M) * np.dot(xi_eta_trig, neigh)
#     elif val == 4:
#         neigh = surface_points[directNeighbours[vertex]]
#         neigh_hat = neigh.copy()
#         if simpleDiagonals[vertex]:
#             simpleNeigh = surface_points[simpleDiagonals[vertex]]
#             neigh_hat = np.concatenate((neigh_hat, simpleNeigh))
#         if interDiagonals[vertex]:
#             interNeigh = np.mean(np.array(surface_points[interDiagonals[vertex], :]), axis = 1)
#             neigh_hat = np.concatenate((neigh_hat, interNeigh))
#         CFeta[vertex, :] = (2 / val) * (neigh[0] - neigh[2])
#         CFeta[vertex, :] = (2 / val) * (neigh[1] - neigh[3])
#         CFxi_xi[vertex, :] = neigh[0] + neigh[2]
#         CFxi_xi[vertex, :] = neigh[1] + neigh[3]
#         M = 8
#         trig_base = np.arange(M)
#         xi_eta_trig = np.cos(((trig_base + 0.5) * 2 * np.pi) / M) * np.sin(((trig_base + 0.5) * 2 * np.pi) / M)
#         CFxi_eta[vertex, :] = (2 / M) * np.dot(xi_eta_trig, neigh_hat)

# print(xi[0, :])
# print(eta[0, :])
# print(xi_xi[0, :])
# print(eta_eta[0, :])
# print(xi_eta[0, :])
# theta = (-0.5) * ((xi_xi * xi) / g_33[: , None])
# print(xi.shape)
# print(xi_xi.shape)
# print(g_33.shape)
# print(theta)
# print(theta.shape)
# ##########################################################

# ####################################################################




####################################################
