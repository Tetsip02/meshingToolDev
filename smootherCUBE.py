import meshio
import numpy as np
# import math
from helpers import *
# import json
import pickle
import math

np.set_printoptions(precision=16)

surface_triangles_ini, surface_quads_ini, surface_points = read_surface("./testGeometries/Mesh_3.stl")

##########################################################################################

with open("./dataOutput/Mesh_3_triFaceIndices.txt", "rb") as fp:   # Unpickling
    triFaceIndices = pickle.load(fp)
with open("./dataOutput/Mesh_3_directValence.txt", "rb") as fp:   # Unpickling
    directValence = pickle.load(fp)
with open("./dataOutput/Mesh_3_directNeighbours_sorted.txt", "rb") as fp:   # Unpickling
    directNeighbours = pickle.load(fp)
with open("./dataOutput/Mesh_3_simpleDiagonals.txt", "rb") as fp:   # Unpickling
    simpleDiagonals = pickle.load(fp)

######################################################################
# print(len(triFaceIndices))
# print(directValence[0:100])
##########################################
# print(directValence)


#change orientation
surface_triangles = surface_triangles_ini.copy()
# surface_triangles[:,[1, 2]] = surface_triangles[:,[2, 1]]
# print(surface_triangles[0:10])

######################################

nPoints = len(surface_points)
nTriFaces = len(surface_triangles)
nFaces = nTriFaces

######################################################

level2_triangles = surface_triangles + nPoints
level3_triangles = level2_triangles + nPoints

########################################################################


#########################################################

######################################
def get_layer_normals(points, triangles, tri_neighbours):
    # build face normals
    triNormal = np.ndarray((nTriFaces, 3), dtype = np.float64)
    for i, triFace in enumerate(triangles):
        coor = points[triFace, :]
        triNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
    # print(triNormal)
    # print(triNormal.shape)
    vertexNormal = np.zeros([points.shape[0], 3], dtype = np.float64)
    for vertex, triNeighbours in enumerate(tri_neighbours):
        vertexNormal[vertex] += triNormal[triNeighbours].sum(axis = 0)
    # normalize
    # print(vertexNormal)
    # print("vnormal", vertexNormal.shape)
    vertexNormal_norm = vertexNormal / np.linalg.norm(vertexNormal, axis = 1)[: , None]

    return vertexNormal, vertexNormal_norm

test_normals, test_normals_norm = get_layer_normals(surface_points, surface_triangles, triFaceIndices)
# print(test_normals[[0, 400, 800], :])
# print(test_normals_norm[5])
# print("pause")

#############################################################

def get_derivatives(level2, level1_normals, level2_normals, d1, d2, directValence, directNeighbours, simpleDiagonals):
    zeta = (level1_normals * d1 + level2_normals * d2) / (d1 + d2)
    zeta_zeta = (2 * (level2_normals - level1_normals)) / (d1 + d2) ## double check if the 2 multiplier if correct
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
            xi[vertex, :] = (2 / val) * (neigh[0] - neigh[2])
            eta[vertex, :] = (2 / val) * (neigh[1] - neigh[3])
            # trig_base = np.arange(val)
            # xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / val)[:, None]) #
            # eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / val)[:, None]) #
            # xi[vertex, :] = (2 / val) * np.dot(xi_trig, neigh) #
            # eta[vertex, :] = (2 / val) * np.dot(eta_trig, neigh) #
            xi_xi[vertex, :] = neigh[0] + neigh[2]
            eta_eta[vertex, :] = neigh[1] + neigh[3]
            # xi_xi_trig = np.square(xi_trig)
            # eta_eta_trig = np.square(eta_trig)
            M = 8
            trig_base = np.arange(M)
            xi_eta_trig = np.cos(((trig_base + 0.5) * 2 * np.pi) / M) * np.sin(((trig_base + 0.5) * 2 * np.pi) / M)
            xi_eta[vertex, :] = (2 / M) * np.dot(xi_eta_trig, neigh_hat)
            # xi_eta[vertex, :] = (2 / val) * np.dot(xi_eta_trig, neigh_hat)

    return xi, eta, xi_xi, eta_eta, xi_eta, zeta, zeta_zeta

#######################################################################

def get_tensors(xi, eta, xi_xi, eta_eta, xi_eta, zeta, zeta_zeta):
    g_11 = xi[: , 0] ** 2 + xi[: , 1] ** 2 + xi[: , 2] ** 2
    g_22 = eta[: , 0] ** 2 + eta[: , 1] ** 2 + eta[: , 2] ** 2
    g_33 = zeta[: , 0] ** 2 + zeta[: , 1] ** 2 + zeta[: , 2] ** 2
    g_12 = xi[: , 0] * eta[: , 0] + xi[: , 1] * eta[: , 1] + xi[: , 2] * eta[: , 2]
    g = xi[: , 0] * (eta[: , 1] * zeta[: , 2] - eta[: , 2] * zeta[: , 1]) - xi[: , 1] * (eta[: , 0] * zeta[: , 2] - eta[: , 2] * zeta[: , 0]) + xi[: , 2] * (eta[: , 0] * zeta[: , 1] - eta[: , 1] * zeta[: , 0])
    # g = 100 * (xi[: , 0] * (eta[: , 1] * zeta[: , 2] - eta[: , 2] * zeta[: , 1]) - xi[: , 1] * (eta[: , 0] * zeta[: , 2] - eta[: , 2] * zeta[: , 0]) + xi[: , 2] * (eta[: , 0] * zeta[: , 1] - eta[: , 1] * zeta[: , 0]))
    # g = g.astype(np.float64)
    g_square = g ** 2
    # g_square = g
    c1 = (g_22 * g_33) / g_square
    c2 = (g_11 * g_33) / g_square
    c3 = (-2) * ((g_12 * g_33) / g_square)
    c4 = (g_11 * g_22 - g_12 ** 2) / g_square

    return g_11, g_22, g_33, g_12, g_square, c1, c2, c3, c4

###################################################################

d1 = 0.05
d2 = 0.07
n_it = 2
level1_normals, level1_normals_norm = get_layer_normals(surface_points, surface_triangles, triFaceIndices)
level2 = surface_points + d1 * level1_normals_norm
level2_normals, level2_normals_norm = get_layer_normals(level2, surface_triangles, triFaceIndices)
level3 = level2 + d2 * level2_normals_norm

# print(level2[[0, 400, 800], :])
# print(level3[[0, 400, 800], :])

# print(level2[directNeighbours[2447]])
# print(level2[simpleDiagonals[2447]])

refMesh_points = np.concatenate((surface_points, level2, level3))
voxels1 = [["wedge", np.concatenate((surface_triangles, level2_triangles),axis=1)]]
voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)]]
voxels1and2 = voxels1.copy()
voxels1and2.extend(voxels2)
V6_referenceMesh = meshio.Mesh(points = refMesh_points, cells = voxels1and2)
meshio.write("./output/Mesh_3_ref.vtk", V6_referenceMesh, file_format="vtk", binary=False)
# meshio.write("./output/ref_mesh_test2.msh", V5_mesh, file_format="gmsh22", binary=False)
# v = 500

##############################################

# control functions
# need to sort out zeta_zeta
xiCF, etaCF, xi_xiCF, eta_etaCF, xi_etaCF, zetaCF, zeta_zetaCF = get_derivatives(surface_points, level1_normals_norm, level2_normals_norm, d1, d2, directValence, directNeighbours, simpleDiagonals)
zetaCF = level1_normals
zeta_zetaCF = 0 * zeta_zetaCF
g_11CF, g_22CF, g_33CF, g_12CF, g_squareCF, c1CF, c2CF, c3CF, c4CF = get_tensors(xiCF, etaCF, xi_xiCF, eta_etaCF, xi_etaCF, zetaCF, zeta_zetaCF)
# g11_xi = 2 * xi_xiCF * xiCF
g11_xi = 2 * np.einsum('ij,ij->i', xi_xiCF, xiCF)
g22_xi = 2 * np.einsum('ij,ij->i', xi_etaCF, etaCF)
g11_eta = 2 * np.einsum('ij,ij->i', xi_etaCF, xiCF)
g12_eta = np.einsum('ij,ij->i', eta_etaCF, xiCF) + np.einsum('ij,ij->i', xi_etaCF, etaCF)
g22_eta = 2 * np.einsum('ij,ij->i', eta_etaCF, etaCF)
g12_xi = np.einsum('ij,ij->i', xi_xiCF, etaCF) + np.einsum('ij,ij->i', xi_etaCF, xiCF)

# print(g11_xi.shape)
# print(xi_xiCF[0])
print("xi", xiCF[677])
print("eta", etaCF[677])
print("xi_xi", xi_xiCF[677])
print("eta_eta", eta_etaCF[677])
print("xi_eta", xi_etaCF[677])
print(surface_points[677])
# print(interDiagonals[677])
# print(g11_xi[0])
RHSa = (g_12CF / g_22CF) * ((g11_eta / g_11CF) - (g12_eta / g_12CF)) - 0.5 * ((g11_xi / g_11CF) - (g22_xi / g_22CF))
RHSb = (g_12CF / g_11CF) * ((g22_xi / g_22CF) - (g12_xi / g_12CF)) - 0.5 * ((g22_eta / g_22CF) - (g11_eta / g_11CF))
phi = ((g_11CF * g_22CF) / (g_11CF * g_22CF - g_12CF ** 2)) * (RHSa - (g_12CF / g_22CF) * RHSb)
psi = ((g_11CF * g_22CF) / (g_11CF * g_22CF - g_12CF ** 2)) * (RHSb - (g_12CF / g_11CF) * RHSa)

# inf1 = math.inf
# for i in range(nPoints):
#     if np.abs(phi[i]) == inf1:
#         phi[i] = 0
#     if np.abs(psi[i]) == inf1:
#         psi[i] = 0

############################################



for it in range(n_it):
    xi, eta, xi_xi, eta_eta, xi_eta, zeta, zeta_zeta = get_derivatives(level2, level1_normals_norm, level2_normals_norm, d1, d2, directValence, directNeighbours, simpleDiagonals) #not sure if normalized or absolute normals work better

    # xi -= xiCF
    # eta -= etaCF
    # xi_xi -= xi_xiCF
    # eta_eta -= eta_etaCF
    # zeta -= zetaCF
    # zeta_zeta -= zeta_zetaCF
    # xi *= 0.1
    # eta *= 0.1
    # xi_xi *= 0.1
    # eta_eta *= 0.1
    # zeta *= 0.1
    # zeta_zeta *= 0.1

    # gDisp = np.array([[xi[v, 0], xi[v, 1], xi[v, 2]], [eta[v, 0], eta[v, 1], eta[v, 2]], [zeta[v, 0], zeta[v, 1], zeta[v, 2]]])
    # print(g[v])
    # print("pause")
    # print(gDisp)
    # print(np.linalg.det(gDisp))
    g_11, g_22, g_33, g_12, g_square, c1, c2, c3, c4 = get_tensors(xi, eta, xi_xi, eta_eta, xi_eta, zeta, zeta_zeta)
    level2 = (c1[: , None] * xi_xi + c2[: , None] * eta_eta + c3[: , None] * xi_eta + c4[: , None] * zeta_zeta) / (2 * (c1[: , None] + c2[: , None]))
    # level2 = (c1[: , None] * (xi_xi + phi[: , None] * xi) + c2[: , None] * (eta_eta + psi[: , None] * eta) + c3[: , None] * xi_eta + c4[: , None] * zeta_zeta) / (2 * (c1[: , None] + c2[: , None]))
    level2_normals, level2_normals_norm = get_layer_normals(level2, surface_triangles, triFaceIndices)
    print("it", it)
    # print(xi.dtype)
    # print(xi[2447, :])
    # print(directNeighbours[2447])
    # print(simpleDiagonals[2447])

# print(level2[500:750])
level3 = level2 + d2 * level2_normals_norm



smoothMesh_points = np.concatenate((surface_points, level2))
level2Surface = [["triangle", surface_triangles]]
# voxels1 = [["wedge", np.concatenate((surface_triangles, level2_triangles),axis=1)], ["hexahedron", np.concatenate((surface_quads, level2_quads),axis=1)]]
# voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)], ["hexahedron", np.concatenate((level2_quads, level3_quads),axis=1)]]
# voxels1and2 = voxels1
# voxels1and2.extend(voxels2)
V6_smoothMesh = meshio.Mesh(points = smoothMesh_points, cells = voxels1)
V6_surface = meshio.Mesh(points = level2, cells = level2Surface)
meshio.write("./output/Mesh_3_smooth.vtk", V6_smoothMesh, file_format="vtk", binary=False)
meshio.write("./output/Mesh_3_surface.vtk", V6_surface, file_format="vtk", binary=False)
# meshio.write("./output/ref_mesh_test2.msh", V5_mesh, file_format="gmsh22", binary=False)

#############################################################
#
# # control functions
# # need to sort out zeta_zeta
# xiCF, etaCF, xi_xiCF, eta_etaCF, xi_etaCF, zetaCF, zeta_zetaCF = get_derivatives(surface_points, level1_normals_norm, level2_normals_norm, d1, d2, directValence, directNeighbours, simpleDiagonals, interDiagonals)
# # zetaCF = level1_normals
# g_11CF, g_22CF, g_33CF, g_12CF, g_squareCF, c1CF, c2CF, c3CF, c4CF = get_tensors(xiCF, etaCF, xi_xiCF, eta_etaCF, xi_etaCF, zetaCF, zeta_zetaCF)
# # g11_xi = 2 * xi_xiCF * xiCF
# g11_xi = 2 * np.einsum('ij,ij->i', xi_xiCF, xiCF)
# g22_xi = 2 * np.einsum('ij,ij->i', xi_etaCF, etaCF)
# g11_eta = 2 * np.einsum('ij,ij->i', xi_etaCF, xiCF)
# g12_eta = np.einsum('ij,ij->i', eta_etaCF, xiCF) + np.einsum('ij,ij->i', xi_etaCF, etaCF)
# g22_eta = 2 * np.einsum('ij,ij->i', eta_etaCF, etaCF)
# g12_xi = np.einsum('ij,ij->i', xi_xiCF, etaCF) + np.einsum('ij,ij->i', xi_etaCF, xiCF)
#
# # print(g11_xi.shape)
# # print(xi_xiCF[0])
# # print(xiCF[0])
# # print(g11_xi[0])
# RHSa = (g_12CF / g_22CF) * ((g11_eta / g_11CF) - (g12_eta / g_12CF)) - 0.5 * ((g11_xi / g_11CF) - (g22_xi / g_22CF))
# RHSb = (g_12CF / g_11CF) * ((g22_xi / g_22CF) - (g12_xi / g_12CF)) - 0.5 * ((g22_eta / g_22CF) - (g11_eta / g_11CF))
# phi = ((g_11CF * g_22CF) / (g_11CF * g_22CF - g_12CF ** 2)) * (RHSa - (g_12CF / g_22CF) * RHSb)
# psi = ((g_11CF * g_22CF) / (g_11CF * g_22CF - g_12CF ** 2)) * (RHSb - (g_12CF / g_11CF) * RHSa)
# # RHSa = (g_12CF / g_22CF)
# # print(phi.shape)
# # print(g_12CF[0])
# # print(g_22CF[0])
# # print(g_11CF[1900:])
# # print(g_22CF[1900:])
# print(g_12CF[677])
# inf1 = math.inf
# for i, j in enumerate(phi):
#     if np.abs(j) == inf1:
#         print(i)
