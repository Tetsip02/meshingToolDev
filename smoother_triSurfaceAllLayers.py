import meshio
import numpy as np
# import math
from helpers import *
# import json
import pickle
import math

np.set_printoptions(precision=16)

surface_triangles_ini, surface_quads_ini, surface_points = read_surface("./testGeometries/P023rotor0.stl")

##########################################################################################

with open("./dataOutput/P023rotor0_triFaceIndices.txt", "rb") as fp:   # Unpickling
    triFaceIndices = pickle.load(fp)
with open("./dataOutput/P023rotor0_directValence.txt", "rb") as fp:   # Unpickling
    directValence = pickle.load(fp)
with open("./dataOutput/P023rotor0_directNeighbours.txt", "rb") as fp:   # Unpickling
    directNeighbours = pickle.load(fp)
with open("./dataOutput/P023rotor0_neighbours_and_diagonals.txt", "rb") as fp:   # Unpickling
    neighbours_and_diagonals = pickle.load(fp)

######################################################################

#change orientation
surface_triangles = surface_triangles_ini.copy()
surface_triangles[:,[1, 2]] = surface_triangles[:,[2, 1]]
surface_quads = surface_quads_ini.copy()

######################################

nPoints = len(surface_points)
nFaces = len(surface_triangles)

######################################################

level2_triangles = surface_triangles + nPoints
level3_triangles = level2_triangles + nPoints

########################################################################


#########################################################

######################################
def get_layer_normals(points, triangles, tri_neighbours):
    # build face normals
    triNormal = np.ndarray((nFaces, 3), dtype = np.float64)
    for i, triFace in enumerate(triangles):
        coor = points[triFace, :]
        triNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
    # build vertex normals
    vertexNormal = np.zeros([points.shape[0], 3], dtype = np.float64)
    for vertex, triNeighbours in enumerate(tri_neighbours):
        vertexNormal[vertex] += triNormal[triNeighbours].sum(axis = 0)
    # normalize
    vertexNormal_norm = vertexNormal / np.linalg.norm(vertexNormal, axis = 1)[: , None]

    return vertexNormal, vertexNormal_norm

#############################################################

def get_derivatives(level2, level1_normals, level2_normals, d1, d2, directValence, directNeighbours, neighbours_and_diagonals):
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
            M = 6
            neigh = np.ndarray((M, 3), dtype = np.float64)
            for p, m in enumerate(directNeighbours[vertex]):
                try:
                    if len(m) == 2:
                        neigh[p] = np.mean((level2[m]), axis = 0)
                except TypeError:
                    neigh[p] = level2[m]
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
            xi[vertex, :] = (2 / val) * (neigh[0] - neigh[2])
            eta[vertex, :] = (2 / val) * (neigh[1] - neigh[3])
            xi_xi[vertex, :] = neigh[0] + neigh[2]
            eta_eta[vertex, :] = neigh[1] + neigh[3]
            # get mixed derivative
            M = 8
            neigh_hat = np.ndarray((M, 3), dtype = np.float64)
            trig_base = np.arange(M)
            xi_eta_trig = np.cos(((trig_base + 0.5) * 2 * np.pi) / M) * np.sin(((trig_base + 0.5) * 2 * np.pi) / M)
            for p, m in enumerate(neighbours_and_diagonals[vertex]):
                try:
                    if len(m) == 2:
                        neigh_hat[p] = np.mean((level2[m]), axis = 0)
                except TypeError:
                    neigh_hat[p] = level2[m]
            xi_eta[vertex, :] = (2 / M) * np.dot(xi_eta_trig, neigh_hat)

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
    c3 = -2 * ((g_12 * g_33) / g_square)
    c4 = (g_11 * g_22 - g_12 ** 2) / g_square

    return g_11, g_22, g_33, g_12, g_square, c1, c2, c3, c4

# ####################################################################
#
# # control functions
# # need to sort out zeta_zeta
# xiCF, etaCF, xi_xiCF, eta_etaCF, xi_etaCF, zetaCF, zeta_zetaCF = get_derivatives(surface_points, level1_normals_norm, level2_normals_norm, d1, d2, directValence, directNeighbours, neighbours_and_diagonals)
# # zetaCF = level1_normals
# g11CF, g22CF, g33CF, g12CF, gsquareCF, c1CF, c2CF, c3CF, c4CF = get_tensors(xiCF, etaCF, xi_xiCF, eta_etaCF, xi_etaCF, zetaCF, zeta_zetaCF)
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
# RHSa = (g12CF / g22CF) * ((g11eta / g11CF) - (g12eta / g12CF)) - 0.5 * ((g11xi / g11CF) - (g22xi / g22CF))
# RHSb = (g12CF / g11CF) * ((g22xi / g22CF) - (g12xi / g12CF)) - 0.5 * ((g22eta / g22CF) - (g11eta / g11CF))
# phi = ((g11CF * g22CF) / (g11CF * g22CF - g12CF ** 2)) * (RHSa - (g12CF / g22CF) * RHSb)
# psi = ((g11CF * g22CF) / (g11CF * g22CF - g12CF ** 2)) * (RHSb - (g12CF / g11CF) * RHSa)
# # RHSa = (g_12CF / g_22CF)
# # print(phi.shape)
# # print(g_12CF[0])
# # print(g_22CF[0])
# # print(g_11CF[1900:])
# # print(g_22CF[1900:])
# # print(g_12CF[677])
# # inf1 = math.inf
# # for i, j in enumerate(phi):
# #     if np.abs(j) == inf1:
# #         print(i)
#
# # ###################################################################
# #directional smoothing coefficients | evaluated at level2, different here just for testing
# vertex = 0
# # print(directNeighbours[vertex])
# neigh = surface_points[directNeighbours[vertex]]
# val = directValence[vertex]
# # print(neigh)
# r0 = surface_points[vertex]
# g33array = np.random.rand(nPoints)
# g33 = g33array[vertex]
# g33 = 0.006
# gm = np.sum((neigh - r0) ** 2, axis = 1)
# print(gm)
# print(g33)
# vm = ((np.maximum(gm, g33) / g33) ** 0.5) - 1
# print(vm)
# normals, normals_norm = get_layer_normals(surface_points, surface_triangles, triFaceIndices)
# refvec = normals[vertex]
# print(refvec)
# alpham = np.dot((neigh - r0), refvec)
# print(neigh - r0)
# print(alpham)
# refvec_norm = np.linalg.norm(refvec)
# print(refvec_norm)
# other_norm = np.linalg.norm(neigh - r0, axis = 1)
# print(other_norm)
# print(other_norm * refvec_norm)
# alpham = alpham / (other_norm * refvec_norm)
# print(alpham)
# alpham = np.arccos(alpham)
# print(alpham)
# Falpha = alpham.copy()
# for n, alpha in enumerate(alpham):
#     if alpha <= np.pi / 4:
#         Falpha[n] = 0.5
#     elif alpha > np.pi / 4 and alpha < np.pi / 2:
#         Falpha[n] = 0.5 * np.sin(2 * alpha)
#     elif alpha >= np.pi / 2:
#         Falpha[n] = 0
# print(Falpha)
# vm = vm * Falpha
# print(vm)
# # vm += 1
# # vm[3] = 4
# # print(vm)
# trig_base = np.arange(val)
# v_xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / val))
# v_eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / val))
# print(v_xi_trig)
# print(v_eta_trig)
# v_xi = np.dot(vm, v_xi_trig)
# print(v_xi)
# v_eta = np.dot(vm, v_eta_trig)
# print(v_eta)
#
# for vertex, r0 in enumerate(surface_points):
#     neigh = surface_points[directNeighbours[vertex]]
#     val = directValence[vertex]
#     # r0 = surface_points[vertex]
#     g33array = np.random.rand(nPoints)
#     g33 = g33array[vertex]
#     g33 = 0.006
#     gm = np.sum((neigh - r0) ** 2, axis = 1)
#     vm = ((np.maximum(gm, g33) / g33) ** 0.5) - 1
#     normals, normals_norm = get_layer_normals(surface_points, surface_triangles, triFaceIndices)
#     refvec = normals[vertex]
#     alpham = np.dot((neigh - r0), refvec)
#     refvec_norm = np.linalg.norm(refvec)
#     other_norm = np.linalg.norm(neigh - r0, axis = 1)
#     alpham = alpham / (other_norm * refvec_norm)
#     alpham = np.arccos(alpham)
#     Falpha = alpham.copy()
#     for n, alpha in enumerate(alpham):
#         if alpha <= np.pi / 4:
#             Falpha[n] = 0.5
#         elif alpha > np.pi / 4 and alpha < np.pi / 2:
#             Falpha[n] = 0.5 * np.sin(2 * alpha)
#         elif alpha >= np.pi / 2:
#             Falpha[n] = 0
#     vm = vm * Falpha
#     trig_base = np.arange(val)
#     v_xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / val))
#     v_eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / val))
#     v_xi = np.dot(vm, v_xi_trig)
#     v_eta = np.dot(vm, v_eta_trig)

######################################################################
# get smoothed mesh points
nLayers = 50
GR = 1.2
firstLayerHeight = 0.000005
nSmoothingIt = 1
layerHeight = np.arange(nLayers + 1)
layerHeight = firstLayerHeight * GR ** layerHeight
level1 = surface_points.copy()
points = level1.copy()

for layer in range(nLayers):
    print(layer)

    #reference mesh
    level1_normals, level1_normals_norm = get_layer_normals(level1, surface_triangles, triFaceIndices)
    level2 = level1 + layerHeight[layer] * level1_normals_norm
    level2_normals, level2_normals_norm = get_layer_normals(level2, surface_triangles, triFaceIndices)

    # control functions
    if layer == 0:
        xiCF, etaCF, xi_xiCF, eta_etaCF, xi_etaCF, zetaCF, zeta_zetaCF = get_derivatives(level1, level1_normals_norm, level2_normals_norm, layerHeight[layer], layerHeight[layer + 1], directValence, directNeighbours, neighbours_and_diagonals)
        xi_xiCF -= 2 * level1
        eta_etaCF -= 2 * level1
        zetaCF = level1_normals
        zeta_zetaCF *= 0
    else:
        xiCF, etaCF, xi_xiCF, eta_etaCF, xi_etaCF, zetaCF, zeta_zetaCF = get_derivatives(level1, theta_input, level1_normals_norm, layerHeight[layer - 1], layerHeight[layer], directValence, directNeighbours, neighbours_and_diagonals)
        xi_xiCF -= 2 * level1
        eta_etaCF -= 2 * level1
        # zetaCF = level1_normals_norm
        # zeta_zetaCF *= 0
    g11CF, g22CF, g33CF, g12CF, gsquareCF, c1CF, c2CF, c3CF, c4CF = get_tensors(xiCF, etaCF, xi_xiCF, eta_etaCF, xi_etaCF, zetaCF, zeta_zetaCF)
    g33_zeta = 2 * np.einsum('ij,ij->i', zeta_zetaCF, zetaCF)
    print(zeta_zetaCF[5])
    print(zetaCF[5])
    print(g33_zeta[5])
    print("g33", g33CF.shape)
    print(nPoints)
    g11_xi = 2 * np.einsum('ij,ij->i', xi_xiCF, xiCF)
    g22_xi = 2 * np.einsum('ij,ij->i', xi_etaCF, etaCF)
    g11_eta = 2 * np.einsum('ij,ij->i', xi_etaCF, xiCF)
    g12_eta = np.einsum('ij,ij->i', eta_etaCF, xiCF) + np.einsum('ij,ij->i', xi_etaCF, etaCF)
    g22_eta = 2 * np.einsum('ij,ij->i', eta_etaCF, etaCF)
    g12_xi = np.einsum('ij,ij->i', xi_xiCF, etaCF) + np.einsum('ij,ij->i', xi_etaCF, xiCF)
    print("test1")
    # for i, j in enumerate(g12CF):
    #     if np.abs(j) == 0:
    #         g12CF[i] = 0.00001
    RHSa = (g12CF / g22CF) * ((g11_eta / g11CF) - (g12_eta / g12CF)) - 0.5 * ((g11_xi / g11CF) - (g22_xi / g22CF))
    # print(level2[677])
    # RHSa = ((g11_eta / g11CF)) - (g12_eta / g12CF))
    # RHSa = ((g12_eta / g12CF))
    print("test2")
    RHSb = (g12CF / g11CF) * ((g22_xi / g22CF) - (g12_xi / g12CF)) - 0.5 * ((g22_eta / g22CF) - (g11_eta / g11CF))
    phi = ((g11CF * g22CF) / (g11CF * g22CF - g12CF ** 2)) * (RHSa - (g12CF / g22CF) * RHSb)
    psi = ((g11CF * g22CF) / (g11CF * g22CF - g12CF ** 2)) * (RHSb - (g12CF / g11CF) * RHSa)
    theta = (-0.5) * (g33_zeta / g33CF)
    # phi *= 0
    # psi *= 0
    # inf1 = math.inf
    # for i, j in enumerate(RHSa):
    #     if np.abs(j) == inf1:
    #         print(i)
    # for i, j in enumerate(g12CF):
    #     if np.abs(j) == 0:
    #         g12CF[i] = 0.00001
    # print(g12_eta[677])
    # print(g12CF[677])
    # print(g12CF[400:600].shape)
    # print(xiCF[677])
    # print(etaCF[677])


    for it in range(nSmoothingIt):
        xi, eta, xi_xi, eta_eta, xi_eta, zeta, zeta_zeta = get_derivatives(level2, level1_normals_norm, level2_normals_norm, layerHeight[layer], layerHeight[layer + 1], directValence, directNeighbours, neighbours_and_diagonals) #not sure if normalized or absolute normals work better
        # zeta_zeta *= 0.1
        g_11, g_22, g_33, g_12, g_square, c1, c2, c3, c4 = get_tensors(xi, eta, xi_xi, eta_eta, xi_eta, zeta, zeta_zeta)
        # c4 *= 5

        v_xi = np.ndarray((nPoints, 1), dtype = np.float64)
        v_eta = np.ndarray((nPoints, 1), dtype = np.float64)
        for vertex, r0 in enumerate(level2):
            neigh = level2[directNeighbours[vertex]]
            val = directValence[vertex]
            # r0 = surface_points[vertex]
            # g33array = np.random.rand(nPoints)
            g33 = g_33[vertex]
            # g33 = 0.006
            gm = np.sum((neigh - r0) ** 2, axis = 1)
            vm = ((np.maximum(gm, g33) / g33) ** 0.5) - 1
            # normals, normals_norm = get_layer_normals(surface_points, surface_triangles, triFaceIndices)
            refvec = level2_normals[vertex]
            alpham = np.dot((neigh - r0), refvec)
            refvec_norm = np.linalg.norm(refvec)
            other_norm = np.linalg.norm(neigh - r0, axis = 1)
            alpham = alpham / (other_norm * refvec_norm)
            alpham = np.arccos(alpham)
            Falpha = alpham.copy()
            for n, alpha in enumerate(alpham):
                if alpha <= np.pi / 4:
                    Falpha[n] = 0.5
                elif alpha > np.pi / 4 and alpha < np.pi / 2:
                    Falpha[n] = 0.5 * np.sin(2 * alpha)
                elif alpha >= np.pi / 2:
                    Falpha[n] = 0
            vm = vm * Falpha
            trig_base = np.arange(val)
            v_xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / val))
            v_eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / val))
            v_xi[vertex] = np.dot(vm, v_xi_trig)
            v_eta[vertex] = np.dot(vm, v_eta_trig)

        # print(v_xi)
        # print(v_eta)

        # level2 = (c1[: , None] * xi_xi + c2[: , None] * eta_eta + c3[: , None] * xi_eta + c4[: , None] * zeta_zeta) / (2 * (c1[: , None] + c2[: , None]))
        # level2 = (c1[: , None] * (xi_xi + phi[: , None] * xi) + c2[: , None] * (eta_eta + psi[: , None] * eta) + c3[: , None] * xi_eta + c4[: , None] * (zeta_zeta + theta[: , None] * zeta)) / (2 * (c1[: , None] + c2[: , None]))
        # level2 = (c1[: , None] * ((1 + v_xi) * xi_xi + phi[: , None] * xi) + c2[: , None] * ((1 + v_eta) * eta_eta + psi[: , None] * eta) + c3[: , None] * xi_eta + c4[: , None] * (zeta_zeta + theta[: , None] * zeta)) / (2 * ((1 + v_xi) * c1[: , None] + (1 + v_eta) * c2[: , None]))
        level2_normals, level2_normals_norm = get_layer_normals(level2, surface_triangles, triFaceIndices)

    # if layer == 0:
    #     level2 = level1 + layerHeight[layer] * level1_normals_norm
    points = np.append(points, level2, axis = 0)
    level1 = level2
    theta_input = level1_normals_norm.copy() #not sure again if normalized or not

################################################################
# build mesh
voxels = []
for layer in range(nLayers):
    layerVoxel = [["wedge", np.concatenate((surface_triangles + (layer * nPoints), surface_triangles + ((layer + 1) * nPoints)),axis=1)]]
    voxels.extend(layerVoxel)
# print(points[:, 0, :].size)
# nMeshPoints = nPoints + nLayers * nPoints
# print(nMeshPoints)
# points2 = np.reshape(points, (nMeshPoints, 3), order='C')
mesh = meshio.Mesh(points = points, cells = voxels)
meshio.write("./output/P023rotor0_completeMesh.vtk", mesh, file_format="vtk", binary=False)
meshio.write("./output/P023rotor0_completeMesh.msh", mesh, file_format="gmsh22", binary=False)
# meshio.write("./output/ref_mesh_test2.msh", V5_mesh, file_format="gmsh22", binary=False)
###############################################################

# d1 = 0.05
# d2 = 0.06
# n_it = 3
# level1_normals, level1_normals_norm = get_layer_normals(surface_points, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices)
# level2 = surface_points + d1 * level1_normals_norm
# level2_normals, level2_normals_norm = get_layer_normals(level2, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices)
# level3 = level2 + d2 * level2_normals_norm
#
# refMesh_points = np.concatenate((surface_points, level2, level3))
# voxels1 = [["wedge", np.concatenate((surface_triangles, level2_triangles),axis=1)], ["hexahedron", np.concatenate((surface_quads, level2_quads),axis=1)]]
# voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)], ["hexahedron", np.concatenate((level2_quads, level3_quads),axis=1)]]
# voxels1and2 = voxels1.copy()
# voxels1and2.extend(voxels2)
# V6_referenceMesh = meshio.Mesh(points = refMesh_points, cells = voxels1and2)
# meshio.write("./output/mixedT_referenceMesh.vtk", V6_referenceMesh, file_format="vtk", binary=False)
# # meshio.write("./output/ref_mesh_test2.msh", V5_mesh, file_format="gmsh22", binary=False)
# # v = 500

##############################################

# for it in range(n_it):
#     xi, eta, xi_xi, eta_eta, xi_eta, zeta, zeta_zeta = get_derivatives(level2, level1_normals_norm, level2_normals_norm, d1, d2, directValence, directNeighbours, neighbours_and_diagonals) #not sure if normalized or absolute normals work better
#     # gDisp = np.array([[xi[v, 0], xi[v, 1], xi[v, 2]], [eta[v, 0], eta[v, 1], eta[v, 2]], [zeta[v, 0], zeta[v, 1], zeta[v, 2]]])
#     # print(g[v])
#     # print("pause")
#     # print(gDisp)
#     # print(np.linalg.det(gDisp))
#     g_11, g_22, g_33, g_12, g_square, c1, c2, c3, c4 = get_tensors(xi, eta, xi_xi, eta_eta, xi_eta, zeta, zeta_zeta)
#     level2 = (c1[: , None] * xi_xi + c2[: , None] * eta_eta + c3[: , None] * xi_eta + c4[: , None] * zeta_zeta) / (2 * (c1[: , None] + c2[: , None]))
#     # level2 = (c1[: , None] * (xi_xi + phi[: , None] * xi) + c2[: , None] * (eta_eta + psi[: , None] * eta) + c3[: , None] * xi_eta + c4[: , None] * zeta_zeta) / (2 * (c1[: , None] + c2[: , None]))
#     level2_normals, level2_normals_norm = get_layer_normals(level2, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices)
#     print("it", it)
#     # print(xi.dtype)
#     # print(xi[2447, :])
#     # print(directNeighbours[2447])
#     # print(simpleDiagonals[2447])
#
# # print(level2[500:750])
# level3 = level2 + d2 * level2_normals_norm
#
# print(points3[2448])
# print(level2[0])
#
# smoothMesh_points = np.concatenate((surface_points, level2))
# level2Surface = [["triangle", surface_triangles], ["quad", surface_quads]]
# # voxels1 = [["wedge", np.concatenate((surface_triangles, level2_triangles),axis=1)], ["hexahedron", np.concatenate((surface_quads, level2_quads),axis=1)]]
# # voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)], ["hexahedron", np.concatenate((level2_quads, level3_quads),axis=1)]]
# # voxels1and2 = voxels1
# # voxels1and2.extend(voxels2)
# V6_smoothMesh = meshio.Mesh(points = smoothMesh_points, cells = voxels1)
# V6_surface = meshio.Mesh(points = level2, cells = level2Surface)
# meshio.write("./output/mixedT_smoothMesh.vtk", V6_smoothMesh, file_format="vtk", binary=False)
# meshio.write("./output/mixedT_surface.vtk", V6_surface, file_format="vtk", binary=False)
# # meshio.write("./output/ref_mesh_test2.msh", V5_mesh, file_format="gmsh22", binary=False)

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
