import meshio
import numpy as np
# import math
from helpers import *
# import json
import pickle

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

surface_triangles, surface_quads, surface_points = read_surface("./testGeometries/sphere.obj")

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

#####################################################################################################
# get normals
level1_normals, level1_normals_norm = get_layer_normals(surface_points, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices)

###########################################################################################################
# fix neighbour orientation

directNeighbours_sorted = []
neighbours_and_diagonals_sorted = []

for vertex, neighInd in enumerate(directNeighbours):
    allInd_sorted = []
    if directValence[vertex] > 4: # case 1: direct valence >= 5
        neigh = surface_points[neighInd] # neighbour coordinates
        centroid = surface_points[vertex] # vertex in question
        plane = level1_normals_norm[vertex] #vertex normal
        neigh_proj = neigh - np.dot(neigh - centroid, plane)[:, None] * plane # vertex neighbours projected onto plane
        start = neigh_proj[0] # start ordering from neighbour with index 0

        orient = np.cross(start, neigh_proj[1:]) # determine direction of neighbour relative to starting point
        orient = orient / np.linalg.norm(orient, axis = 1)[: , None]
        orient = np.dot(orient, plane) # positive if point is in clockwise direction, negative if in CCW direction
        orient /= np.abs(orient) # set abs(orient) to 1

        vector = neigh_proj - centroid
        vector = vector / np.linalg.norm(vector, axis = 1)[: , None]

        refvec = vector[0] # vector from which angles are measured

        angle = np.dot(vector[1:], refvec)
        angle = np.arccos(angle)
        for i, orient in enumerate(orient):
            if orient < 0:
                angle[i] = 2 * np.pi - angle[i]

        key = np.argsort(angle)
        key += 1
        neighInd = np.array(neighInd)
        neighInd_sorted = neighInd.copy()
        neighInd_sorted[1:] = neighInd_sorted[key]
        # directNeighbours_sorted.append(neighInd_sorted)

    elif directValence[vertex] == 3: # simpleDiagonals is empty, only consider direct neighbours and inter diagonals
        neigh = surface_points[neighInd]
        if interDiagonals[vertex]:
            interNeigh = np.mean(np.array(surface_points[interDiagonals[vertex], :]), axis = 1)
            neigh = np.concatenate((neigh, interNeigh))
        centroid = surface_points[vertex]
        plane = level1_normals_norm[vertex]
        neigh_proj = neigh - np.dot(neigh - centroid, plane)[:, None] * plane
        start = neigh_proj[0]
        orient = np.cross(start, neigh_proj[1:])
        orient = orient / np.linalg.norm(orient, axis = 1)[: , None]
        orient = np.dot(orient, plane)
        orient /= np.abs(orient)
        vector = neigh_proj - centroid
        vector = vector / np.linalg.norm(vector, axis = 1)[: , None]
        refvec = vector[0]
        angle = np.dot(vector[1:], refvec)
        angle = np.arccos(angle)
        for i, orient in enumerate(orient):
            if orient < 0:
                angle[i] = 2 * np.pi - angle[i]
        key = np.argsort(angle)
        key += 1

        allInd = neighInd.copy()
        allInd.extend(interDiagonals[i])
        allInd_sorted = allInd.copy()
        for i, key in enumerate(key):
            allInd_sorted[i + 1] = allInd[key]
        neighInd_sorted = allInd_sorted
        # directNeighbours_sorted.append(allInd_sorted)

    elif directValence[vertex] == 4: # output list of all neighbours including diagonals and list of direct neighbours only
        neigh = surface_points[neighInd]
        if simpleDiagonals[vertex]:
            simpleNeigh = surface_points[simpleDiagonals[vertex]]
            neigh = np.concatenate((neigh, simpleNeigh))
        if interDiagonals[vertex]:
            interNeigh = np.mean(np.array(surface_points[interDiagonals[vertex], :]), axis = 1)
            neigh = np.concatenate((neigh, interNeigh))
        centroid = surface_points[vertex]
        plane = level1_normals_norm[vertex]
        neigh_proj = neigh - np.dot(neigh - centroid, plane)[:, None] * plane
        start = neigh_proj[0]
        orient = np.cross(start, neigh_proj[1:])
        orient = orient / np.linalg.norm(orient, axis = 1)[: , None]
        orient = np.dot(orient, plane)
        orient /= np.abs(orient)
        vector = neigh_proj - centroid
        vector = vector / np.linalg.norm(vector, axis = 1)[: , None]
        refvec = vector[0]
        angle = np.dot(vector[1:], refvec)
        angle = np.arccos(angle)
        for i, orient in enumerate(orient):
            if orient < 0:
                angle[i] = 2 * np.pi - angle[i]
        key = np.argsort(angle)
        key += 1
        allInd = neighInd.copy()
        allInd.extend(simpleDiagonals[vertex])
        allInd.extend(interDiagonals[vertex])
        allInd_sorted = allInd.copy()
        for i, key in enumerate(key):
            allInd_sorted[i + 1] = allInd[key]
        neighInd_sorted = []
        for i in allInd_sorted:
            try:
                if len(i):
                    pass
            except TypeError:
                if i not in simpleDiagonals[vertex]:
                    neighInd_sorted.append(i)

    directNeighbours_sorted.append(neighInd_sorted)
    neighbours_and_diagonals_sorted.append(allInd_sorted)



print(len(neighbours_and_diagonals_sorted))
print(len(directNeighbours_sorted))

# print(len(directNeighbours_sorted))
# print(nPoints)
# vertex = 1
# neighInd = directNeighbours[vertex]
# print(neighInd)
# print(simpleDiagonals[vertex])
# print(interDiagonals[vertex])
# neigh = surface_points[neighInd]
# if simpleDiagonals[vertex]:
#     simpleNeigh = surface_points[simpleDiagonals[vertex]]
#     neigh = np.concatenate((neigh, simpleNeigh))
# if interDiagonals[vertex]:
#     interNeigh = np.mean(np.array(surface_points[interDiagonals[vertex], :]), axis = 1)
#     neigh = np.concatenate((neigh, interNeigh))
# print(neigh)
# centroid = surface_points[vertex]
# plane = level1_normals_norm[vertex]
# neigh_proj = neigh - np.dot(neigh - centroid, plane)[:, None] * plane
# start = neigh_proj[0]
# orient = np.cross(start, neigh_proj[1:])
# orient = orient / np.linalg.norm(orient, axis = 1)[: , None]
# orient = np.dot(orient, plane)
# orient /= np.abs(orient)
# vector = neigh_proj - centroid
# vector = vector / np.linalg.norm(vector, axis = 1)[: , None]
# refvec = vector[0]
# angle = np.dot(vector[1:], refvec)
# angle = np.arccos(angle)
# for i, orient in enumerate(orient):
#     if orient < 0:
#         angle[i] = 2 * np.pi - angle[i]
# key = np.argsort(angle)
# key += 1
# print(key)
# allInd = neighInd.copy()
# allInd.extend(simpleDiagonals[vertex])
# allInd.extend(interDiagonals[vertex])
# allInd_sorted = allInd.copy()
# print(len(allInd_sorted))
# for i, key in enumerate(key):
#     allInd_sorted[i + 1] = allInd[key]
# print(allInd_sorted)
# neighbours_and_diagonals_sorted.append(allInd_sorted)
# neighInd_sorted = []
# for i in allInd_sorted:
#     try:
#         if len(i):
#             pass
#     except TypeError:
#         if i not in simpleDiagonals[vertex]:
#             neighInd_sorted.append(i)
#
# print(neighInd_sorted)

    # if len(i) == 1
# print(removeList)
# for i in removeList:
#     # neighInd_sorted.remove(all(i))
#     print(i)
#     try:
#         # print("norError")
#         neighInd_sorted.remove(all(i))
#     except TypeError:
#         print("typeError")
#         neighInd_sorted.remove(i)
#     except ValueError:
#         pass
# for i in neighInd_sorted:
#     if i not in removeList:
#
# reduced = [x for x in neighInd_sorted if x not in simpleDiagonals[vertex] except ValueError: pass]
# print(neighInd_sorted)
# print(reduced)
