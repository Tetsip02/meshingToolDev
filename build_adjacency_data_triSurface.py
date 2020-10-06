import meshio
import numpy as np
# import math
from helpers import *
# import json
import pickle

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

surface_triangles, surface_quads, surface_points = read_surface("./testGeometries/P023rotor0.stl")

######################################################################

nPoints = len(surface_points)
nFaces = len(surface_triangles)

######################################

# build faceIndices for each point
triFaceIndices = [] # [nPoints, number_of_attached_triangle_faces]
valence = [] # [nPoints, 1]
neighbours = []
diagonals = []
for index in range(nPoints):
    # get face indices for each point split into quad and tri lists
    mask1 = np.isin(surface_triangles, index)
    triFaceIndices.append(np.nonzero(mask1)[0])

    # get total number of attached faces for each point = direct valence
    indexValence = len(np.nonzero(mask1)[0])
    valence.append(indexValence)


    # # get relevant neighbour nodes used later for calculating derivatives
    # # initialize direct neighbours for point index
    indexNeighbours = []
    indexSimpleDiagonals = []
    indexInterDiagonals = []

    if indexValence > 4:
        # add all points of attached triangles
        for faceIndex in np.nonzero(mask1)[0]:
            indexNeighbours.extend(surface_triangles[faceIndex , :])

        # remove duplicate points ##
        indexNeighbours = list(set(indexNeighbours))
        # remove point itself from list of neighbours
        try:
            indexNeighbours.remove(index)
        except ValueError:
            pass

    elif indexValence == 4:
        # add all points of attached triangles
        for faceIndex, pos in zip(np.nonzero(mask1)[0], np.nonzero(mask1)[1]):
            indexNeighbours.extend(surface_triangles[faceIndex , :])

        # find faces that share two nodes with attached triangles, if thats a triangle add to simpleDiagonals, if quad add both diagonals to interDiagonals
        for triNeighbour in np.nonzero(mask1)[0]:
            x = surface_triangles[triNeighbour, :]
            for triFace in surface_triangles:
                triDiaMask = np.isin(x, triFace)
                if np.sum(triDiaMask) == 2 and index not in triFace:
                    triFaceReduced = [e for e in triFace if e not in x]
                    indexSimpleDiagonals.append(triFaceReduced[0])

        # remove duplicate points ##
        indexNeighbours = list(set(indexNeighbours))
        # remove point itself from list of neighbours
        try:
            indexNeighbours.remove(index)
        except ValueError:
            pass

    elif indexValence == 3:
        # add all points of attached triangles
        for faceIndex, pos in zip(np.nonzero(mask1)[0], np.nonzero(mask1)[1]):
            indexNeighbours.extend(surface_triangles[faceIndex , :])

        # find faces that share two nodes with attached triangles, if thats a triangle add to directNeighbours, if quad add both diagonals to interDiagonals
        for triNeighbour in np.nonzero(mask1)[0]:
            x = surface_triangles[triNeighbour, :]
            for triFace in surface_triangles:
                triDiaMask = np.isin(x, triFace)
                if np.sum(triDiaMask) == 2 and index not in triFace:
                    triFaceReduced = [e for e in triFace if e not in x]
                    indexNeighbours.append(triFaceReduced)

        # remove duplicate points ##
        indexNeighbours = list(set(indexNeighbours))
        # remove point itself from list of neighbours
        try:
            indexNeighbours.remove(index)
        except ValueError:
            pass


    neighbours.append(indexNeighbours)
    diagonals.append(indexSimpleDiagonals)


#####################################################################################################
# get normals
level1_normals, level1_normals_norm = get_layer_normals(surface_points, surface_triangles, triFaceIndices)

###########################################################################################################
# fix neighbour orientation

directNeighbours_sorted = []
neighbours_and_diagonals_sorted = []

for vertex, neighInd in enumerate(neighbours):
    allInd_sorted = []
    if valence[vertex] > 4: # case 1: direct valence >= 5
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

    elif valence[vertex] == 3: # simpleDiagonals is empty, only consider direct neighbours and inter diagonals
        neigh = surface_points[neighInd]
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
        allInd_sorted = allInd.copy()
        for i, key in enumerate(key):
            allInd_sorted[i + 1] = allInd[key]
        neighInd_sorted = allInd_sorted
        # directNeighbours_sorted.append(allInd_sorted)

    elif valence[vertex] == 4: # output list of all neighbours including diagonals and list of direct neighbours only
        neigh = surface_points[neighInd]
        if diagonals[vertex]:
            simpleNeigh = surface_points[diagonals[vertex]]
            neigh = np.concatenate((neigh, simpleNeigh))
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
        allInd.extend(diagonals[vertex])
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






#####################################################################
# write data

with open("./dataOutput/P023rotor0_triFaceIndices.txt", "wb") as fp:   #Pickling
    pickle.dump(triFaceIndices, fp)

with open("./dataOutput/P023rotor0_directValence.txt", "wb") as fp:   #Pickling
    pickle.dump(valence, fp)

with open("./dataOutput/P023rotor0_directNeighbours.txt", "wb") as fp:   #Pickling
    pickle.dump(directNeighbours_sorted, fp)

with open("./dataOutput/P023rotor0_neighbours_and_diagonals.txt", "wb") as fp:   #Pickling
    pickle.dump(neighbours_and_diagonals_sorted, fp)

# with open("./dataOutput/V2_simpleDiagonals.txt", "wb") as fp:   #Pickling
#     pickle.dump(simpleDiagonals, fp)
#
# with open("./dataOutput/V2_interDiagonals.txt", "wb") as fp:   #Pickling
#     pickle.dump(interDiagonals, fp)

####################################################################
