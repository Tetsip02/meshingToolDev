import meshio
import numpy as np
# import pickle
# from helpers import *
# from helpers import centroid, read_surface, get_face_normals, area, hex_contains, root_neighbour, bounding_box, triInBox
# import helpers
# import config

########################################################################

def centroid(vertexes):
     _x_list = [vertex [0] for vertex in vertexes]
     _y_list = [vertex [1] for vertex in vertexes]
     _z_list = [vertex [2] for vertex in vertexes]
     _len = len(vertexes)
     _x = sum(_x_list) / _len
     _y = sum(_y_list) / _len
     _z = sum(_z_list) / _len
     return(_x, _y, _z)

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

def get_face_normals(points, triangles, quads):
    # build face normals
    triNormal = np.ndarray((len(triangles), 3), dtype = np.float64)
    quadNormal = np.ndarray((len(quads), 3), dtype = np.float64)
    for i, triFace in enumerate(triangles):
        coor = points[triFace, :]
        triNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
    for i, quadFace in enumerate(quads):
        coor = points[quadFace, :]
        quadNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
    # normalize
    triNormal = triNormal / np.linalg.norm(triNormal, axis = 1)[: , None]
    quadNormal = quadNormal / np.linalg.norm(quadNormal, axis = 1)[: , None]

    return triNormal, quadNormal

#area of polygon poly
def area(poly, unitNormal):
    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unitNormal)
    return abs(result/2)

def hex_contains(hex_points, x): # points need to satisfy certain order
    # n1 = np.cross(hex_points[1] - hex_points[0], hex_points[2] - hex_points[0])
    n1 = np.array([0, 0, 1])
    x1 = np.dot(x - hex_points[0], n1)
    if x1 < 0:
        return False
    # n2 = np.cross(hex_points[7] - hex_points[4], hex_points[6] - hex_points[4])
    n2 = np.array([0, 0, -1])
    x2 = np.dot(x - hex_points[4], n2)
    if x2 < 0:
        return False
    # n3 = np.cross(hex_points[3] - hex_points[0], hex_points[7] - hex_points[0])
    n3 = np.array([1, 0, 0])
    x3 = np.dot(x - hex_points[0], n3)
    if x3 < 0:
        return False
    # n4 = np.cross(hex_points[5] - hex_points[1], hex_points[6] - hex_points[1])
    n4 = np.array([-1, 0, 0])
    x4 = np.dot(x - hex_points[1], n4)
    if x4 < 0:
        return False
    # n5 = np.cross(hex_points[4] - hex_points[0], hex_points[5] - hex_points[0])
    n5 = np.array([0, 1, 0])
    x5 = np.dot(x - hex_points[0], n5)
    if x5 < 0:
        return False
    # n6 = np.cross(hex_points[2] - hex_points[3], hex_points[6] - hex_points[3])
    n6 = np.array([0, -1, 0])
    x6 = np.dot(x - hex_points[3], n6)
    if x6 < 0:
        return False
    return True

def root_neighbour(target_root, direction):
    # target_root: index of root voxel
    # direction: {"U", "B", "R", "D", "F", "L"}
    # global FS
    if direction == "U":
        if target_root in TS:
            return -1 # out of bounds
        return target_root + nDivisions[0] * nDivisions[1]
    if direction == "B":
        if target_root in BS:
            return -1 # out of bounds
        return target_root + nDivisions[0]
    if direction == "R":
        if target_root in RS:
            return -1 # out of bounds
        return target_root + 1
    if direction == "D":
        if target_root in DS:
            return -1 # out of bounds
        return target_root - nDivisions[0] * nDivisions[1]
    if direction == "F":
        if target_root in FS:
            return -1 # out of bounds
        return target_root - nDivisions[0]
    if direction == "L":
        if target_root in FS:
            return -1 # out of bounds
        return target_root - 1

def bounding_box(points):
    P0 = np.min(points, axis = 0)
    P6 = np.max(points, axis = 0)
    dist = P6 - P0
    P1 = P0.copy()
    P1[0] += dist[0]
    P2 = P1.copy()
    P2[1] += dist[1]
    P3 = P0.copy()
    P3[1] += dist[1]
    P4 = P0.copy()
    P4[2] += dist[2]
    P5 = P1.copy()
    P5[2] += dist[2]
    P7 = P3.copy()
    P7[2] += dist[2]
    bbox = np.concatenate(([P0], [P1], [P2], [P3], [P4], [P5], [P6], [P7]), axis = 0)
    return bbox

def triInBox(AABB, triangle):
    h = np.abs((AABB[1, 0] - AABB[0, 0]) / 2)
    c = AABB[0] + h
    triangle -= c
    # test 1, a00
    p0 = triangle[0, 2] * triangle[1, 1] - triangle[0, 1] * triangle[1, 2]
    p2 = triangle[2, 2] * (triangle[1, 1] - triangle[0, 1]) - triangle[2, 1] * (triangle[1, 2] - triangle[0, 2])
    min_p = min(p0, p2)
    max_p = max(p0, p2)
    r = (np.abs(triangle[1, 2] - triangle[0, 2]) + np.abs(triangle[1, 1] - triangle[0, 1])) * h
    if r < min_p or -r > max_p:
        return 0; # False, no overlap
    # test 2, a10
    p0 = triangle[0, 0] * triangle[1, 2] - triangle[0, 2] * triangle[1, 0]
    p2 = triangle[2, 0] * (triangle[1, 2] - triangle[0, 2]) - triangle[2, 2] * (triangle[1, 0] - triangle[0, 0])
    min_p = min(p0, p2)
    max_p = max(p0, p2)
    r = (np.abs(triangle[1, 2] - triangle[0, 2]) + np.abs(triangle[1, 0] - triangle[0, 0])) * h
    if r < min_p or -r > max_p:
        return 0; # False, no overlap
    # test 3, a20
    p0 = triangle[0, 1] * triangle[1, 0] - triangle[0, 0] * triangle[1, 1]
    p2 = triangle[2, 1] * (triangle[1, 0] - triangle[0, 0]) - triangle[2, 0] * (triangle[1, 1] - triangle[0, 1])
    min_p = min(p0, p2)
    max_p = max(p0, p2)
    r = (np.abs(triangle[1, 1] - triangle[0, 1]) + np.abs(triangle[1, 0] - triangle[0, 0])) * h
    if r < min_p or -r > max_p:
        return 0; # False, no overlap
    # test 4, a01
    p0 = triangle[0, 2] * (triangle[2, 1] - triangle[1, 1]) - triangle[0, 1] * (triangle[2, 2] - triangle[1, 2])
    p1 = triangle[1, 2] * triangle[2, 1] - triangle[1, 1] * triangle[2, 2]
    min_p = min(p0, p1)
    max_p = max(p0, p1)
    r = (np.abs(triangle[2, 2] - triangle[1, 2]) + np.abs(triangle[2, 1] - triangle[1, 1])) * h
    if r < min_p or -r > max_p:
        return 0; # False, no overlap
    # test 5, a11
    p0 = triangle[0, 0] * (triangle[2, 2] - triangle[1, 2]) - triangle[0, 2] * (triangle[2, 0] - triangle[1, 0])
    p1 = triangle[1, 0] * triangle[2, 2] - triangle[1, 2] * triangle[2, 0]
    min_p = min(p0, p1)
    max_p = max(p0, p1)
    r = (np.abs(triangle[2, 2] - triangle[1, 2]) + np.abs(triangle[2, 0] - triangle[1, 0])) * h
    if r < min_p or -r > max_p:
        return 0; # False, no overlap
    # test 6, a21
    p0 = triangle[0, 1] * (triangle[2, 0] - triangle[1, 0]) - triangle[0, 0] * (triangle[2, 1] - triangle[1, 1])
    p1 = triangle[1, 1] * triangle[2, 0] - triangle[1, 0] * triangle[2, 1]
    min_p = min(p0, p1)
    max_p = max(p0, p1)
    r = (np.abs(triangle[2, 1] - triangle[1, 1]) + np.abs(triangle[2, 0] - triangle[1, 0])) * h
    if r < min_p or -r > max_p:
        return 0; # False, no overlap
    # test 7, a02
    p0 = triangle[0, 1] * triangle[2, 2] - triangle[0, 2] * triangle[2, 1]
    p1 = triangle[1, 2] * (triangle[0, 1] - triangle[2, 1]) - triangle[1, 1] * (triangle[0, 2] - triangle[2, 2])
    min_p = min(p0, p1)
    max_p = max(p0, p1)
    r = (np.abs(triangle[0, 2] - triangle[2, 2]) + np.abs(triangle[0, 1] - triangle[2, 1])) * h
    if r < min_p or -r > max_p:
        return 0; # False, no overlap
    # test 8, a12
    p0 = triangle[0, 2] * triangle[2, 0] - triangle[0, 0] * triangle[2, 2]
    p1 = triangle[1, 0] * (triangle[0, 2] - triangle[2, 2]) - triangle[1, 2] * (triangle[0, 0] - triangle[2, 0])
    min_p = min(p0, p1)
    max_p = max(p0, p1)
    r = (np.abs(triangle[0, 2] - triangle[2, 2]) + np.abs(triangle[0, 0] - triangle[2, 0])) * h
    if r < min_p or -r > max_p:
        return 0; # False, no overlap
    # test 9, a22
    p0 = triangle[0, 0] * triangle[2, 1] - triangle[0, 1] * triangle[2, 0]
    p1 = triangle[1, 1] * (triangle[0, 0] - triangle[2, 0]) - triangle[1, 0] * (triangle[0, 1] - triangle[2, 1])
    min_p = min(p0, p1)
    max_p = max(p0, p1)
    r = (np.abs(triangle[0, 1] - triangle[2, 1]) + np.abs(triangle[0, 0] - triangle[2, 0])) * h
    if r < min_p or -r > max_p:
        return 0; # False, no overlap
    # test 10, x-plane
    min_e = min(triangle[:, 0])
    max_e = max(triangle[:, 0])
    if h < min_e or -h > max_e:
        return 0; # False, no overlap
    # test 11, y-plane
    min_e = min(triangle[:, 1])
    max_e = max(triangle[:, 1])
    if h < min_e or -h > max_e:
        return 0; # False, no overlap
    # test 12, z-plane
    min_e = min(triangle[:, 2])
    max_e = max(triangle[:, 2])
    if h < min_e or -h > max_e:
        return 0; # False, no overlap
    # test 13, triangle normal
    n = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    v = triangle[0]
    min_v = -1 * v
    max_v = min_v.copy()
    for i, nor in enumerate(n):
        if nor > 0:
            min_v[i] -= h
            max_v[i] += h
        else:
            min_v[i] += h
            max_v[i] -= h
    res = np.dot(n, min_v)
    res2 = np.dot(n, max_v)
    if res > 0:
        return 0; # False, no overlap
    elif res2 > 0:
        # print("N:True, possibly overlap") # possibly overlap
        pass
    else:
        return 0; # False, no overlap
    return 1; # triangle intersects all 12 planes, and box intersects normal plane

####################################################

surface_triangles_ini, surface_quads_ini, surface_points = read_surface("./sphere.obj")

######################################################################

#change orientation
surface_triangles = surface_triangles_ini.copy()
surface_quads = surface_quads_ini.copy()
reverseOrientation = 1
if reverseOrientation == 1:
    surface_triangles[:,[1, 2]] = surface_triangles[:,[2, 1]]
    surface_quads[:,[1, 3]] = surface_quads[:,[3, 1]]

######################################

nPoints = len(surface_points)
nTriFaces = len(surface_triangles)
nQuadFaces = len(surface_quads)
nFaces = nTriFaces + nQuadFaces

# get list of face normals
triangleNormals, quadNormals = get_face_normals(surface_points, surface_triangles, surface_quads)

# build neighbour faces for each point on surface mesh
triFaceIndices = [] # [nPoints, number_of_attached_triangle_faces]
quadFaceIndices = [] # [nPoints, number_of_attached_quad_faces]
directValence = [] # [nPoints, 1]
for index in range(nPoints):
    # get face indices for each point split into quad and tri lists
    mask1 = np.isin(surface_triangles, index)
    triFaceIndices.append(np.nonzero(mask1)[0].tolist())
    # triFaceIndices.extend(np.nonzero(mask1)[0])
    mask2 = np.isin(surface_quads, index)
    quadFaceIndices.append(np.nonzero(mask2)[0].tolist())
    # quadFaceIndices.extend(np.nonzero(mask2)[0])

    # get total number of attached faces for each point = direct valence
    indexValence = len(np.nonzero(mask1)[0]) + len(np.nonzero(mask2)[0])
    directValence.append(indexValence)

# get list of face centroids
# faceCentroids = []
# faceArea = []
# sizeField = []
# triangleSizeField = [] #new
# quadSizeField = [] #new
triangleFaceArea = [] #new
quadFaceArea = [] #new
# targetLevel = []
triangleTargetLevel = [] #new
quadTargetLevel = [] #new
triSizeFieldFactor = 2 * 3 ** -0.25
maxCellSize = 1.2
maxLevel = 10
transLayers = 2

for i, face in enumerate(surface_triangles):
    # faceCentroids.append(centroid(surface_points[face]))
    triArea = area(surface_points[face], triangleNormals[i])
    # faceArea.append(triArea)
    triangleFaceArea.append(triArea) #new
    edgeSize = triSizeFieldFactor * (triArea ** 0.5)
    # sizeField.append(edgeSize)
    # triangleSizeField.append(edgeSize) #new
    r = np.log2(maxCellSize/edgeSize)
    triangleTargetLevel.append(r) #new
    # if np.abs(2 ** np.int(r) - maxCellSize) < np.abs(2 ** (np.int(r) + 1) - maxCellSize):
    #     targetLevel.append(np.int(r))
    #     # triangleTargetLevel.append(np.int(r)) #new
    # else:
    #     targetLevel.append(np.int(r) + 1)
    #     # triangleTargetLevel.append(np.int(r) + 1) #new
for i, face in enumerate(surface_quads):
    # faceCentroids.append(centroid(surface_points[face]))
    quadArea = area(surface_points[face], quadNormals[i])
    # faceArea.append(quadArea)
    quadFaceArea.append(quadArea) #new
    edgeSize = quadArea ** 0.5
    # sizeField.append(edgeSize)
    # quadSizeField.append(edgeSize) #new
    r = np.log2(maxCellSize/edgeSize)
    quadTargetLevel.append(r) #new
    # if np.abs(2 ** np.int(r) - maxCellSize) < np.abs(2 ** (np.int(r) + 1) - maxCellSize):
    #     targetLevel.append(np.int(r))
    #     # quadTargetLevel.append(np.int(r)) #new
    # else:
    #     targetLevel.append(np.int(r) + 1)
    #     # quadTargetLevel.append(np.int(r) + 1) #new

# faceCentroids = np.array(faceCentroids)
# faceArea = np.array(faceArea)
# sizeField = np.array(sizeField)
# targetLevel = np.array(targetLevel)
# triangleSizeField = np.array(triangleSizeField) #new
# quadSizeField = np.array(quadSizeField) #new
triangleFaceArea = np.array(triangleFaceArea) #new
quadFaceArea = np.array(quadFaceArea) #new
triangleTargetLevel = np.array(triangleTargetLevel) #new
quadTargetLevel = np.array(quadTargetLevel) #new

# compute target level / size field of each surface point, weighted average of neighbour faces
pointTargetLevel = []
for index in range(nPoints):
    neigh_targetLevels = np.concatenate((triangleTargetLevel[triFaceIndices[index]], quadTargetLevel[quadFaceIndices[index]]), axis = 0)
    neigh_weights = np.concatenate((triangleFaceArea[triFaceIndices[index]], quadFaceArea[quadFaceIndices[index]]), axis = 0)
    avg_target = np.average(neigh_targetLevels, weights = neigh_weights)

    if np.abs(2 ** np.int(avg_target) - maxCellSize) < np.abs(2 ** (np.int(avg_target) + 1) - maxCellSize):
        pointTargetLevel.append(np.int(avg_target))
    else:
        pointTargetLevel.append(np.int(avg_target) + 1)

pointTargetLevel = np.array(pointTargetLevel)
maxLevel = min(np.max(pointTargetLevel), maxLevel)
if maxLevel == 0:
    print("nothing to refine")

# set up base mesh
bBox_origin = np.array([0.0, 0.0, 0.0])
xmin = 6.0
xmax = 6.0
ymin = 6.0
ymax = 6.0
zmin = 6.0
zmax = 6.0
bBox = np.array([-xmin, xmax, -ymin, ymax, -zmin, zmax])
bBox_length = np.array([xmin + xmax, ymin + ymax, zmin + zmax])
nDivisions = np.array(bBox_length / maxCellSize, dtype = np.int)
p0 = bBox_origin - np.array([xmin, ymin, zmin]) # coordinates of bottom-left-forward corner

# get basemesh points
Px = []
for x in range(nDivisions[0] + 1):
    Px.append([x, 0, 0])
Px = np.array(Px)
Py = []
for y in range(nDivisions[1] + 1):
    Py.extend(Px + [0, y, 0])
Py = np.array(Py)
Pz = []
for z in range(nDivisions[2] + 1):
    Pz.extend(Py + [0, 0, z])
Pz = np.array(Pz)
l0_points = maxCellSize * Pz + p0

#get base mesh cells
z_lay = (nDivisions[0] + 1) * (nDivisions[1] + 1) # number of points of each z=constant plane

x_hexas = []
for x in range(nDivisions[0]):
    # hex = [x, nDivisions[0] + x + 1, nDivisions[0] + x + 2, x + 1]
    hex = [x, x + 1, nDivisions[0] + x + 2, nDivisions[0] + x + 1]
    hex.extend(hex + z_lay)
    x_hexas.append(hex)
x_hexas = np.array(x_hexas)
y_hexas = []
for y in range(nDivisions[1]):
    y_hexas.extend(x_hexas + y * (nDivisions[0] + 1))
y_hexas = np.array(y_hexas)
z_hexas = []
for z in range(nDivisions[2]):
    z_hexas.extend(y_hexas + z * z_lay)
z_hexas = np.array(z_hexas)

voxel1 = [("hexahedron", z_hexas)]
mesh = meshio.Mesh(points = l0_points, cells = voxel1)
meshio.write("./basemesh.vtk", mesh, file_format="vtk", binary=False)
meshio.write("./basemesh.msh", mesh, file_format="gmsh22", binary=False)


rootIndex = np.arange(z_hexas.shape[0])

# down side:
DS = np.arange(nDivisions[0] * nDivisions[1])
# top Side:
TS = DS + nDivisions[0] * nDivisions[1] * (nDivisions[2] - 1)
# left side
LS = []
for a in range(nDivisions[1]):
    for b in range(nDivisions[2]):
        LS.append(a * nDivisions[0] + b * (nDivisions[0] * nDivisions[1]))
LS = np.array(LS)
# right side:
RS = LS + (nDivisions[0] - 1)
# front side:
FS = []
for c in range(nDivisions[0]):
    for d in range(nDivisions[2]):
        FS.append(c + d * (nDivisions[0] * nDivisions[1]))
FS = np.array(FS)
# back side:
BS = FS + (nDivisions[0] * (nDivisions[1] - 1))

# print(FS)
# x = root_neighbour(820, "F")

############################################################
# variables at this point
# rootIndex [n_l0_cells, 1] = [0, 1, 2, ..., n_l0_cells - 1]
# pointTargetLevel [number of surface_points, 1]
#################################################

def all_root_neighbours(target_root):
    # target_root: index of root voxel
    neighbours = []
    if target_root not in TS:
        neighbours.append(target_root + nDivisions[0] * nDivisions[1]) # upper neighbour
    if target_root not in BS:
        neighbours.append(target_root + nDivisions[0]) # back neighbour
    if target_root not in RS:
        neighbours.append(target_root + 1) # right neighbour
    if target_root not in DS:
        neighbours.append(target_root - nDivisions[0] * nDivisions[1]) # down neighbour
    if target_root not in FS:
        neighbours.append(target_root - nDivisions[0]) # front neighbour
    if target_root not in LS:
        neighbours.append(target_root - 1) # left neighbour
    return neighbours

# level 0 refinement
# create mask of surface points whose level is above 0
# select random point from mask
# locate start cell (cell whose midpoint is closest to seleceted surface point)
# add start cell to list of target cells
# remove start cell from list of non targets
# compute influence radius
# locate neighbours of start cell
# for each neighbour
#   get midpoint of neighbour
#   locate surface point (from list of mask) closest to midpoint of neighbour
#   inflate neighbour by influence radius
#   if inflated neighbour contains the surface point
#       add neighbour to list of target cells
#       remove neighbour from list of non targets
# for each neighbour that was selceted as a target
#   compute their own neighbours
#   add new neighbours to list of new neighbours if new neighbour is not start cell or inside list of old neighbours
# compute new targets, add new list of neighbours to old one, and repeat
# stop condition: list of new neighbours is empty

level = 0
rInf = transLayers * maxCellSize * 0.5 ** level
all_points = l0_points.copy()
level0_mesh = z_hexas.copy()

level0_targets = [] # refinement targets
level0_non_targets = rootIndex.copy() # list of cells to be left alone
level0_checked = [] # level 0 cells which have already been tested whether they need refinement

surface_0_plus_mask = np.array(np.where(pointTargetLevel > 0)).ravel() # surface point indices whose target level is above zero

level0_midpoints = all_points[level0_mesh[:, 0]] + maxCellSize * 0.5 # coordinates of level0 midpoints
distToMidpoint = np.sum((surface_points[surface_0_plus_mask[0]] - level0_midpoints) ** 2, axis = 1) ** 0.5 # list of distances between level0_midpoints and random point on surface whose target level is above 0
start_cell = np.argmin(distToMidpoint) # index of cell closest to selceted point on surface

level0_targets.append(start_cell) # add start_cell to list of targets
level0_non_targets = np.delete(level0_non_targets, start_cell)

level0_checked.append(start_cell) # add targets to list of cells that needn't be checked again

root_neighbours = all_root_neighbours(start_cell) # neighbours (in ortho-directions) of start_cell

# for cellIndex in root_neighbours:
#     midpoint = level0_midpoints[cellIndex] # midpoint of neighbour
#     distToMidpoint = np.sum((surface_points[surface_0_plus_mask] - midpoint) ** 2, axis = 1) ** 0.5 # distance between midpoint and surface_points whose level is above 0
#     test_surface_point = surface_points[surface_0_plus_mask[np.argmin(distToMidpoint)]] # coordinates of surface point closest to midpoint
#     hex = rInf * all_points[level0_mesh[cellIndex]] # coordinates of inflated neighbour
#     if hex_contains(hex, test_surface_point): # condition for whether neighbour needs to be refined
#         level0_targets.append(cellIndex) # if condition is met, add neighbour to list of refinement targets
#         level0_non_targets = np.delete(level0_non_targets, cellIndex) # if condition is met, remove neighbour from list of non-targets
# print(level0_targets)

tempTargets = []
for cellIndex in root_neighbours:
    midpoint = level0_midpoints[cellIndex] # midpoint of neighbour
    distToMidpoint = np.sum((surface_points[surface_0_plus_mask] - midpoint) ** 2, axis = 1) ** 0.5 # distance between midpoint and surface_points whose level is above 0
    test_surface_point = surface_points[surface_0_plus_mask[np.argmin(distToMidpoint)]] # coordinates of surface point closest to midpoint
    hex = rInf * all_points[level0_mesh[cellIndex]] # coordinates of inflated neighbour
    if hex_contains(hex, test_surface_point): # condition for whether neighbour needs to be refined
        tempTargets.append(cellIndex)
print(tempTargets)
if len(tempTargets) == 0:
    print("nothing else to refine")
else:
    level0_targets.extend(tempTargets)
print(level0_targets)

level0_checked.extend(root_neighbours) # # add target neighbours to list of cells that needn't be checked again
print("checked", level0_checked)

# go through each neighbour that was selected and compute their neighbours
root_neighbours = []
for cellIndex in tempTargets:
    root_neighbours.extend(all_root_neighbours(cellIndex))
# remove duplicates
root_neighbours = list(set(root_neighbours))
# remove cells which have have already been checked for refinement
root_neighbours = [x for x in root_neighbours if x not in level0_checked]
print(root_neighbours)

# assess each new root neighbour on whether they need refining
tempTargets = []
for cellIndex in root_neighbours:
    midpoint = level0_midpoints[cellIndex] # midpoint of neighbour
    distToMidpoint = np.sum((surface_points[surface_0_plus_mask] - midpoint) ** 2, axis = 1) ** 0.5 # distance between midpoint and surface_points whose level is above 0
    test_surface_point = surface_points[surface_0_plus_mask[np.argmin(distToMidpoint)]] # coordinates of surface point closest to midpoint
    hex = rInf * all_points[level0_mesh[cellIndex]] # coordinates of inflated neighbour
    if hex_contains(hex, test_surface_point): # condition for whether neighbour needs to be refined
        tempTargets.append(cellIndex)
print(tempTargets)
if len(tempTargets) == 0:
    print("nothing else to refine")
else:
    level0_targets.extend(tempTargets)
print(level0_targets)

voxel1 = [("hexahedron", level0_mesh[level0_targets])]
mesh = meshio.Mesh(points = all_points, cells = voxel1)
meshio.write("./it2Targets.vtk", mesh, file_format="vtk", binary=False)

# print(root_neighbours)
level0_checked.extend(root_neighbours) # # add target neighbours to list of cells that needn't be checked again
print("checked", level0_checked)

# go through each neighbour that was selected and compute their neighbours
root_neighbours = []
for cellIndex in tempTargets:
    root_neighbours.extend(all_root_neighbours(cellIndex))
# remove duplicates
root_neighbours = list(set(root_neighbours))
# remove cells which have have already been checked for refinement
root_neighbours = [x for x in root_neighbours if x not in level0_checked]
print(root_neighbours)

# assess each new root neighbour on whether they need refining
tempTargets = []
for cellIndex in root_neighbours:
    midpoint = level0_midpoints[cellIndex] # midpoint of neighbour
    distToMidpoint = np.sum((surface_points[surface_0_plus_mask] - midpoint) ** 2, axis = 1) ** 0.5 # distance between midpoint and surface_points whose level is above 0
    test_surface_point = surface_points[surface_0_plus_mask[np.argmin(distToMidpoint)]] # coordinates of surface point closest to midpoint
    hex = rInf * all_points[level0_mesh[cellIndex]] # coordinates of inflated neighbour
    if hex_contains(hex, test_surface_point): # condition for whether neighbour needs to be refined
        tempTargets.append(cellIndex)
print(tempTargets)
if len(tempTargets) == 0:
    print("nothing else to refine")
else:
    level0_targets.extend(tempTargets)
print(level0_targets)

voxel1 = [("hexahedron", level0_mesh[level0_targets])]
mesh = meshio.Mesh(points = all_points, cells = voxel1)
meshio.write("./it3Targets.vtk", mesh, file_format="vtk", binary=False)

level0_checked.extend(root_neighbours) # # add target neighbours to list of cells that needn't be checked again
print("checked", level0_checked)

# go through each neighbour that was selected and compute their neighbours
root_neighbours = []
for cellIndex in tempTargets:
    root_neighbours.extend(all_root_neighbours(cellIndex))
# remove duplicates
root_neighbours = list(set(root_neighbours))
# remove cells which have have already been checked for refinement
root_neighbours = [x for x in root_neighbours if x not in level0_checked]
print(root_neighbours)

# assess each new root neighbour on whether they need refining
tempTargets = []
for cellIndex in root_neighbours:
    midpoint = level0_midpoints[cellIndex] # midpoint of neighbour
    distToMidpoint = np.sum((surface_points[surface_0_plus_mask] - midpoint) ** 2, axis = 1) ** 0.5 # distance between midpoint and surface_points whose level is above 0
    test_surface_point = surface_points[surface_0_plus_mask[np.argmin(distToMidpoint)]] # coordinates of surface point closest to midpoint
    hex = rInf * all_points[level0_mesh[cellIndex]] # coordinates of inflated neighbour
    if hex_contains(hex, test_surface_point): # condition for whether neighbour needs to be refined
        tempTargets.append(cellIndex)
print(tempTargets)
if len(tempTargets) == 0:
    print("nothing else to refine")
else:
    level0_targets.extend(tempTargets)
print(level0_targets)

# cellIndex = b[0]
# for cellIndex in b: # for each neighbour
#     midpoint = level0_midpoints[cellIndex] # midpoint of neighbour
#     distToMidpoint = np.sum((surface_points - midpoint) ** 2, axis = 1) ** 0.5 # distance between midpoint and surface_points
# test_surface_point = np.argmin(distToMidpoint) # index of surface point closest to midpoint
# hex = rInf * all_points[level0_mesh[cellIndex]] # coordinates of inflated neighbour
# if hex_contains(hex, surface_points[test_surface_point]): # condition for whether neighbour needs to be refined
#     level0_targets.append(cellIndex) # if condition is met, add neighbour to list of refinement targets
#     level0_non_targets = np.delete(level0_non_targets, cellIndex) # if condition is met, remove neighbour from list of non-targets



# for index in range(nPoints): # select random point on surface whose target level is above 0
#     if pointTargetLevel[index] > 0:
#         distToMidpoint = np.sum((surface_points[index] - level0_midpoints) ** 2, axis = 1) ** 0.5 # list of distances between level0_midpoints and random point on surface
#         start_cell = np.argmin(distToMidpoint) # index of cell closest to selceted point on surface
#         break
#     else:
#         pass

# test_voxels = [("hexahedron", np.array([level0_mesh[cellIndex]]))]
# mesh = meshio.Mesh(points = all_points, cells = test_voxels)
# meshio.write("./test.vtk", mesh, file_format="vtk", binary=False)

# for cellIndex in b:
#     midpoint = level0_midpoints[cellIndex]
#     distToMidpoint = np.sum((surface_points - midpoint) ** 2, axis = 1) ** 0.5
#     test_surface_point = np.argmin(distToMidpoint)
#     hex = rInf * level0_mesh[cellIndex]
#     if hex_contains(hex, test_surface_point):
#         level0_targets.append(cellIndex)
#         level0_non_targets = np.delete(level0_non_targets, cellIndex)
# print(level0_targets)

# mask = np.array(np.where(distToMidpoint < rInf)).ravel() # indices of points inside area of influence
    # print(mask)
#     for i, leaf in enumerate(level0_mesh):
#         if np.any(np.in1d(leaf, mask)):
#             l0_trans_targets.append(i)
# ############
# l0_meshTemp = np.delete(level0_mesh, level0_targetCells, axis = 0)
# print(level0_mesh.shape)
# print(l0_meshTemp.shape)

# l0_trans_targets = []
# for midpoint in level2_midpoints:
#     distToMidpoint = np.sum((all_points - midpoint) ** 2, axis = 1) ** 0.5

#############################################################


# refinement:
# during refinement store root voxel of each cell and boolean saying whether cell is a leaf or not

newPointsTemplate = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [1, 2, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 1], [1, 1, 1], [2, 1, 1], [0, 2, 1], [1, 2, 1], [2, 2, 1], [1, 0, 2], [0, 1, 2], [1, 1, 2], [2, 1, 2], [1, 2, 2]])

newLevelTemplate = np.array([[0, 0, 2, 1, 5, 6, 9, 8], [0, 0, 3, 2, 6, 7, 10, 9], [2, 3, 0, 4, 9, 10, 13, 12], [1, 2, 4, 0, 8, 9, 12, 11], [5, 6, 9, 8, 0, 14, 16, 15], [6, 7, 10, 9, 14, 0, 17, 16], [9, 10, 13, 12, 16, 17, 0, 18], [8, 9, 12, 11, 15, 16, 18, 0]])

# # # produce culled mesh of root voxels that contain at least one surface mesh point:
# surface_bbox = bounding_box(surface_points)
# culled_l0_points = [] # all points contained by bounding_box of surface mesh
# for i, node in enumerate(l0_points):
#     if hex_contains(surface_bbox, node):
#         culled_l0_points.append(i)
# culled_l0_mesh_ind = []
# for i, V in enumerate(z_hexas):
#     common = np.intersect1d(V, culled_l0_points, assume_unique = True)
#     if common.size > 0:
#         culled_l0_mesh_ind.append(i)
# print(culled_l0_mesh_ind)
# culled_l0_mesh = z_hexas[culled_l0_mesh_ind]
#
# culled_l0_voxels = [("hexahedron", culled_l0_mesh)]
# mesh = meshio.Mesh(points = l0_points, cells = culled_l0_voxels)
# meshio.write("./culled_l0_mesh.vtk", mesh, file_format="vtk", binary=False)
#
#
#
# level = 0
# level0_targetPoints = []
# level0_targetCells = []
# to_refine = []
# has_children = []
# for i, leaf in zip(culled_l0_mesh_ind, culled_l0_mesh):
#     tempList = []
#     for j, target in enumerate(faceCentroids):
#         if hex_contains(l0_points[leaf], target):
#             tempList.append(j)
#     level0_targetPoints.append(tempList)
#     if len(tempList) != 0 and np.max(targetLevel[tempList]) > level and np.max(targetLevel[tempList]) < maxLevel:
#         level0_targetCells.append(i)
#         has_children.append(i)
#         to_refine.append(i)
#
#
# all_points = l0_points.copy()
# level0_mesh = z_hexas.copy()

# # refine marked level 0 cells:
# level = 1
# level1_mesh = []
# level1_targetPoints = [] # length = number of level1 cells, each entry contains list of target points in side each level1 cell
# level1_targetCells = [] # indices in level1_mesh of cells that need refining
# for i, targetCell in enumerate(level0_targetCells):
#     # assemble refined cells
#     new_level = np.diag(level0_mesh[targetCell])
#     template = newLevelTemplate + all_points.shape[0] #- 19
#     np.fill_diagonal(template, 0)
#     new_level += template # [8 x 8]
#     level1_mesh.extend(new_level)
#     # compute new points of children (19 new points per refinement)
#     new_points = newPointsTemplate * 0.5 ** level * maxCellSize + all_points[level0_mesh[targetCell][0]]
#     all_points = np.concatenate((all_points, new_points), axis = 0)
#     # compute targetCells and targetPoints among Level1 cells:
#     for j, child in enumerate(new_level):
#         tempList = []
#         # for targetPoint in level0_targetPoints[targetCell]:
#         for targetPoint in level0_targetPoints[i]:
#             if hex_contains(all_points[child], faceCentroids[targetPoint]):
#                 tempList.append(targetPoint)
#         level1_targetPoints.append(tempList)
#         if len(tempList) != 0 and np.max(targetLevel[tempList]) > level and np.max(targetLevel[tempList]) < maxLevel:
#             level1_targetCells.append((i * 8) + j)
#
# level1_mesh = np.array(level1_mesh)
#
# ##############
# temp_voxels = [("hexahedron", level1_mesh)]
# tempMesh = meshio.Mesh(points = all_points, cells = temp_voxels)
# meshio.write("./test.vtk", tempMesh, file_format="vtk", binary=False)
# ##############
#
# #create refined mesh
# level0_mesh_reduced = np.delete(level0_mesh, level0_targetCells, axis=0)
# refined_mesh = np.concatenate((level0_mesh_reduced, level1_mesh), axis = 0)
# all_voxels = [("hexahedron", refined_mesh)]
# level1_refinement = meshio.Mesh(points = all_points, cells = all_voxels)
# meshio.write("./level1_refinement.vtk", level1_refinement, file_format="vtk", binary=False)
# meshio.write("./level1_refinement.msh", level1_refinement, file_format="gmsh22", binary=False)
#
# level = 2
# level2_mesh = []
# level2_targetPoints = [] # length = number of level2 cells, each entry contains list of target points in side each level2 cell
# level2_targetCells = [] # indices in level2_mesh of cells that need refining
# for i, targetCell in enumerate(level1_targetCells):
#     # assemble refined cells
#     new_level = np.diag(level1_mesh[targetCell])
#     template = newLevelTemplate + all_points.shape[0] #- 19
#     np.fill_diagonal(template, 0)
#     new_level += template # [8 x 8]
#     level2_mesh.extend(new_level)
#     # compute new points of children (19 new points per refinement)
#     new_points = newPointsTemplate * 0.5 ** level * maxCellSize + all_points[level1_mesh[targetCell][0]]
#     all_points = np.concatenate((all_points, new_points), axis = 0)
#     # compute targetCells and targetPoints among Level1 cells:
#     for j, child in enumerate(new_level):
#         tempList = []
#         for targetPoint in level1_targetPoints[targetCell]:
#             if hex_contains(all_points[child], faceCentroids[targetPoint]):
#                 tempList.append(targetPoint)
#         level2_targetPoints.append(tempList)
#         if len(tempList) != 0 and np.max(targetLevel[tempList]) > level and np.max(targetLevel[tempList]) < maxLevel:
#             level2_targetCells.append((i * 8) + j)
# level2_mesh = np.array(level2_mesh)

# ##############
# temp_voxels = [("hexahedron", level2_mesh)]
# tempMesh = meshio.Mesh(points = all_points, cells = temp_voxels)
# meshio.write("./test.vtk", tempMesh, file_format="vtk", binary=False)
# ##############

# currentLevel = 2
# level2_midpoints = all_points[level2_mesh[:, 0]] + maxCellSize * 0.5 ** (currentLevel + 1)
# rInf = transLayers * maxCellSize * 0.5 ** (currentLevel - 1)
# ############
# l0_meshTemp = np.delete(level0_mesh, level0_targetCells, axis = 0)
# print(level0_mesh.shape)
# print(l0_meshTemp.shape)

# l0_trans_targets = []
# for midpoint in level2_midpoints:
#     distToMidpoint = np.sum((all_points - midpoint) ** 2, axis = 1) ** 0.5
#     mask = np.array(np.where(distToMidpoint < rInf)).ravel() # indices of points inside area of influence
    # print(mask)
#     for i, leaf in enumerate(level0_mesh):
#         if np.any(np.in1d(leaf, mask)):
#             l0_trans_targets.append(i)
# l0_trans_targets = np.array(l0_trans_targets)
# l0_trans_targets = np.unique(l0_trans_targets, axis = 0)
# with open("./l0_trans_targets.txt", "wb") as fp:   #Pickling
#     pickle.dump(l0_trans_targets, fp)
# with open("./l0_trans_targets.txt", "rb") as fp:   # Unpickling
#     l0_trans_targets = pickle.load(fp)
# #############
# # trans_voxels = [("hexahedron", level2_mesh_reduced[l2_trans_targets])]
# # trans_targets = meshio.Mesh(points = all_points, cells = trans_voxels)
# # meshio.write("./trans_targets.vtk", trans_targets, file_format="vtk", binary=False)
# # refine trans targets
# l0_trans_mesh = []
# for trans_target in l0_trans_targets:
#     # assemble refined cells
#     new_level = np.diag(level0_mesh_reduced[trans_target])
#     template = newLevelTemplate + all_points.shape[0]
#     np.fill_diagonal(template, 0)
#     new_level += template # [8 x 8]
#     l0_trans_mesh.extend(new_level)
#     # compute new points of children (19 new points per refinement)
#     new_points = newPointsTemplate * 0.5 ** (currentLevel - 1) * maxCellSize + all_points[level0_mesh_reduced[trans_target][0]]
#     all_points = np.concatenate((all_points, new_points), axis = 0)
# l0_trans_mesh = np.array(l0_trans_mesh)
# # trans_voxels2 = [("hexahedron", l2_trans_mesh)]
# # trans_targets2 = meshio.Mesh(points = all_points, cells = trans_voxels2)
# # meshio.write("./trans_targets2.vtk", trans_targets2, file_format="vtk", binary=False)

#
# #create refined mesh
# level1_mesh_reduced = np.delete(level1_mesh, level1_targetCells, axis=0)
# refined_mesh = np.concatenate((level0_mesh_reduced, level1_mesh_reduced, level2_mesh), axis = 0)
# all_voxels = [("hexahedron", refined_mesh)]
# level2_refinement = meshio.Mesh(points = all_points, cells = all_voxels)
# meshio.write("./level2_refinement.vtk", level2_refinement, file_format="vtk", binary=False)
# meshio.write("./level2_refinement.msh", level2_refinement, file_format="gmsh22", binary=False)
#
# ###################################################
#
# level = 3
# level3_mesh = []
# level3_targetPoints = []
# level3_targetCells = []
# for i, targetCell in enumerate(level2_targetCells):
#     # assemble refined cells
#     new_level = np.diag(level2_mesh[targetCell])
#     template = newLevelTemplate + all_points.shape[0] #- 19
#     np.fill_diagonal(template, 0)
#     new_level += template # [8 x 8]
#     level3_mesh.extend(new_level)
#     # compute new points of children (19 new points per refinement)
#     new_points = newPointsTemplate * 0.5 ** level * maxCellSize + all_points[level2_mesh[targetCell][0]]
#     all_points = np.concatenate((all_points, new_points), axis = 0)
#     # compute targetCells and targetPoints among Level1 cells:
#     for j, child in enumerate(new_level):
#         tempList = []
#         for targetPoint in level2_targetPoints[targetCell]:
#             if hex_contains(all_points[child], faceCentroids[targetPoint]):
#                 tempList.append(targetPoint)
#         level3_targetPoints.append(tempList)
#         if len(tempList) != 0 and np.max(targetLevel[tempList]) > level and np.max(targetLevel[tempList]) < maxLevel:
#             level3_targetCells.append((i * 8) + j)
# level3_mesh = np.array(level3_mesh)
#
# #create refined mesh
# level2_mesh_reduced = np.delete(level2_mesh, level2_targetCells, axis=0)
# refined_mesh = np.concatenate((level0_mesh_reduced, level1_mesh_reduced, level2_mesh_reduced, level3_mesh), axis = 0)
# all_voxels = [("hexahedron", refined_mesh)]
# level3_refinement = meshio.Mesh(points = all_points, cells = all_voxels)
# meshio.write("./level3_refinement.vtk", level3_refinement, file_format="vtk", binary=False)
# meshio.write("./level3_refinement.msh", level3_refinement, file_format="gmsh22", binary=False)
#
# level = 4
# level4_mesh = []
# level4_targetPoints = []
# level4_targetCells = []
# for i, targetCell in enumerate(level3_targetCells):
#     # assemble refined cells
#     new_level = np.diag(level3_mesh[targetCell])
#     template = newLevelTemplate + all_points.shape[0] #- 19
#     np.fill_diagonal(template, 0)
#     new_level += template # [8 x 8]
#     level4_mesh.extend(new_level)
#     # compute new points of children (19 new points per refinement)
#     new_points = newPointsTemplate * 0.5 ** level * maxCellSize + all_points[level3_mesh[targetCell][0]]
#     all_points = np.concatenate((all_points, new_points), axis = 0)
#     # compute targetCells and targetPoints among Level4 cells:
#     for j, child in enumerate(new_level):
#         tempList = []
#         for targetPoint in level3_targetPoints[targetCell]:
#             if hex_contains(all_points[child], faceCentroids[targetPoint]):
#                 tempList.append(targetPoint)
#         level4_targetPoints.append(tempList)
#         if len(tempList) != 0 and np.max(targetLevel[tempList]) > level and np.max(targetLevel[tempList]) < maxLevel:
#             level4_targetCells.append((i * 8) + j)
# level4_mesh = np.array(level4_mesh)
#
# #create refined mesh
# level3_mesh_reduced = np.delete(level3_mesh, level3_targetCells, axis=0)
# refined_mesh = np.concatenate((level0_mesh_reduced, level1_mesh_reduced, level2_mesh_reduced, level3_mesh_reduced, level4_mesh), axis = 0)
# all_voxels = [("hexahedron", refined_mesh)]
# level4_refinement = meshio.Mesh(points = all_points, cells = all_voxels)
# meshio.write("./level4_refinement.vtk", level4_refinement, file_format="vtk", binary=False)
# meshio.write("./level4_refinement.msh", level4_refinement, file_format="gmsh22", binary=False)
#
# #################################################################################
# # new strategy:
# # set currentLevel to 2
# # compute level 2 midpoints
# # set rInf to: transLayers * maxCellSize * 0.5 ** (currentLevel - 1)
# # cycle through level 0 cells and refine to level1 if one of its points is inside influence area
# # set currentLevel to 3
# # compute level 3 midpoints
# # set rInf to: transLayers * maxCellSize * 0.5 ** (currentLevel - 1)
# # cycle through level 1 cells and refine to level2 if one of its points is inside influence area
# # set currentLevel to 4
# # compute level 4 midpoints
# # set rInf to: transLayers * maxCellSize * 0.5 ** (currentLevel - 1)
# # cycle through level 2 cells and refine to level3 if one of its points is inside influence area
# # assembly:
# # level0_mesh_reduced - l0_trans_targets, l1_cells - l1_trans_targets, l2_cells - l2_trans_targets, level3_mesh_reduced, l2_trans_mesh, level4_mesh
#
# transLayers = 2
# currentLevel = 2
# level2_midpoints = all_points[level2_mesh[:, 0]] + maxCellSize * 0.5 ** (currentLevel + 1)
# rInf = transLayers * maxCellSize * 0.5 ** (currentLevel - 1)
# ############
# l0_trans_targets = []
# for midpoint in level2_midpoints:
#     distToMidpoint = np.sum((all_points - midpoint) ** 2, axis = 1) ** 0.5
#     mask = np.array(np.where(distToMidpoint < rInf)).ravel() # indices of points inside area of influence
#     for i, leaf in enumerate(level0_mesh_reduced):
#         if np.any(np.in1d(leaf, mask)):
#             l0_trans_targets.append(i)
# l0_trans_targets = np.array(l0_trans_targets)
# l0_trans_targets = np.unique(l0_trans_targets, axis = 0)
# # with open("./l0_trans_targets.txt", "wb") as fp:   #Pickling
# #     pickle.dump(l0_trans_targets, fp)
# # with open("./l0_trans_targets.txt", "rb") as fp:   # Unpickling
# #     l0_trans_targets = pickle.load(fp)
# #############
# # trans_voxels = [("hexahedron", level2_mesh_reduced[l2_trans_targets])]
# # trans_targets = meshio.Mesh(points = all_points, cells = trans_voxels)
# # meshio.write("./trans_targets.vtk", trans_targets, file_format="vtk", binary=False)
# # refine trans targets
# l0_trans_mesh = []
# for trans_target in l0_trans_targets:
#     # assemble refined cells
#     new_level = np.diag(level0_mesh_reduced[trans_target])
#     template = newLevelTemplate + all_points.shape[0]
#     np.fill_diagonal(template, 0)
#     new_level += template # [8 x 8]
#     l0_trans_mesh.extend(new_level)
#     # compute new points of children (19 new points per refinement)
#     new_points = newPointsTemplate * 0.5 ** (currentLevel - 1) * maxCellSize + all_points[level0_mesh_reduced[trans_target][0]]
#     all_points = np.concatenate((all_points, new_points), axis = 0)
# l0_trans_mesh = np.array(l0_trans_mesh)
# # trans_voxels2 = [("hexahedron", l2_trans_mesh)]
# # trans_targets2 = meshio.Mesh(points = all_points, cells = trans_voxels2)
# # meshio.write("./trans_targets2.vtk", trans_targets2, file_format="vtk", binary=False)
#
# currentLevel = 3
# level3_midpoints = all_points[level3_mesh[:, 0]] + maxCellSize * 0.5 ** (currentLevel + 1)
# rInf = transLayers * maxCellSize * 0.5 ** (currentLevel - 1)
# ############
# l1_cells = np.concatenate((level1_mesh_reduced, l0_trans_mesh), axis = 0)
# l1_trans_targets = []
# for midpoint in level3_midpoints:
#     distToMidpoint = np.sum((all_points - midpoint) ** 2, axis = 1) ** 0.5
#     mask = np.array(np.where(distToMidpoint < rInf)).ravel() # indices of points inside area of influence
#     for i, leaf in enumerate(l1_cells):
#         if np.any(np.in1d(leaf, mask)):
#             l1_trans_targets.append(i)
# l1_trans_targets = np.array(l1_trans_targets)
# l1_trans_targets = np.unique(l1_trans_targets, axis = 0)
# # with open("./l1_trans_targets.txt", "wb") as fp:   #Pickling
# #     pickle.dump(l1_trans_targets, fp)
# # with open("./l1_trans_targets.txt", "rb") as fp:   # Unpickling
# #     l1_trans_targets = pickle.load(fp)
# #############
# # trans_voxels = [("hexahedron", level2_mesh_reduced[l2_trans_targets])]
# # trans_targets = meshio.Mesh(points = all_points, cells = trans_voxels)
# # meshio.write("./trans_targets.vtk", trans_targets, file_format="vtk", binary=False)
# # refine trans targets
# l1_trans_mesh = []
# for trans_target in l1_trans_targets:
#     # assemble refined cells
#     new_level = np.diag(l1_cells[trans_target])
#     template = newLevelTemplate + all_points.shape[0]
#     np.fill_diagonal(template, 0)
#     new_level += template # [8 x 8]
#     l1_trans_mesh.extend(new_level)
#     # compute new points of children (19 new points per refinement)
#     new_points = newPointsTemplate * 0.5 ** (currentLevel - 1) * maxCellSize + all_points[l1_cells[trans_target][0]]
#     all_points = np.concatenate((all_points, new_points), axis = 0)
# l1_trans_mesh = np.array(l1_trans_mesh)
# # trans_voxels2 = [("hexahedron", l2_trans_mesh)]
# # trans_targets2 = meshio.Mesh(points = all_points, cells = trans_voxels2)
# # meshio.write("./trans_targets2.vtk", trans_targets2, file_format="vtk", binary=False)
#
# currentLevel = 4
# level4_midpoints = all_points[level4_mesh[:, 0]] + maxCellSize * 0.5 ** (currentLevel + 1)
# rInf = transLayers * maxCellSize * 0.5 ** (currentLevel - 1)
# ############
# l2_cells = np.concatenate((level2_mesh_reduced, l1_trans_mesh), axis = 0)
# l2_trans_targets = []
# for midpoint in level4_midpoints:
#     distToMidpoint = np.sum((all_points - midpoint) ** 2, axis = 1) ** 0.5
#     mask = np.array(np.where(distToMidpoint < rInf)).ravel() # indices of points inside area of influence
#     for i, leaf in enumerate(l2_cells):
#         if np.any(np.in1d(leaf, mask)):
#             l2_trans_targets.append(i)
# l2_trans_targets = np.array(l2_trans_targets)
# l2_trans_targets = np.unique(l2_trans_targets, axis = 0)
# # with open("./l2_trans_targets.txt", "wb") as fp:   #Pickling
# #     pickle.dump(l2_trans_targets, fp)
# # with open("./l2_trans_targets.txt", "rb") as fp:   # Unpickling
# #     l2_trans_targets = pickle.load(fp)
# #############
# # trans_voxels = [("hexahedron", level2_mesh_reduced[l2_trans_targets])]
# # trans_targets = meshio.Mesh(points = all_points, cells = trans_voxels)
# # meshio.write("./trans_targets.vtk", trans_targets, file_format="vtk", binary=False)
# # refine trans targets
# l2_trans_mesh = []
# for trans_target in l2_trans_targets:
#     # assemble refined cells
#     new_level = np.diag(l2_cells[trans_target])
#     template = newLevelTemplate + all_points.shape[0]
#     np.fill_diagonal(template, 0)
#     new_level += template # [8 x 8]
#     l2_trans_mesh.extend(new_level)
#     # compute new points of children (19 new points per refinement)
#     new_points = newPointsTemplate * 0.5 ** (currentLevel - 1) * maxCellSize + all_points[l2_cells[trans_target][0]]
#     all_points = np.concatenate((all_points, new_points), axis = 0)
# l2_trans_mesh = np.array(l2_trans_mesh)
# # trans_voxels2 = [("hexahedron", l2_trans_mesh)]
# # trans_targets2 = meshio.Mesh(points = all_points, cells = trans_voxels2)
# # meshio.write("./trans_targets2.vtk", trans_targets2, file_format="vtk", binary=False)
# # ###################################################
# # #assemble final mesh
# l0_mesh_final = np.delete(level0_mesh_reduced, l0_trans_targets, axis=0)
# l1_mesh_final = np.delete(l1_cells, l1_trans_targets, axis=0)
# l2_mesh_final = np.delete(l2_cells, l2_trans_targets, axis=0)
# refined_mesh = np.concatenate((l0_mesh_final, l1_mesh_final, l2_mesh_final, level3_mesh_reduced, l2_trans_mesh, level4_mesh), axis = 0)
# all_voxels = [("hexahedron", refined_mesh)]
# trans_refinement = meshio.Mesh(points = all_points, cells = all_voxels)
# meshio.write("./trans_refinement2.vtk", trans_refinement, file_format="vtk", binary=False)
# meshio.write("./trans_refinement2.msh", trans_refinement, file_format="gmsh22", binary=False)
# # #################################################################################################
# with open("./refined_mesh.txt", "wb") as fp:   #Pickling
#     pickle.dump(refined_mesh, fp)
# with open("./all_points.txt", "wb") as fp:   #Pickling
#     pickle.dump(all_points, fp)
# # with open("./l1_trans_targets.txt", "rb") as fp:   # Unpickling
# #     l1_trans_targets = pickle.load(fp)
