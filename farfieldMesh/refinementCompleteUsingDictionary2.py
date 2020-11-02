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


triangleFaceArea = [] #new
quadFaceArea = [] #new
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
for i, face in enumerate(surface_quads):
    quadArea = area(surface_points[face], quadNormals[i])
    quadFaceArea.append(quadArea) #new
    edgeSize = quadArea ** 0.5
    r = np.log2(maxCellSize/edgeSize)
    quadTargetLevel.append(r) #new
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
minLevel = min(pointTargetLevel)

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

# print("export base mesh")
# voxel1 = [("hexahedron", z_hexas)]
# mesh = meshio.Mesh(points = l0_points, cells = voxel1)
# meshio.write("./basemesh.vtk", mesh, file_format="vtk", binary=False)
# meshio.write("./basemesh.msh", mesh, file_format="gmsh22", binary=False)


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

###########################################################
# 'target_index' = mesh indicies of cells whose siblings have already been computed
mesh = { 0: {'cells': z_hexas, 'targets': [], 'cut_cells': [], 'target_index': [], 'siblings': [], 'ID': rootIndex[: , None], 'midpoints': []}} #'ID': rootIndex[: , None]

############################################################

#                0:U      1:UF     2:UB     3:UF     4:UR    5:UFL    6:UFR    7:UBL    8:UBR     9:D     10:DF    11:DB    12:DL    13:DR    14:DFL   15:DFR   16:DBL   17:DBR    18:F    19:B     20:L     21:R     22:FL    23:FR    24:BL    25:BR
octreeMap3 = [[[4, 26], [7, 18], [7, 26], [5, 20], [5, 26], [6, 22], [6, 18], [6, 20], [6, 26], [4,  9], [7, 10], [7,  9], [5, 12], [5,  9], [6, 14], [6, 10], [6, 12], [6,  9], [3, 18], [3, 26], [1, 20], [1, 26], [2, 22], [2, 18], [2, 20], [2, 26]],
              [[5, 26], [6, 18], [6, 26], [4, 26], [4, 21], [7, 18], [7, 23], [7, 26], [7, 21], [5,  9], [6, 10], [6,  9], [4,  9], [4, 13], [7, 10], [7, 15], [7,  9], [7, 13], [2, 18], [2, 26], [0, 26], [0, 21], [3, 18], [3, 23], [3, 26], [3, 21]],
              [[6, 26], [5, 26], [5, 19], [7, 26], [7, 21], [4, 26], [4, 21], [4, 19], [4, 25], [6,  9], [5,  9], [5, 11], [7,  9], [7, 13], [4,  9], [4, 13], [4, 11], [4, 17], [1, 26], [1, 19], [3, 26], [3, 21], [0, 26], [0, 21], [0, 19], [0, 25]],
              [[7, 26], [4, 26], [4, 19], [6, 20], [6, 26], [5, 20], [5, 26], [5, 24], [5, 19], [7,  9], [4,  9], [4, 11], [6, 12], [6,  9], [5, 12], [5,  9], [5, 16], [5, 11], [0, 26], [0, 19], [2, 20], [2, 26], [1, 20], [1, 26], [1, 24], [1, 19]],
              [[0,  0], [3,  1], [3,  0], [1,  3], [1,  0], [2,  5], [2,  1], [2,  3], [2,  0], [0, 26], [3, 18], [3, 26], [1, 20], [1, 26], [2, 22], [2, 18], [2, 20], [2, 26], [7, 18], [7, 26], [5, 20], [5, 26], [6, 22], [6, 18], [6, 20], [6, 26]],
              [[1,  0], [2,  1], [2,  0], [0,  0], [0,  4], [3,  1], [3,  6], [3,  0], [3,  4], [1, 26], [2, 18], [2, 26], [0, 26], [0, 21], [3, 18], [3, 23], [3, 26], [3, 21], [6, 18], [6, 26], [4, 26], [4, 21], [7, 18], [7, 23], [7, 26], [7, 21]],
              [[2,  0], [1,  0], [1,  2], [3,  0], [3,  4], [0,  0], [0,  4], [0,  2], [0,  8], [2, 26], [1, 26], [1, 19], [3, 26], [3, 21], [0, 26], [0, 21], [0, 19], [0, 25], [5, 26], [5, 19], [7, 26], [7, 21], [4, 26], [4, 21], [4, 19], [4, 25]],
              [[3,  0], [0,  0], [0,  2], [2,  3], [2,  0], [1,  3], [1,  0], [1,  7], [1,  2], [3, 26], [0, 26], [0, 19], [2, 20], [2, 26], [1, 20], [1, 26], [1, 24], [1, 19], [4, 26], [4, 19], [6, 20], [6, 26], [5, 20], [5, 26], [5, 24], [5, 19]]]

################################################################

U = nDivisions[0] * nDivisions[1]
D = -nDivisions[0] * nDivisions[1]
B = nDivisions[0]
F = -nDivisions[0]
R = 1
L = -1
N = np.array([U, U + F, U + B, U + L, U + R, U + F + L, U + F + R, U + B + L, U + B + R,
              D, D + F, D + B, D + L, D + R, D + F + L, D + F + R, D + B + L, D + B + R,
              F, B, L, R, F + L, F + R, B + L, B + R])
# print(N[[9, 10]])
##############################################

def all_root_neighbours3(target, placeholder = 0):
    neigh = N + target
    if target in TS:
        neigh[[0, 1, 2, 3, 4, 5, 6, 7, 8]] = -1
    else:
        if target in DS:
            neigh[[9, 10, 11, 12, 13, 14, 15, 16, 17]] = -1
    if target in FS:
        neigh[[1, 5, 6, 10, 14, 15, 18, 22, 23]] = -1
    else:
        if target in BS:
            neigh[[2, 7, 8, 11, 16, 17, 19, 24, 25]] = -1
    if target in LS:
        neigh[[3, 5, 7, 12, 14, 16, 20, 22, 24]] = -1
    else:
        if target in RS:
            neigh[[4, 6, 8, 13, 15, 17, 21, 23, 25]] = -1
    return neigh




def intersects_surface(cellIndex):
    # h = maxCellSize / 2 ** (level + 1)
    # h = maxCellSize / 2
    # c = level0_midpoints[cellIndex]
    # AABB = c + h * inflationTemplate
    # AABB = all_points[current_mesh[cellIndex]]
    c = midpoints[cellIndex]
    h = maxCellSize / 2 ** (level + 1)
    AABB = c + h * inflationTemplate * 1.6
    dist = c - surface_points
    dist = np.linalg.norm(dist, axis = 1)[: , None]
    a = np.argmin(dist) # surface mesh index closest to voxel centre
    cut_tris = triFaceIndices[a]
    cut_quads = quadFaceIndices[a]
    if len(cut_tris) > 0:
        for cut_tri in cut_tris:
            if triInBox(AABB, surface_points[surface_triangles[cut_tri]]):
                return True
    elif len(cut_quads) > 0:
        for cut_quad in cut_quads:
            quad = surface_quads[cut_quad]
            tri1 = [quad[0], quad[1], quad[2]]
            tri2 = [quad[1], quad[2], quad[3]]
            if triInBox(AABB, surface_points[tri1]):
                return True
            elif triInBox(AABB, surface_points[tri2]):
                return True
    return False

def all_siblings(target, level):
    sib = []
    octant = target % 8
    parent_index = target // 8
    parent = mesh[level - 1]['targets'][parent_index]
    map = octreeMap3[octant]
    for pos, action in map:
        if action == 26:
            neigh = parent_index * 8 + pos
        else:
            neigh_parent = mesh[level - 1]['siblings'][mesh[level - 1]['target_index'].index(parent)][action]
            neigh = mesh[level - 1]['targets'].index(neigh_parent) * 8 + pos
        sib.extend([neigh])
    # print(sib)
    return sib


newPointsTemplate = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [1, 2, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 1], [1, 1, 1], [2, 1, 1], [0, 2, 1], [1, 2, 1], [2, 2, 1], [1, 0, 2], [0, 1, 2], [1, 1, 2], [2, 1, 2], [1, 2, 2]])

newLevelTemplate = np.array([[0, 0, 2, 1, 5, 6, 9, 8], [0, 0, 3, 2, 6, 7, 10, 9], [2, 3, 0, 4, 9, 10, 13, 12], [1, 2, 4, 0, 8, 9, 12, 11], [5, 6, 9, 8, 0, 14, 16, 15], [6, 7, 10, 9, 14, 0, 17, 16], [9, 10, 13, 12, 16, 17, 0, 18], [8, 9, 12, 11, 15, 16, 18, 0]])

inflationTemplate = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])

all_sib_func = {'root': all_root_neighbours3, 'oct': all_siblings}

###########################################################
all_points = l0_points.copy()

for level in range(maxLevel):
    if level == 0:
        neigh_func = 'root'
    else:
        neigh_func = 'oct'

    current_mesh = mesh[level]['cells']
    midpoints = all_points[current_mesh[:, 0]] + maxCellSize * 0.5 ** (level + 1)
    mesh[level]['midpoints'] = midpoints
    distToMidpoint = np.sum((surface_points[0] - midpoints) ** 2, axis = 1)
    start_cell = np.argmin(distToMidpoint)
    mesh[level]["cut_cells"].append(start_cell)
    cut_cell_checked = []
    cut_cell_checked.append(start_cell)
    neighbours_tmp = all_sib_func[neigh_func](start_cell, level)
    mesh[level]["target_index"].append(start_cell)
    mesh[level]["siblings"].append(neighbours_tmp)
    while True:
        cut_cell_tmp = []
        for cell in neighbours_tmp:
            if intersects_surface(cell):
                cut_cell_tmp.append(cell)
        if len(cut_cell_tmp) == 0:
            break
        cut_cell_checked.extend(neighbours_tmp)
        mesh[level]["cut_cells"].extend(cut_cell_tmp)
        neighbours_tmp = []
        for cell in cut_cell_tmp:
            cell_neigh = all_sib_func[neigh_func](cell, level)
            mesh[level]["target_index"].append(cell)
            mesh[level]["siblings"].append(cell_neigh)
            neighbours_tmp.extend(cell_neigh)
        neighbours_tmp = list(set(neighbours_tmp))
        neighbours_tmp = [x for x in neighbours_tmp if x not in cut_cell_checked]

    # if level == maxLevel:
    #     break

    rInf = transLayers * maxCellSize * 0.5 ** (level)
    surface_target_mask = np.array(np.where(pointTargetLevel > level)).ravel()
    targets_checked = mesh[level]["cut_cells"].copy()
    for cell in targets_checked:
        midpoint = midpoints[cell]
        distToMidpoint = np.sum((surface_points[surface_target_mask] - midpoint) ** 2, axis = 1)
        test_surface_point = surface_points[surface_target_mask[np.argmin(distToMidpoint)]]
        hex = np.tile(midpoint, (8, 1)) + rInf * inflationTemplate
        if hex_contains(hex, test_surface_point):
            mesh[level]["targets"].extend([cell])

    neighbours_tmp = []
    for cell in mesh[level]["targets"]:
        cell_neigh = mesh[level]["siblings"][mesh[level]["target_index"].index(cell)]
        neighbours_tmp.extend(cell_neigh)
    neighbours_tmp = list(set(neighbours_tmp))
    neighbours_tmp = [x for x in neighbours_tmp if x not in targets_checked]

    while True:
        targets_tmp = []
        for cell in neighbours_tmp:
            midpoint = midpoints[cell]
            distToMidpoint = np.sum((surface_points[surface_target_mask] - midpoint) ** 2, axis = 1)
            test_surface_point = surface_points[surface_target_mask[np.argmin(distToMidpoint)]]
            hex = np.tile(midpoint, (8, 1)) + rInf * inflationTemplate
            if hex_contains(hex, test_surface_point):
                targets_tmp.extend([cell])
        if len(targets_tmp) == 0:
            break
        # targets.extend(targets_tmp)
        mesh[level]["targets"].extend(targets_tmp)
        targets_checked.extend(neighbours_tmp)
        neighbours_tmp = []
        for cell in targets_tmp:
            cell_neigh = all_sib_func[neigh_func](cell, level)
            mesh[level]["target_index"].append(cell)
            mesh[level]["siblings"].append(cell_neigh)
            neighbours_tmp.extend(cell_neigh)
        neighbours_tmp = list(set(neighbours_tmp))
        neighbours_tmp = [x for x in neighbours_tmp if x not in targets_checked]

    mesh[level + 1] = {'cells': [], 'targets': [], 'cut_cells': [], 'target_index': [], 'siblings': [], 'ID': [], 'midpoints': []}
    next_level_mesh = []
    for target in mesh[level]["targets"]:
        # split level 0 target cell into 8 children and add children to level1_mesh:
        cell = current_mesh[target]
        # print(cell)
        # mesh[level + 1]['ID'].extend([[target, 0], [target, 1], [target, 2], [target, 3], [target, 4], [target, 5], [target, 6], [target, 7]])
        children = np.diag(cell)
        template = newLevelTemplate + all_points.shape[0]
        np.fill_diagonal(template, 0)
        children += template
        next_level_mesh.extend(children)
        # compute coordinates of children cells:
        chidlren_coords = newPointsTemplate * 0.5 ** (level + 1) * maxCellSize + all_points[cell[0]]
        all_points = np.concatenate((all_points, chidlren_coords), axis = 0)
    mesh[level + 1]['cells'] = np.array(next_level_mesh)

    # if level > 0:
    for target in mesh[level]["targets"]:
        x = list(mesh[level]['ID'][target])
        # print("x")
        # print(x)
        # x.append(0)
        # print(x)
        a0 = x + [0]
        # print(a0)
        mesh[level + 1]['ID'].append(a0)
        a1 = x + [1]
        mesh[level + 1]['ID'].append(a1)
        a2 = x + [2]
        mesh[level + 1]['ID'].append(a2)
        a3 = x + [3]
        mesh[level + 1]['ID'].append(a3)
        a4 = x + [4]
        mesh[level + 1]['ID'].append(a4)
        a5 = x + [5]
        mesh[level + 1]['ID'].append(a5)
        a6 = x + [6]
        mesh[level + 1]['ID'].append(a6)
        a7 = x + [7]
        mesh[level + 1]['ID'].append(a7)


##############################################
# assemble mesh
refined_mesh = np.concatenate((np.delete(mesh[0]["cells"], mesh[0]["targets"], axis = 0), np.delete(mesh[1]["cells"], mesh[1]["targets"], axis = 0), np.delete(mesh[2]["cells"], mesh[2]["targets"], axis = 0), np.delete(mesh[3]["cells"], mesh[3]["targets"], axis = 0), mesh[4]["cells"]), axis = 0)
# voxel1 = [("hexahedron", refined_mesh)]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./refined_mesh.vtk", meshF, file_format="vtk", binary=False)
#########################################
# voxel1 = [("hexahedron", mesh[2]["cells"][[testCell]])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./test1.vtk", meshF, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", mesh[2]["cells"][x])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./test2.vtk", meshF, file_format="vtk", binary=False)
#########################################################

#########################################################
# classify mesh
cut_mesh3 = mesh[3]['cells'][[x for x in mesh[3]["cut_cells"] if x not in mesh[3]["targets"]]]
# print(cut_mesh3)
level = 4
current_mesh = mesh[level]['cells']
midpoints = all_points[current_mesh[:, 0]] + maxCellSize * 0.5 ** (level + 1)
mesh[level]['midpoints'] = midpoints
for cell in range(current_mesh.shape[0]):
    if intersects_surface(cell):
        mesh[4]['cut_cells'].append(cell)
cut_mesh4 = mesh[4]['cells'][mesh[4]['cut_cells']]
# print(cut_mesh4)
cut_mesh = np.concatenate((cut_mesh3, cut_mesh4), axis = 0)
# voxel1 = [("hexahedron", cut_mesh)]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./cut_mesh.vtk", meshF, file_format="vtk", binary=False)
# print(cut_mesh.shape)
# test = []
# for cell in refined_mesh:
#     if cell in cut_mesh:
#         test.append(cell)
# print(test)
# print(len(test))

start_cell = 1
# voxel1 = [("hexahedron", refined_mesh[[start_cell]])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./test1.vtk", meshF, file_format="vtk", binary=False)

# def ref_mesh_neighbours(cell):

# def all_sibling2(target, level):
#     sib = []
#     octant = target % 8
#     parent_index = target // 8
#     parent = mesh[level - 1]['targets'][parent_index]
#     map = octreeMap3[octant]
#     for pos, action in map:
#         if action == 26:
#             neigh = parent_index * 8 + pos
#         else:
#             # neigh_parent = mesh[level - 1]['siblings'][mesh[level - 1]['target_index'].index(parent)][action]
#             neigh_parent = mesh[level - 1].index
#             neigh = mesh[level - 1]['targets'].index(neigh_parent) * 8 + pos
#         sib.extend([neigh])
#     # print(sib)
#     return sib
# print("target", mesh[4]['ID'][0])
# print(mesh[4]['ID'].index(mesh[4]['ID'][0]))

def sibling_ref_mesh_using_ID(targetID, direction):
    # sib = []
    neighbourID = targetID.copy()
    action = direction
    for i in range(1, len(targetID)):
        j = -i
        octant = targetID[j]
        map = octreeMap3[octant][action]
        neighbourID[j] = map[0]
        action = map[1]
        if action == 26:
            break
    pos = mesh[0]['targets'].index(neighbourID[0]) * 8 + neighbourID[1]
    for i in range(1, len(neighbourID) - 1):
        print(i)
        print(pos)
        pos = mesh[i]['targets'].index(pos) * 8 + neighbourID[i + 1]
    return pos
    # parent_index = target // 8
    # parent = mesh[level - 1]['targets'][parent_index]
    # map = octreeMap3[octant]
    # for pos, action in map:
    #     if action == 26:
    #         neigh = parent_index * 8 + pos
    #     else:
    #         neigh_parent = mesh[level - 1]['siblings'][mesh[level - 1]['target_index'].index(parent)][action]
    #         neigh = mesh[level - 1]['targets'].index(neigh_parent) * 8 + pos
    #     sib.extend([neigh])
    # # print(sib)
    # return sib

print("upper neighbour", sibling_ref_mesh([454, 2, 4, 5, 0], 0))
# neigh0 = sibling_ref_mesh([454, 2, 4, 5, 4], 0)
# print(neigh0)
# print(mesh[4]['ID'].index(neigh0))
# voxel1 = [("hexahedron", mesh[4]["cells"][[0]])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./test1.vtk", meshF, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", mesh[4]["cells"][[8]])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./test3.vtk", meshF, file_format="vtk", binary=False)
# print(mesh[4]['ID'].index([454, 2, 4, 5, 0]))
# print(mesh[4]['ID'].index([454, 2, 4, 5, 1]))
# print(mesh[4]['ID'].index([454, 2, 4, 5, 2]))
# print(mesh[4]['ID'].index([454, 2, 4, 5, 3]))
# print(mesh[4]['ID'].index([454, 2, 4, 5, 4]))
# voxel1 = [("hexahedron", mesh[4]["cells"][[0]])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./oct0.vtk", meshF, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", mesh[4]["cells"][[1]])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./oct1.vtk", meshF, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", mesh[4]["cells"][[2]])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./oct2.vtk", meshF, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", mesh[4]["cells"][[3]])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./oct3.vtk", meshF, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", mesh[4]["cells"][[4]])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./oct4.vtk", meshF, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", mesh[4]["cells"][[5]])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./oct5.vtk", meshF, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", mesh[4]["cells"][[6]])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./oct6.vtk", meshF, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", mesh[4]["cells"][[7]])]
# meshF = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./oct7.vtk", meshF, file_format="vtk", binary=False)
# print("target", mesh[4]['ID'])
