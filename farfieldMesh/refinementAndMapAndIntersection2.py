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


############################################################
octreeMap = {'U': [[4, "H"], [5, "H"], [6, "H"], [7, "H"], [0, "U"], [1, "U"], [2, "U"], [3, "U"]],
             'B': [[3, "H"], [2, "H"], [1, "B"], [0, "B"], [7, "H"], [6, "H"], [5, "B"], [4, "B"]],
             'R': [[1, "H"], [0, "R"], [3, "R"], [2, "H"], [5, "H"], [4, "R"], [7, "R"], [6, "H"]],
             'D': [[4, "D"], [5, "D"], [6, "D"], [7, "D"], [0, "H"], [1, "H"], [2, "H"], [3, "H"]],
             'F': [[3, "F"], [2, "F"], [1, "H"], [0, "H"], [7, "F"], [6, "F"], [5, "H"], [4, "H"]],
             'L': [[1, "L"], [0, "H"], [3, "H"], [2, "L"], [5, "L"], [4, "H"], [7, "H"], [6, "L"]]}
octreeMap2 = [[[4, "H"], [3, "H"], [1, "H"], [4, "D"], [3, "F"], [1, "L"]],
              [[5, "H"], [2, "H"], [0, "R"], [5, "D"], [2, "F"], [0, "H"]],
              [[6, "H"], [1, "B"], [3, "R"], [6, "D"], [1, "H"], [3, "H"]],
              [[7, "H"], [0, "B"], [2, "H"], [7, "D"], [0, "H"], [2, "L"]],
              [[0, "U"], [7, "H"], [5, "H"], [0, "H"], [7, "F"], [5, "L"]],
              [[1, "U"], [6, "H"], [4, "R"], [1, "H"], [6, "F"], [4, "H"]],
              [[2, "U"], [5, "B"], [7, "R"], [2, "H"], [5, "H"], [7, "H"]],
              [[3, "U"], [4, "B"], [6, "H"], [3, "H"], [4, "H"], [6, "L"]]]


def root_neighbour(target_root, direction):
    # target_root: index of root voxel within rootIndex
    # direction: {"U", "B", "R", "D", "F", "L"}
    # returns: index of neighbour within rootIndex as interger
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

def all_root_neighbours(target_root):
    # target_root: index of root voxel within rootIndex
    # returns: list; contains indices of all neighbours within rootIndex
    neighbours = [-1, -1, -1, -1, -1, -1]
    if target_root not in TS:
        # neighbours.append(target_root + nDivisions[0] * nDivisions[1]) # upper neighbour
        neighbours[0] = target_root + nDivisions[0] * nDivisions[1]
    if target_root not in BS:
        neighbours[1] = target_root + nDivisions[0] # back neighbour
    if target_root not in RS:
        neighbours[2] = target_root + 1 # right neighbour
    if target_root not in DS:
        neighbours[3] = target_root - nDivisions[0] * nDivisions[1] # down neighbour
    if target_root not in FS:
        neighbours[4] = target_root - nDivisions[0] # front neighbour
    if target_root not in LS:
        neighbours[5] = target_root - 1 # left neighbour
    return neighbours

def level1_neighbour(target, direction):
    # target: cell index within level1_mesh
    # direction: {"U", "B", "R", "D", "F", "L"}
    # return neighbour index within level1_mesh
    target_octant = target % 8
    target_parent = level0_targets[target // 8]
    map = octreeMap[direction][target_octant]
    neighbour_octant = map[0]
    if map[1] == "H":
        return np.argwhere(level0_targets == target_parent).ravel() * 8 + neighbour_octant
    else:
        neighbour_parent = root_neighbour(target_parent, map[1])
        if neighbour_parent == -1:
            return -1
        return np.argwhere(level0_targets == neighbour_parent).ravel() * 8 + neighbour_octant


def all_level1_neighbours(target):
    # target: cell index within level1_mesh
    # return level 1 neighbours in all orthogonal directions {"U", "B", "R", "D", "F", "L"}
    target_octant = target % 8
    target_parent = level0_targets[target // 8]
    map = octreeMap2[target_octant]
    neighs = []
    for neighbour_octant, direction in map:
        if direction == "H":
            neighs.extend(np.argwhere(level0_targets == target_parent).ravel() * 8 + neighbour_octant)
        else:
            neigh_parent = root_neighbour(target_parent, direction)
            if neigh_parent == -1:
                neighs.extend(-1)
            else:
                neighs.extend(np.argwhere(level0_targets == neigh_parent).ravel() * 8 + neighbour_octant)
    return neighs

def level2_neighbour(target, direction):
    # target: cell index within level1_mesh
    # direction: {"U", "B", "R", "D", "F", "L"}
    # return neighbour index within level2_mesh
    target_octant = target % 8
    target_parent = level1_targets[target // 8]
    map = octreeMap[direction][target_octant]
    neighbour_octant = map[0]
    if map[1] == "H":
        return np.argwhere(level1_targets == target_parent).ravel() * 8 + neighbour_octant
    else:
        neighbour_parent = level1_neighbour(target_parent, map[1])
        if neighbour_parent == -1:
            return -1
        return np.argwhere(level1_targets == neighbour_parent).ravel() * 8 + neighbour_octant

def all_level2_neighbours(target):
    # target: cell index within level1_mesh
    # return level 2 neighbours in all orthogonal directions {"U", "B", "R", "D", "F", "L"}
    target_octant = target % 8
    target_parent = level1_targets[target // 8]
    map = octreeMap2[target_octant]
    neighs = []
    for neighbour_octant, direction in map:
        if direction == "H":
            neighs.extend(np.argwhere(level1_targets == target_parent).ravel() * 8 + neighbour_octant)
        else:
            neigh_parent = level1_neighbour(target_parent, direction)
            if neigh_parent == -1:
                neighs.extend(-1)
            else:
                neighs.extend(np.argwhere(level1_targets == neigh_parent).ravel() * 8 + neighbour_octant)
    return neighs

def level3_neighbour(target, direction):
    # target: cell index within level1_mesh
    # direction: {"U", "B", "R", "D", "F", "L"}
    # return neighbour index within level3_mesh
    target_octant = target % 8
    target_parent = level2_targets[target // 8]
    map = octreeMap[direction][target_octant]
    neighbour_octant = map[0]
    if map[1] == "H":
        return np.argwhere(level2_targets == target_parent).ravel() * 8 + neighbour_octant
    else:
        neighbour_parent = level2_neighbour(target_parent, map[1])
        if neighbour_parent == -1:
            return -1
        return np.argwhere(level2_targets == neighbour_parent).ravel() * 8 + neighbour_octant

def all_level3_neighbours(target):
    # target: cell index within level1_mesh
    # return level 3 neighbours in all orthogonal directions {"U", "B", "R", "D", "F", "L"}
    target_octant = target % 8
    target_parent = level2_targets[target // 8]
    map = octreeMap2[target_octant]
    neighs = []
    for neighbour_octant, direction in map:
        if direction == "H":
            neighs.extend(np.argwhere(level2_targets == target_parent).ravel() * 8 + neighbour_octant)
        else:
            neigh_parent = level2_neighbour(target_parent, direction)
            if neigh_parent == -1:
                neighs.extend(-1)
            else:
                neighs.extend(np.argwhere(level2_targets == neigh_parent).ravel() * 8 + neighbour_octant)
    return neighs

def level4_neighbour(target, direction):
    # target: cell index within level1_mesh
    # direction: {"U", "B", "R", "D", "F", "L"}
    # return neighbour index within level3_mesh
    target_octant = target % 8
    target_parent = level3_targets[target // 8]
    map = octreeMap[direction][target_octant]
    neighbour_octant = map[0]
    if map[1] == "H":
        return np.argwhere(level3_targets == target_parent).ravel() * 8 + neighbour_octant
    else:
        neighbour_parent = level3_neighbour(target_parent, map[1])
        if neighbour_parent == -1:
            return -1
        return np.argwhere(level3_targets == neighbour_parent).ravel() * 8 + neighbour_octant

def all_level4_neighbours(target):
    # target: cell index within level1_mesh
    # return level 3 neighbours in all orthogonal directions {"U", "B", "R", "D", "F", "L"}
    target_octant = target % 8
    target_parent = level3_targets[target // 8]
    map = octreeMap2[target_octant]
    neighs = []
    for neighbour_octant, direction in map:
        if direction == "H":
            neighs.extend(np.argwhere(level3_targets == target_parent).ravel() * 8 + neighbour_octant)
        else:
            neigh_parent = level3_neighbour(target_parent, direction)
            if neigh_parent == -1:
                neighs.extend(-1)
            else:
                neighs.extend(np.argwhere(level3_targets == neigh_parent).ravel() * 8 + neighbour_octant)
    return neighs
#################################################

def level0_intersects_surface(cellIndex):
    # h = maxCellSize / 2 ** (level + 1)
    # h = maxCellSize / 2
    # c = level0_midpoints[cellIndex]
    # AABB = c + h * inflationTemplate
    AABB = all_points[level0_mesh[cellIndex]]
    c = level0_midpoints[cellIndex]
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

def level1_intersects_surface(cellIndex):
    # h = maxCellSize / 2 ** (level + 1)
    # h = maxCellSize / 2
    # c = level0_midpoints[cellIndex]
    # AABB = c + h * inflationTemplate
    AABB = all_points[level1_mesh[cellIndex]]
    c = level1_midpoints[cellIndex]
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

def level2_intersects_surface(cellIndex):
    # h = maxCellSize / 2 ** (level + 1)
    # h = maxCellSize / 2
    # c = level0_midpoints[cellIndex]
    # AABB = c + h * inflationTemplate
    AABB = all_points[level2_mesh[cellIndex]]
    c = level2_midpoints[cellIndex]
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

def level3_intersects_surface(cellIndex):
    # h = maxCellSize / 2 ** (level + 1)
    # h = maxCellSize / 2
    # c = level0_midpoints[cellIndex]
    # AABB = c + h * inflationTemplate
    AABB = all_points[level3_mesh[cellIndex]]
    c = level3_midpoints[cellIndex]
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

def level4_intersects_surface(cellIndex):
    h = maxCellSize / 2 ** (level + 1)
    # h = maxCellSize / 2
    # c = level0_midpoints[cellIndex]
    # AABB = c + h * inflationTemplate
    # AABB = all_points[level4_mesh[cellIndex]]
    c = level4_midpoints[cellIndex]
    AABB = c + h * 1.4 * inflationTemplate
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

#############################################################

# level 0 refinement
# compute influence radius
# compute coordinates of midpoints of each level 0 cell
# locate starting cell (level0 cell that needs refinement):
#       create mask of surface points whose level is above 0
#       select random point from mask
#       locate start cell (cell whose midpoint is closest to seleceted surface point)
# add start cell to list of target cells
# add start cell to list of cells whihc have already been checked for whether they need refinement (level0_checked)
# locate neighbours of start cell (root_neighbours)
# initialize list of temporary refinement targets (tempTargets)
### begin_loop ####
# for each neighbour in root_neighbours
#   retrieve midpoint of neighbour
#   locate surface point whose target is above 0 (i.e. from list of mask) closest to midpoint of neighbour
#   inflate neighbour by influence radius
#   if inflated neighbour contains the surface point
#       add neighbour to list of temporary targets
# if tempTargets is empty, then all refinement targets have been identified and search can stop
# else, add tempTargets to list of target cells
# add root_neighbours to level0_checked so they won't needlessly get assessed again for refinement
# reinitialze list of root_neighbours, (root_neighbours = [])
# remove all entries from root_neighbours which have already been assessed (using level0_checked)
# re-initialize list of temporary targets (tempTargets = [])
# restart process from begin_loop
# break out of loop when list of temporary targets is empty
# remove selected targets from list of non-targets
# refine targets: add refined cells to level1_mesh and append new children coordinates to all_points

newPointsTemplate = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [1, 2, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 1], [1, 1, 1], [2, 1, 1], [0, 2, 1], [1, 2, 1], [2, 2, 1], [1, 0, 2], [0, 1, 2], [1, 1, 2], [2, 1, 2], [1, 2, 2]])

newLevelTemplate = np.array([[0, 0, 2, 1, 5, 6, 9, 8], [0, 0, 3, 2, 6, 7, 10, 9], [2, 3, 0, 4, 9, 10, 13, 12], [1, 2, 4, 0, 8, 9, 12, 11], [5, 6, 9, 8, 0, 14, 16, 15], [6, 7, 10, 9, 14, 0, 17, 16], [9, 10, 13, 12, 16, 17, 0, 18], [8, 9, 12, 11, 15, 16, 18, 0]])

inflationTemplate = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])

level = 0
rInf = transLayers * maxCellSize * 0.5 ** (level)
all_points = l0_points.copy()
level0_mesh = z_hexas.copy()

level0_targets = [] # refinement targets
level0_non_targets = rootIndex.copy() # list of cells to be left alone
level0_checked = [] # level 0 cells which have already been tested whether they need refinement

surface_0_plus_mask = np.array(np.where(pointTargetLevel > level)).ravel() # surface point indices whose target level is above zero

level0_midpoints = all_points[level0_mesh[:, 0]] + maxCellSize * 0.5 ** (level + 1) # coordinates of level0 midpoints
distToMidpoint = np.sum((surface_points[surface_0_plus_mask[0]] - level0_midpoints) ** 2, axis = 1) ** 0.5 # list of distances between level0_midpoints and random point on surface whose target level is above 0
start_cell = np.argmin(distToMidpoint) # index of cell closest to selceted point on surface

level0_targets.append(start_cell) # add start_cell to list of targets
# level0_non_targets = np.delete(level0_non_targets, start_cell)

level0_checked.append(start_cell) # add targets to list of cells that needn't be checked again

root_neighbours = all_root_neighbours(start_cell) # neighbours (in ortho-directions) of start_cell

###################################################################
# strategy for identifying intersecting cells:
# start with start_cell (which should be intersecting)
# add start_cell to level0_intersecting
# add start_cell to level0_inter_checked
# get neighbours of start_cell
# add neighbours to level0_inter_neighbours
# begin loop
# for each neighbour in level0_inter_neighbours:
#   check if neighbour is intersecting
#   if yes:
#       add neighbour to level0_intersecting_tmp
# if level0_intersecting_tmp is empty:
#   stop search, all intersecting cells have been identified
# else
#   add level0_intersecting_tmp to level0_intersecting
# add level0_inter_neighbours to level0_inter_checked
# reset level0_inter_neighbours
# for each entry in level0_intersecting_tmp:
#   get neighbour cells of each entry and add to level0_inter_neighbours
# remove all duplicates from level0_inter_neighbours
# remove all entries from level0_inter_neighbours that are present inside level0_inter_checked
# reset level0_intersecting_tmp = []
# restart loop

level0_intersecting = [] # all level0_mesh indices intersecting the surface
level0_inter_checked = [] # all level0 cell indices that have been checked for intersection
level0_inter_neighbours = []
level0_intersecting_tmp = []

level0_intersecting.append(start_cell)
level0_inter_checked.append(start_cell)
level0_inter_neighbours = all_root_neighbours(start_cell)
#begin "while True"-loop
while True:
    level0_intersecting_tmp = []
    for cell in level0_inter_neighbours:
        if level0_intersects_surface(cell):
            level0_intersecting_tmp.append(cell)
    if len(level0_intersecting_tmp) == 0:
        print("all level 0 intersecting cells have been found")
        break
    level0_intersecting.extend(level0_intersecting_tmp)
    level0_inter_checked.extend(level0_inter_neighbours)
    level0_inter_neighbours = []
    for cell in level0_intersecting_tmp:
        level0_inter_neighbours.extend(all_root_neighbours(cell))
    level0_inter_neighbours = list(set(level0_inter_neighbours))
    level0_inter_neighbours = [x for x in level0_inter_neighbours if x not in level0_inter_checked]

# voxel1 = [("hexahedron", level0_mesh[[start_cell]])]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./inter_l0_start.vtk", mesh, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", level0_mesh[level0_inter_neighbours])]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./inter_l0_start_neigh.vtk", mesh, file_format="vtk", binary=False)

##################################################################


# begin loop for identifying targets for refinement:
while True:
    tempTargets = []
    for cellIndex in root_neighbours:
        midpoint = level0_midpoints[cellIndex] # midpoint of neighbour
        distToMidpoint = np.sum((surface_points[surface_0_plus_mask] - midpoint) ** 2, axis = 1) ** 0.5 # distance between midpoint and surface_points whose level is above 0
        test_surface_point = surface_points[surface_0_plus_mask[np.argmin(distToMidpoint)]] # coordinates of surface point closest to midpoint
        hex = np.tile(midpoint, (8, 1)) + rInf * inflationTemplate # coordinates of inflated neighbour
        if hex_contains(hex, test_surface_point): # condition for whether neighbour needs to be refined
            tempTargets.append(cellIndex)
    if len(tempTargets) == 0:
        break
    else:
        level0_targets.extend(tempTargets)
    level0_checked.extend(root_neighbours)
    root_neighbours = []
    for cellIndex in tempTargets:
        root_neighbours.extend(all_root_neighbours(cellIndex))
    root_neighbours = list(set(root_neighbours))
    root_neighbours = [x for x in root_neighbours if x not in level0_checked]
level0_non_targets = np.delete(level0_non_targets, level0_targets)

# refine seleceted targets from level 0 to level 1
level1_mesh = []
for targetCell in level0_mesh[level0_targets]:
    # split level 0 target cell into 8 children and add children to level1_mesh:
    children = np.diag(targetCell)
    template = newLevelTemplate + all_points.shape[0]
    np.fill_diagonal(template, 0)
    children += template
    level1_mesh.extend(children)
    # compute coordinates of children cells:
    chidlren_coords = newPointsTemplate * 0.5 ** (level + 1) * maxCellSize + all_points[targetCell[0]]
    all_points = np.concatenate((all_points, chidlren_coords), axis = 0)
level1_mesh = np.array(level1_mesh)

# print("export level 1 mesh")
# voxel1 = [("hexahedron", level1_mesh)]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./l1_mesh.vtk", mesh, file_format="vtk", binary=False)

level += 1 # level = 1
rInf = transLayers * maxCellSize * 0.5 ** (level)
level1_targets = []
level1_non_targets = np.arange(level1_mesh.shape[0])
level1_checked = []
surface_1_plus_mask = np.array(np.where(pointTargetLevel > level)).ravel()
level1_midpoints = all_points[level1_mesh[:, 0]] + maxCellSize * 0.5 ** (level + 1)
distToMidpoint = np.sum((surface_points[surface_1_plus_mask[0]] - level1_midpoints) ** 2, axis = 1) ** 0.5 # list of distances between level0_midpoints and random point on surface whose target level is above 0
start_cell = np.argmin(distToMidpoint) # index of cell closest to selceted point on surface
level1_targets.append(start_cell)
level1_checked.append(start_cell) # add targets to list of cells that needn't be checked again


level1_neighbours = all_level1_neighbours(start_cell)

#####################################################
level1_intersecting = [] # all level0_mesh indices intersecting the surface
level1_inter_checked = [] # all level0 cell indices that have been checked for intersection
level1_inter_neighbours = []
level1_intersecting_tmp = []

level1_intersecting.append(start_cell)
level1_inter_checked.append(start_cell)
level1_inter_neighbours = all_level1_neighbours(start_cell)
#begin "while True"-loop
while True:
    level1_intersecting_tmp = []
    for cell in level1_inter_neighbours:
        if level1_intersects_surface(cell):
            level1_intersecting_tmp.append(cell)
    if len(level1_intersecting_tmp) == 0:
        print("all level 1 intersecting cells have been found")
        break
    level1_intersecting.extend(level1_intersecting_tmp)
    level1_inter_checked.extend(level1_inter_neighbours)
    level1_inter_neighbours = []
    for cell in level1_intersecting_tmp:
        level1_inter_neighbours.extend(all_level1_neighbours(cell))
    level1_inter_neighbours = list(set(level1_inter_neighbours))
    level1_inter_neighbours = [x for x in level1_inter_neighbours if x not in level1_inter_checked]
# print(level1_intersecting)
# voxel1 = [("hexahedron", level1_mesh[level1_intersecting])]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./inter_l1.vtk", mesh, file_format="vtk", binary=False)
###################################################

while True:
    tempTargets = []
    for cellIndex in level1_neighbours:
        midpoint = level1_midpoints[cellIndex] # midpoint of neighbour
        distToMidpoint = np.sum((surface_points[surface_1_plus_mask] - midpoint) ** 2, axis = 1) ** 0.5 # distance between midpoint and surface_points whose level is above 0
        test_surface_point = surface_points[surface_1_plus_mask[np.argmin(distToMidpoint)]] # coordinates of surface point closest to midpoint
        hex = np.tile(midpoint, (8, 1)) + rInf * inflationTemplate # coordinates of inflated neighbour
        if hex_contains(hex, test_surface_point): # condition for whether neighbour needs to be refined
            tempTargets.append(cellIndex)
    if len(tempTargets) == 0:
        break
    else:
        level1_targets.extend(tempTargets)
    level1_checked.extend(level1_neighbours)
    level1_neighbours = []
    for cellIndex in tempTargets:
        level1_neighbours.extend(all_level1_neighbours(cellIndex))
    level1_neighbours = list(set(level1_neighbours))
    level1_neighbours = [x for x in level1_neighbours if x not in level1_checked]
level1_non_targets = np.delete(level1_non_targets, level1_targets)

# refine seleceted targets from level 1 to level 2
level2_mesh = []
for targetCell in level1_mesh[level1_targets]:
    # split level 1 target cell into 8 children and add children to level2_mesh:
    children = np.diag(targetCell)
    template = newLevelTemplate + all_points.shape[0]
    np.fill_diagonal(template, 0)
    children += template
    level2_mesh.extend(children)
    # compute coordinates of children cells:
    chidlren_coords = newPointsTemplate * 0.5 ** (level + 1) * maxCellSize + all_points[targetCell[0]]
    all_points = np.concatenate((all_points, chidlren_coords), axis = 0)
level2_mesh = np.array(level2_mesh)

# print("export level 2 mesh")
# voxel1 = [("hexahedron", level2_mesh)]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./l2_mesh.vtk", mesh, file_format="vtk", binary=False)

level += 1 # level = 2
rInf = transLayers * maxCellSize * 0.5 ** (level)
level2_targets = []
level2_non_targets = np.arange(level2_mesh.shape[0])
level2_checked = []
surface_2_plus_mask = np.array(np.where(pointTargetLevel > level)).ravel()
level2_midpoints = all_points[level2_mesh[:, 0]] + maxCellSize * 0.5 ** (level + 1)
distToMidpoint = np.sum((surface_points[surface_2_plus_mask[0]] - level2_midpoints) ** 2, axis = 1) ** 0.5 # list of distances between level0_midpoints and random point on surface whose target level is above 0
start_cell = np.argmin(distToMidpoint) # index of cell closest to selceted point on surface
level2_targets.append(start_cell)
level2_checked.append(start_cell) # add targets to list of cells that needn't be checked again

level2_neighbours = all_level2_neighbours(start_cell)

#####################################################
level2_intersecting = [] # all level0_mesh indices intersecting the surface
level2_inter_checked = [] # all level0 cell indices that have been checked for intersection
level2_inter_neighbours = []
level2_intersecting_tmp = []

level2_intersecting.append(start_cell)
level2_inter_checked.append(start_cell)
level2_inter_neighbours = all_level2_neighbours(start_cell)
#begin "while True"-loop
while True:
    level2_intersecting_tmp = []
    for cell in level2_inter_neighbours:
        if level2_intersects_surface(cell):
            level2_intersecting_tmp.append(cell)
    if len(level2_intersecting_tmp) == 0:
        print("all level 2 intersecting cells have been found")
        break
    level2_intersecting.extend(level2_intersecting_tmp)
    level2_inter_checked.extend(level2_inter_neighbours)
    level2_inter_neighbours = []
    for cell in level2_intersecting_tmp:
        level2_inter_neighbours.extend(all_level2_neighbours(cell))
    level2_inter_neighbours = list(set(level2_inter_neighbours))
    level2_inter_neighbours = [x for x in level2_inter_neighbours if x not in level2_inter_checked]
# print(level2_intersecting)
# voxel1 = [("hexahedron", level2_mesh[level2_intersecting])]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./inter_l2.vtk", mesh, file_format="vtk", binary=False)
###################################################

while True:
    tempTargets = []
    for cellIndex in level2_neighbours:
        midpoint = level2_midpoints[cellIndex] # midpoint of neighbour
        distToMidpoint = np.sum((surface_points[surface_2_plus_mask] - midpoint) ** 2, axis = 1) ** 0.5 # distance between midpoint and surface_points whose level is above 0
        test_surface_point = surface_points[surface_2_plus_mask[np.argmin(distToMidpoint)]] # coordinates of surface point closest to midpoint
        hex = np.tile(midpoint, (8, 1)) + rInf * inflationTemplate # coordinates of inflated neighbour
        if hex_contains(hex, test_surface_point): # condition for whether neighbour needs to be refined
            tempTargets.append(cellIndex)
    if len(tempTargets) == 0:
        break
    else:
        level2_targets.extend(tempTargets)
    level2_checked.extend(level2_neighbours)
    level2_neighbours = []
    for cellIndex in tempTargets:
        level2_neighbours.extend(all_level2_neighbours(cellIndex))
    level2_neighbours = list(set(level2_neighbours))
    level2_neighbours = [x for x in level2_neighbours if x not in level2_checked]
level2_non_targets = np.delete(level2_non_targets, level2_targets)

# refine seleceted targets from level 2 to level 3
level3_mesh = []
for targetCell in level2_mesh[level2_targets]:
    # split level 2 target cell into 8 children and add children to level3_mesh:
    children = np.diag(targetCell)
    template = newLevelTemplate + all_points.shape[0]
    np.fill_diagonal(template, 0)
    children += template
    level3_mesh.extend(children)
    # compute coordinates of children cells:
    chidlren_coords = newPointsTemplate * 0.5 ** (level + 1) * maxCellSize + all_points[targetCell[0]]
    all_points = np.concatenate((all_points, chidlren_coords), axis = 0)
level3_mesh = np.array(level3_mesh)

# print("export level 3 mesh")
# voxel1 = [("hexahedron", level3_mesh)]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./l3_mesh.vtk", mesh, file_format="vtk", binary=False)


level += 1 # level = 3
rInf = transLayers * maxCellSize * 0.5 ** (level)
level3_targets = []
level3_non_targets = np.arange(level3_mesh.shape[0])
level3_checked = []
surface_3_plus_mask = np.array(np.where(pointTargetLevel > level)).ravel()
level3_midpoints = all_points[level3_mesh[:, 0]] + maxCellSize * 0.5 ** (level + 1)
distToMidpoint = np.sum((surface_points[surface_3_plus_mask[0]] - level3_midpoints) ** 2, axis = 1) ** 0.5 # list of distances between level0_midpoints and random point on surface whose target level is above 0
start_cell = np.argmin(distToMidpoint) # index of cell closest to selceted point on surface
level3_targets.append(start_cell)
level3_checked.append(start_cell) # add targets to list of cells that needn't be checked again
#
# level3_neighbours = all_level3_neighbours(start_cell)

#####################################################
level3_intersecting = [] # all level0_mesh indices intersecting the surface
level3_inter_checked = [] # all level0 cell indices that have been checked for intersection
level3_inter_neighbours = []
level3_intersecting_tmp = []

level3_intersecting.append(start_cell)
level3_inter_checked.append(start_cell)
level3_inter_neighbours = all_level3_neighbours(start_cell)
#begin "while True"-loop
while True:
    level3_intersecting_tmp = []
    for cell in level3_inter_neighbours:
        if level3_intersects_surface(cell):
            level3_intersecting_tmp.append(cell)
    if len(level3_intersecting_tmp) == 0:
        print("all level 3 intersecting cells have been found")
        break
    level3_intersecting.extend(level3_intersecting_tmp)
    level3_inter_checked.extend(level3_inter_neighbours)
    level3_inter_neighbours = []
    for cell in level3_intersecting_tmp:
        level3_inter_neighbours.extend(all_level3_neighbours(cell))
    level3_inter_neighbours = list(set(level3_inter_neighbours))
    level3_inter_neighbours = [x for x in level3_inter_neighbours if x not in level3_inter_checked]
# print(level3_intersecting)
# voxel1 = [("hexahedron", level3_mesh[level3_intersecting])]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./inter_l3.vtk", mesh, file_format="vtk", binary=False)
###################################################

# level3_not_checked_intersecting = np.arange(level3_mesh.shape[0])
level3_not_checked_intersecting = level3_intersecting.copy()
# print("list entries: ", len(level3_not_checked_intersecting))
# level3_not_checked_intersecting.remove(start_cell)
# surface_3_plus_mask: indices of all surface points whose target is above 3

while True:
    tmp = []
    for cell in level3_not_checked_intersecting:
        midpoint = level3_midpoints[cell]
        distToMidpoint = np.sum((surface_points[surface_3_plus_mask] - midpoint) ** 2, axis = 1) ** 0.5
        test_surface_point = surface_points[surface_3_plus_mask[np.argmin(distToMidpoint)]]
        # hex = all_points[level3_mesh[cell]]
        hex = np.tile(midpoint, (8, 1)) + rInf * inflationTemplate
        # print("celllll", cell)
        if hex_contains(hex, test_surface_point):
            # print("hello")
            new_seed = cell
            level3_targets.extend([cell])
            tmp.append(cell)
            break
        else:
            tmp.append(cell)
    # level3_not_checked_intersecting = [x for x in level3_not_checked_intersecting if x not in tmp]
    # print("list entries: ", len(level3_not_checked_intersecting))
    # if len(level3_not_checked_intersecting) == 0:
    #     break
    level3_checked.extend(tmp)
    level3_neighbours = all_level3_neighbours(new_seed)
    while True:
        tempTargets = []
        for cellIndex in level3_neighbours:
            midpoint = level3_midpoints[cellIndex] # midpoint of neighbour
            distToMidpoint = np.sum((surface_points[surface_3_plus_mask] - midpoint) ** 2, axis = 1) ** 0.5 # distance between midpoint and surface_points whose level is above 0
            test_surface_point = surface_points[surface_3_plus_mask[np.argmin(distToMidpoint)]] # coordinates of surface point closest to midpoint
            hex = np.tile(midpoint, (8, 1)) + rInf * inflationTemplate # coordinates of inflated neighbour
            if hex_contains(hex, test_surface_point): # condition for whether neighbour needs to be refined
                tempTargets.append(cellIndex)
        if len(tempTargets) == 0:
            break
        else:
            level3_targets.extend(tempTargets)
        level3_checked.extend(level3_neighbours)
        level3_neighbours = []
        for cellIndex in tempTargets:
            level3_neighbours.extend(all_level3_neighbours(cellIndex))
        level3_neighbours = list(set(level3_neighbours))
        level3_neighbours = [x for x in level3_neighbours if x not in level3_checked]
    level3_not_checked_intersecting = [x for x in level3_not_checked_intersecting if x not in level3_checked]
    # print("list entries: ", len(level3_not_checked_intersecting))
    if len(level3_not_checked_intersecting) == 0:
        break

level3_non_targets = np.delete(level3_non_targets, level3_targets)

# refine seleceted targets from level 3 to level 4
level4_mesh = []
for targetCell in level3_mesh[level3_targets]:
    # split level 2 target cell into 8 children and add children to level3_mesh:
    children = np.diag(targetCell)
    template = newLevelTemplate + all_points.shape[0]
    np.fill_diagonal(template, 0)
    children += template
    level4_mesh.extend(children)
    # compute coordinates of children cells:
    chidlren_coords = newPointsTemplate * 0.5 ** (level + 1) * maxCellSize + all_points[targetCell[0]]
    all_points = np.concatenate((all_points, chidlren_coords), axis = 0)
level4_mesh = np.array(level4_mesh)

# print("export level 4 mesh")
# voxel1 = [("hexahedron", level4_mesh)]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./l4_mesh.vtk", mesh, file_format="vtk", binary=False)

###################################################################
# assembe refined mesh
refined_mesh = np.concatenate((level0_mesh[level0_non_targets], level1_mesh[level1_non_targets], level2_mesh[level2_non_targets], level3_mesh[level3_non_targets], level4_mesh), axis = 0)
all_voxels = [("hexahedron", refined_mesh)]
mesh = meshio.Mesh(points = all_points, cells = all_voxels)
meshio.write("./refined_mesh.vtk", mesh, file_format="vtk", binary=False)
# meshio.write("./trans_refinement2.msh", trans_refinement, file_format="gmsh22", binary=False)

##############################################################

level += 1 # level = 4
rInf = transLayers * maxCellSize * 0.5 ** (level)
# level4_targets = []
# level4_non_targets = np.arange(level4_mesh.shape[0])
# level4_checked = []
surface_4_plus_mask = np.array(np.where(pointTargetLevel == level)).ravel()
level4_midpoints = all_points[level4_mesh[:, 0]] + maxCellSize * 0.5 ** (level + 1)
distToMidpoint = np.sum((surface_points[surface_4_plus_mask[0]] - level4_midpoints) ** 2, axis = 1) ** 0.5 # list of distances between level0_midpoints and random point on surface whose target level is above 0
start_cell = np.argmin(distToMidpoint) # index of cell closest to selceted point on surface
# level4_targets.append(start_cell)
# level4_checked.append(start_cell) # add targets to list of cells that needn't be checked again
#
# level3_neighbours = all_level3_neighbours(start_cell)

#####################################################
level4_intersecting = [] # all level0_mesh indices intersecting the surface
level4_inter_checked = [] # all level0 cell indices that have been checked for intersection
level4_inter_neighbours = []
level4_intersecting_tmp = []

# level4_intersecting.append(start_cell)
# level4_inter_checked.append(start_cell)
level4_inter_neighbours = all_level4_neighbours(start_cell)
#begin "while True"-loop
while True:
    level4_intersecting_tmp = []
    for cell in level4_inter_neighbours:
        if level4_intersects_surface(cell):
            level4_intersecting_tmp.append(cell)
    if len(level4_intersecting_tmp) == 0:
        print("all level 4 intersecting cells have been found")
        break
    level4_intersecting.extend(level4_intersecting_tmp)
    level4_inter_checked.extend(level4_inter_neighbours)
    level4_inter_neighbours = []
    for cell in level4_intersecting_tmp:
        level4_inter_neighbours.extend(all_level4_neighbours(cell))
    level4_inter_neighbours = list(set(level4_inter_neighbours))
    level4_inter_neighbours = [x for x in level4_inter_neighbours if x not in level4_inter_checked]
# print(level3_intersecting)
voxel1 = [("hexahedron", level4_mesh[level4_intersecting])]
mesh = meshio.Mesh(points = all_points, cells = voxel1)
meshio.write("./inter_l4.vtk", mesh, file_format="vtk", binary=False)
###################################################
# assemble all intersected cells
# level4_intersecting
# level3_intersecting - level3_targets
level3_intersecting_temp = [x for x in level3_intersecting if x not in level3_targets]
cut_mesh = np.concatenate((level3_mesh[level3_intersecting_temp], level4_mesh[level4_intersecting]), axis = 0)
all_voxels = [("hexahedron", cut_mesh)]
mesh = meshio.Mesh(points = all_points, cells = all_voxels)
meshio.write("./cut_mesh.vtk", mesh, file_format="vtk", binary=False)
########################################################
# classify mesh via flood fill
# inside_point = np.array([0.0, 0.0, 0.0])
# distToMidpoint = np.sum((inside_point - level3_midpoints) ** 2, axis = 1) ** 0.5 # list of distances between level0_midpoints and random point on surface whose target level is above 0
# start_cell = np.argmin(distToMidpoint) # index of cell closest to selceted point on surface
# given: level0_targets, level1_targets, level2_targets, level3_targets
start_cell = 2312
start_level = 3


# voxel1 = [("hexahedron", level3_mesh[[start_cell]])]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./flood_fill_start.vtk", mesh, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", level3_mesh[level3_targets])]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./l3_targets_it_whatever.vtk", mesh, file_format="vtk", binary=False)
