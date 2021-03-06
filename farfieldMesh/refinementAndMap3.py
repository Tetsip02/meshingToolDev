import meshio
import numpy as np
import pickle
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


octreeMap = {'U': [[4, "H"], [5, "H"], [6, "H"], [7, "H"], [0, "U"], [1, "U"], [2, "U"], [3, "U"]],
             'B': [[3, "H"], [2, "H"], [1, "B"], [0, "B"], [7, "H"], [6, "H"], [5, "B"], [4, "B"]],
             'R': [[1, "H"], [0, "R"], [3, "R"], [2, "H"], [5, "H"], [4, "R"], [7, "R"], [6, "H"]],
             'D': [[4, "D"], [5, "D"], [6, "D"], [7, "D"], [0, "H"], [1, "H"], [2, "H"], [3, "H"]],
             'F': [[3, "F"], [2, "F"], [1, "H"], [0, "H"], [7, "F"], [6, "F"], [5, "H"], [4, "H"]],
             'L': [[1, "L"], [0, "H"], [3, "H"], [2, "L"], [5, "L"], [4, "H"], [7, "H"], [6, "L"]]}

def level1_neighbour(target, direction):
    # target: cell index within level1_mesh
    # direction: {"U", "B", "R", "D", "F", "L"}
    # return neighbour octant and neighbour parent
    x = list(divmod(target, 8))
    target_octant = x[1]
    target_parent = level0_targets[x[0]]
    map = octreeMap[direction][target_octant]
    if map[1] == "H":
        neighbour_parent = target_parent.copy()
    else:
        neighbour_parent = root_neighbour(target_parent, map[1])
    neighbour_octant = map[0]
    return neighbour_octant, neighbour_parent

octreeMap2 = np.array([[[4, "H"], [3, "H"], [1, "H"], [4, "D"], [3, "F"], [1, "L"]],
                       [[5, "H"], [2, "H"], [0, "R"], [5, "D"], [2, "F"], [0, "H"]],
                       [[6, "H"], [1, "B"], [3, "R"], [6, "D"], [1, "H"], [3, "H"]],
                       [[7, "H"], [0, "B"], [2, "H"], [7, "D"], [0, "H"], [2, "L"]],
                       [[0, "U"], [7, "H"], [5, "H"], [0, "H"], [7, "F"], [5, "L"]],
                       [[1, "U"], [6, "H"], [4, "R"], [1, "H"], [6, "F"], [4, "H"]],
                       [[2, "U"], [5, "B"], [7, "R"], [2, "H"], [5, "H"], [7, "H"]],
                       [[3, "U"], [4, "B"], [6, "H"], [3, "H"], [4, "H"], [6, "L"]]])


# retrieving level1 neighbours:
# cycle through neigh_parents
# if neigh_parent in list of level0_targets, get it's index within level0_targets
# compute level 1 neighbour index as: l0_target_index * 8 + corresponding octant number
# add result to list of level1 neighbours and proceed as previously

def all_level1_neighbours_map(target):
    # target: cell index within level1_mesh
    # return level 1 neighbours in all orthogonal directions {"U", "B", "R", "D", "F", "L"}
    x = list(divmod(target, 8))
    target_octant = x[1]
    target_parent = level0_targets[x[0]]
    map = octreeMap2[target_octant]
    neigh_octants = map[:, 0].astype(int)
    neigh_parents = []
    for direction in map[:, 1]:
        if direction == "H":
            neigh_parents.extend([target_parent])
        else:
            neigh_parents.extend([root_neighbour(target_parent, direction)])
    return neigh_octants, neigh_parents
def all_level1_neighbours(target):
    neigh_octants, neigh_parents = all_level1_neighbours_map(target)
    level1_neighbours = [] # neighbours (in ortho-directions) of start_cell
    for parent, octant in zip(neigh_parents, neigh_octants):
        if parent in level0_targets:
            level1_neighbours.extend(np.argwhere(level0_targets == parent).ravel() * 8 + octant) # apparently ravel() is faster than reshape(-1)
    return level1_neighbours

def all_level2_neighbours_map(target):
    # target: cell index within level2_mesh
    # return level 2 neighbours octants and their level1 parent in all orthogonal directions {"U", "B", "R", "D", "F", "L"}
    x = list(divmod(target, 8))
    target_octant = x[1]
    target_parent = level1_targets[x[0]]
    map = octreeMap2[target_octant]
    neigh_octants = map[:, 0].astype(int)
    neigh_parents = []
    for direction in map[:, 1]:
        if direction == "H":
            neigh_parents.extend([target_parent])
        else:
            map2 = level1_neighbour(target_parent, direction)
            neigh_parents.extend(np.argwhere(level0_targets == map2[1]).ravel() * 8 + map2[0])
    return neigh_octants, neigh_parents
def all_level2_neighbours(target):
    neigh_octants, neigh_parents = all_level2_neighbours_map(target)
    level2_neighbours = [] # neighbours (in ortho-directions) of start_cell
    for parent, octant in zip(neigh_parents, neigh_octants):
        if parent in level1_targets:
            level2_neighbours.extend(np.argwhere(level1_targets == parent).ravel() * 8 + octant) # apparently ravel() is faster than reshape(-1)
    return level2_neighbours

# def level2_neighbour(target, direction):
#     # target: cell index within level1_mesh
#     # direction: {"U", "B", "R", "D", "F", "L"}
#     # res = level2_neighbour(target, direction) = (octant_within_level1_cell, (octant_within_root_parent, root_parent))
#     x = list(divmod(target, 8))
#     target_octant = x[1]
#     target_parent = level1_targets[x[0]]
#     map = octreeMap[direction][target_octant]
#     if map[1] == "H":
#         neighbour_parent = target_parent.copy()
#     else:
#         neighbour_parent = level1_neighbour(target_parent, map[1])
#     neighbour_octant = map[0]
#     return neighbour_octant, neighbour_parent
#
# def all_level3_neighbours_map(target):
#     # target: cell index within level3_mesh
#     # return level 3 neighbours octants and their level2 parents in all orthogonal directions {"U", "B", "R", "D", "F", "L"}
#     x = list(divmod(target, 8))
#     target_octant = x[1]
#     target_parent = level2_targets[x[0]]
#     map = octreeMap2[target_octant]
#     neigh_octants = map[:, 0].astype(int)
#     neigh_parents = []
#     for direction in map[:, 1]:
#         if direction == "H":
#             neigh_parents.extend([target_parent])
#         else:
#             map2 = level2_neighbour(target_parent, direction)
#             l1index = np.argwhere(level0_targets == map2[1][1]).ravel() * 8 + map2[1][0]
#             l2index = np.argwhere(level1_targets == l1index).ravel() * 8 + map2[0]
#             neigh_parents.extend(l2index)
#     return neigh_octants, neigh_parents
# def all_level3_neighbours(target):
#     neigh_octants, neigh_parents = all_level3_neighbours_map(target)
#     level3_neighbours = [] # neighbours (in ortho-directions) of start_cell
#     for parent, octant in zip(neigh_parents, neigh_octants):
#         if parent in level2_targets:
#             level3_neighbours.extend(np.argwhere(level2_targets == parent).ravel() * 8 + octant) # apparently ravel() is faster than reshape(-1)
#     return level3_neighbours


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
rInf = transLayers * maxCellSize * 0.5 ** (level + 1)
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

# begin loop for identifying targets fro refinement:
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

# voxel1 = [("hexahedron", level1_mesh)]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./l1_mesh.vtk", mesh, file_format="vtk", binary=False)

level += 1 # level = 1
rInf = transLayers * maxCellSize * 0.5 ** (level + 1)
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

# voxel1 = [("hexahedron", level2_mesh)]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./l2_mesh.vtk", mesh, file_format="vtk", binary=False)

level += 1 # level = 2
rInf = transLayers * maxCellSize * 0.5 ** (level + 1)
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

# voxel1 = [("hexahedron", level3_mesh)]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./l3_mesh.vtk", mesh, file_format="vtk", binary=False)


level += 1 # level = 3
rInf = transLayers * maxCellSize * 0.5 ** (level + 1)
level3_targets = []
level3_non_targets = np.arange(level3_mesh.shape[0])
level3_checked = []
surface_3_plus_mask = np.array(np.where(pointTargetLevel > level)).ravel()
level3_midpoints = all_points[level3_mesh[:, 0]] + maxCellSize * 0.5 ** (level + 1)
distToMidpoint = np.sum((surface_points[surface_3_plus_mask[0]] - level3_midpoints) ** 2, axis = 1) ** 0.5 # list of distances between level0_midpoints and random point on surface whose target level is above 0
start_cell = np.argmin(distToMidpoint) # index of cell closest to selceted point on surface
level3_targets.append(start_cell)
level3_checked.append(start_cell) # add targets to list of cells that needn't be checked again

# level3_neighbours = all_level3_neighbours(start_cell)

#l3 targets[29, 30, 124, 25]

def level1_neighbour_temp(target, direction):
    # target: cell index within level1_mesh
    # direction: {"U", "B", "R", "D", "F", "L"}
    # return neighbour octant and neighbour parent
    x = list(divmod(target, 8))
    target_octant = x[1]
    target_parent = level0_targets[x[0]]
    map = octreeMap[direction][target_octant]
    if map[1] == "H":
        neighbour_parent = target_parent.copy()
    else:
        neighbour_parent = root_neighbour(target_parent, map[1])
    neighbour_octant = map[0]
    return [neighbour_octant, neighbour_parent]

# x = level1_neighbour_temp(1, "U")
# print(x)
# print(type(x))

def level2_neighbour(target, direction):
    global tmp
    # target: cell index within level1_mesh
    # direction: {"U", "B", "R", "D", "F", "L"}
    # res = level2_neighbour(target, direction) = (octant_within_level1_cell, (octant_within_root_parent, root_parent))
    x = list(divmod(target, 8))
    target_octant = x[1]
    target_parent = level1_targets[x[0]]
    map = octreeMap[direction][target_octant]
    if map[1] == "H":
        neighbour_parent = target_parent.copy()
    else:
        # l1_octant, neighbour_parent = level1_neighbour_temp(target_parent, map[1])
        tmp = level1_neighbour_temp(target_parent, map[1])
        # print(tmp)
    neighbour_octant = map[0]
    # neighbour_parent = list(neighbour_parent)
    # return [neighbour_octant, l1_octant, neighbour_parent]
    return [neighbour_octant, tmp[0], tmp[1]]

# x = level2_neighbour(6, "U")
# print(x)
# print(type(x))

def all_level3_neighbours_map(target):
    # target: cell index within level3_mesh
    # return level 3 neighbours octants and their level2 parents in all orthogonal directions {"U", "B", "R", "D", "F", "L"}
    x = list(divmod(target, 8))
    target_octant = x[1]
    target_parent = level2_targets[x[0]]
    # print(target_parent)
    map = octreeMap2[target_octant]
    # map = [octreeMap2[target_octant]]
    # print(map)
    neigh_octants = list(map[:, 0].astype(int))
    # neigh_octants = map[:, 0]
    # print(neigh_octants)
    neigh_parents = []
    for direction in map[:, 1]:
        if direction == "H":
            neigh_parents.extend([target_parent])
        else:
            map2 = level2_neighbour(target_parent, direction)
            # print(map2)
            l1index = np.argwhere(level0_targets == map2[2]).ravel() * 8 + map2[1]
            # print(l1index)
            l2index = np.argwhere(level1_targets == l1index).ravel() * 8 + map2[0]
            neigh_parents.extend(l2index)
            # print(l2index)
            # print(neigh_parents)
    # print(neigh_octants, neigh_parents)
    return neigh_octants, neigh_parents
def all_level3_neighbours(target):
    neigh_octants, neigh_parents = all_level3_neighbours_map(target)
    level3_neighbours = [] # neighbours (in ortho-directions) of start_cell
    for parent, octant in zip(neigh_parents, neigh_octants):
        if parent in level2_targets:
            level3_neighbours.extend(np.argwhere(level2_targets == parent).ravel() * 8 + octant) # apparently ravel() is faster than reshape(-1)
    return level3_neighbours

# print(all_level3_neighbours_map(29))
x = all_level3_neighbours(25)
print(x)
print(type(x))
# print(level2_neighbour(5, "U"))
# x, y = level2_neighbour(5, "U")
# print(x)
# print(y)
# print(type(y))
# test = all_level3_neighbours(30)
# print(test)
# voxel1 = [("hexahedron", level3_mesh[[25]])]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./l3_cell25.vtk", mesh, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", level3_mesh[x])]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./l3_cell25_neigh.vtk", mesh, file_format="vtk", binary=False)

# tempTargets = []
# for cellIndex in level3_neighbours:
#     midpoint = level3_midpoints[cellIndex] # midpoint of neighbour
#     distToMidpoint = np.sum((surface_points[surface_3_plus_mask] - midpoint) ** 2, axis = 1) ** 0.5 # distance between midpoint and surface_points whose level is above 0
#     test_surface_point = surface_points[surface_3_plus_mask[np.argmin(distToMidpoint)]] # coordinates of surface point closest to midpoint
#     hex = np.tile(midpoint, (8, 1)) + rInf * inflationTemplate # coordinates of inflated neighbour
#     if hex_contains(hex, test_surface_point): # condition for whether neighbour needs to be refined
#         tempTargets.append(cellIndex)
# if len(tempTargets) == 0:
#     print("nothing to refine")#break
# else:
#     level3_targets.extend(tempTargets)
# level3_checked.extend(level3_neighbours)
# level3_neighbours = []
# print(tempTargets)
# for cellIndex in tempTargets:
#     level3_neighbours.extend(all_level3_neighbours(cellIndex))
# level3_neighbours = list(set(level3_neighbours))
# level3_neighbours = [x for x in level3_neighbours if x not in level3_checked]
#
# print(level3_neighbours)

# while True:
#     tempTargets = []
#     for cellIndex in level2_neighbours:
#         midpoint = level2_midpoints[cellIndex] # midpoint of neighbour
#         distToMidpoint = np.sum((surface_points[surface_2_plus_mask] - midpoint) ** 2, axis = 1) ** 0.5 # distance between midpoint and surface_points whose level is above 0
#         test_surface_point = surface_points[surface_2_plus_mask[np.argmin(distToMidpoint)]] # coordinates of surface point closest to midpoint
#         hex = np.tile(midpoint, (8, 1)) + rInf * inflationTemplate # coordinates of inflated neighbour
#         if hex_contains(hex, test_surface_point): # condition for whether neighbour needs to be refined
#             tempTargets.append(cellIndex)
#     if len(tempTargets) == 0:
#         break
#     else:
#         level2_targets.extend(tempTargets)
#     level2_checked.extend(level2_neighbours)
#     level2_neighbours = []
#     for cellIndex in tempTargets:
#         level2_neighbours.extend(all_level2_neighbours(cellIndex))
#     level2_neighbours = list(set(level2_neighbours))
#     level2_neighbours = [x for x in level2_neighbours if x not in level2_checked]
# level2_non_targets = np.delete(level2_non_targets, level2_targets)

# voxel1 = [("hexahedron", level3_mesh[y])]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./l3_neigh_it1.vtk", mesh, file_format="vtk", binary=False)
# voxel1 = [("hexahedron", level2_mesh[level2_targets])]
# mesh = meshio.Mesh(points = all_points, cells = voxel1)
# meshio.write("./l2_targets.vtk", mesh, file_format="vtk", binary=False)
