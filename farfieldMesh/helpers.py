import meshio
import numpy as np

# centroid: takes in lsit of points of a polygon, returns x-, y-, z-coordinates of centroid
# read_surface: meshio wrapper that reads in surface mesh, returns list of point-coordinates, list of triangle faces, list of quad faces
# get_face_normals: takes in points, quads and triangles of a surface mesh, returns list of normal-vectors for triangle faces and list of normal-vectors for quad-faces
# area: returns area of polygon
# hex_contains: given a AABB and a point, return True if point is contained in AABB, returns False otherwise
# root_neighbour: given index of a root and a direction, return index of neighbour-root in given direction. Returns -1 if no neighbour exists. basemesh needs to be ordered in certain way for this to work
# bounding_box: given a list of points, return list of 8 points that make up corners of bounding box
# triInBox: given AABB and a triangle: return True if they overlap, return false otherwise


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
    # print(FS)
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
