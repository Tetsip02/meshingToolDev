import meshio
import numpy as np
# from shapely.geometry import Point, Polygon
# from vtk.util import numpy_support
# from pyevtk.hl import gridToVTK
import pickle
# import trimesh
# from pvtrace import *
# from trimesh import contains_points

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
    triNormal = np.ndarray((nTriFaces, 3), dtype = np.float64)
    quadNormal = np.ndarray((nQuadFaces, 3), dtype = np.float64)
    for i, triFace in enumerate(triangles):
        coor = points[triFace, :]
        triNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
    for i, quadFace in enumerate(quads):
        coor = surface_points[quadFace, :]
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


surface_triangles_ini, surface_quads_ini, surface_points = read_surface("./sphere.obj")

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
# get list of face normals
triangleNormals, quadNormals = get_face_normals(surface_points, surface_triangles, surface_quads)

# get list of face centroids
faceCentroids = []
faceArea = []
sizeField = []
targetLevel = []
triSizeFieldFactor = 2 * 3 ** -0.25
maxCellSize = 1.2
maxLevel = 10

for i, face in enumerate(surface_triangles):
    faceCentroids.append(centroid(surface_points[face]))
    triArea = area(surface_points[face], triangleNormals[i])
    faceArea.append(triArea)
    edgeSize = triSizeFieldFactor * (triArea ** 0.5)
    sizeField.append(edgeSize)
    r = np.log2(maxCellSize/edgeSize)
    if np.abs(2 ** np.int(r) - maxCellSize) < np.abs(2 ** (np.int(r) + 1) - maxCellSize):
        targetLevel.append(np.int(r))
    else:
        targetLevel.append(np.int(r) + 1)
for i, face in enumerate(surface_quads):
    faceCentroids.append(centroid(surface_points[face]))
    quadArea = area(surface_points[face], quadNormals[i])
    faceArea.append(quadArea)
    edgeSize = quadArea ** 0.5
    sizeField.append(edgeSize)
    r = np.log2(maxCellSize/edgeSize)
    if np.abs(2 ** np.int(r) - maxCellSize) < np.abs(2 ** (np.int(r) + 1) - maxCellSize):
        targetLevel.append(np.int(r))
    else:
        targetLevel.append(np.int(r) + 1)

faceCentroids = np.array(faceCentroids)
faceArea = np.array(faceArea)
sizeField = np.array(sizeField)
targetLevel = np.array(targetLevel)


# # #assemble final mesh
# l0_mesh_final = np.delete(level0_mesh_reduced, l0_trans_targets, axis=0)
# l1_mesh_final = np.delete(l1_cells, l1_trans_targets, axis=0)
# l2_mesh_final = np.delete(l2_cells, l2_trans_targets, axis=0)
# refined_mesh = np.concatenate((l0_mesh_final, l1_mesh_final, l2_mesh_final, level3_mesh_reduced, l2_trans_mesh, level4_mesh), axis = 0)
# all_voxels = [("hexahedron", refined_mesh)]
# trans_refinement = meshio.Mesh(points = all_points, cells = all_voxels)
# meshio.write("./trans_refinement2.vtk", trans_refinement, file_format="vtk", binary=False)
# meshio.write("./trans_refinement2.msh", trans_refinement, file_format="gmsh22", binary=False)
# #################################################################################################
# with open("./refined_mesh.txt", "wb") as fp:   #Pickling
#     pickle.dump(refined_mesh, fp)
# with open("./all_points.txt", "wb") as fp:   #Pickling
#     pickle.dump(all_points, fp)
with open("./refined_mesh.txt", "rb") as fp:   # Unpickling
    refined_mesh = pickle.load(fp)
with open("./all_points.txt", "rb") as fp:   # Unpickling
    all_points = pickle.load(fp)

all_voxels = [("hexahedron", refined_mesh)]
start = meshio.Mesh(points = all_points, cells = all_voxels)
meshio.write("./start.vtk", start, file_format="vtk", binary=False)
# meshio.write("./trans_refinement2.msh", trans_refinement, file_format="gmsh22", binary=False)

def bounding_box(points):
    # bbox = np.ndarray((2, 3), dtype = np.float64)
    # bbox[0, :] = np.min(points, axis = 0)
    # bbox[1, :] = np.max(points, axis = 0)
    P0 = np.min(points, axis = 0)
    P6 = np.max(points, axis = 0)
    dist = P6 - P0
    # print(dist)
    P1 = P0.copy()
    P1[0] += dist[0]
    # print(P1)
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
    # print(bbox_points)
    return bbox


ref = np.array([0.4395064455, 0.617598629942, 0.652231566745])
# ray_directions = np.tile(default_direction, (inside_aabb.sum(), 1))

# delete unwanted cells
# 1: create list of zeros of size = number of points
# 2: cull points; get all points inside bounding_box
# 3: cast ray in both directions
# 4: count how often each point intersects surface in each direction
# 5: if one ray in either direction hit nothing, that's a solid indicator that we're in free space, so delete point from list
# 6: if total number of intersections is odd, point is probably inside

bbox = bounding_box(surface_points)
inside_points = [] # all points contained by bounding_box of surface mesh
for i, node in enumerate(all_points):
    if hex_contains(bbox, node):
        inside_points.append(i)
# print(all_points.shape)
# print(inside_points)
# print(refined_mesh[10100])
# x = np.array([0, 1, 2, 3, 4])
# x = [0, 1, 2]
# # y = [22866, 22867, 22883]
# y = np.array([4, 5, 3])
# # a = list(set(y).intersection(inside_points))
# # print("a", a)
# # print(refined_mesh.shape)
# a = np.intersect1d(x, y, assume_unique = True)
# # a = np.intersect1d(x, y)
# print(a.size)
# print(a)
# Vtype = np.zeros(refined_mesh.shape[0], dtype = np.int)
# # print(Vtype)
# # print(Vtype.shape)
# t = refined_mesh[0]
# print(t)
culled_mesh_ind = []
for i, V in enumerate(refined_mesh):
    common = np.intersect1d(V, inside_points, assume_unique = True)
    if common.size > 0:
        culled_mesh_ind.append(i)
# print("culled", culled_mesh_ind)
# print(len(culled_mesh_ind))
culled_mesh = refined_mesh[culled_mesh_ind]

culled_voxels = [("hexahedron", culled_mesh)]
inter1 = meshio.Mesh(points = all_points, cells = culled_voxels)
meshio.write("./inter1.vtk", inter1, file_format="vtk", binary=False)

V = culled_mesh_ind[0]
dist = all_points[refined_mesh[V][0]] - surface_points
# print(dist)
# print(dist.shape)
# print(surface_points.shape)
# print(all_points[refined_mesh[V][0]])
# print(surface_points[5])
# print(dist[5])
dist = np.linalg.norm(dist, axis = 1)[: , None]
# print(dist[5])
# print(dist)
# print(dist.shape)
a = np.argmin(dist) # point on surface mesh closest to voxel corner
print(a)
print(dist[1316])


# for V in culled_mesh_ind:
#     dist = all_points[refined_mesh[V][0]]


# all_normals = np.concatenate((triangleNormals, quadNormals), axis = 0) # assume normals are pointing outwards, clockwise face-ordering
# allFaces = surface_triangles.tolist()
# allFaces.extend(surface_quads)
# print(triangleNormals.shape)
# print(quadNormals.shape)
# print(all_normals.shape)
# print(faceCentroids.shape)
# print(sizeField.shape)
# print(len(allFaces))
# print(allFaces[200])
# # for node in inside_points:
# #     P = all_points[node]
# #     for C, n, s in zip(faceCentroids, all_normals, sizeField):
# #         d = np.dot((P - C), n) / np.dot(ref, n)
# eps = 0.1
# node = inside_points[1]
# P = all_points[node]
# temp = 0
# count = []
# for node in inside_points:
#     P = all_points[node]
#     tempL = []
#     for i, (C, n, face) in enumerate(zip(faceCentroids, all_normals, allFaces)):
#         d = np.dot((P - C), n) / np.dot(ref, n)
#         S = P + d * ref
#         if len(face) == 3:
#             n1 = np.cross(surface_points[face[1]] - surface_points[face[0]], S - surface_points[face[0]])
#             n1 = n1 / np.linalg.norm(n1)
#             n2 = np.cross(surface_points[face[2]] - surface_points[face[1]], S - surface_points[face[1]])
#             n2 = n2 / np.linalg.norm(n2)
#             n3 = np.cross(surface_points[face[0]] - surface_points[face[2]], S - surface_points[face[2]])
#             n3 = n3 / np.linalg.norm(n3)
#             # print(np.dot(n1, n2))
#             # print(np.dot(n2, n3))
#             if np.dot(n1, n2) > 0.9 and np.dot(n2, n3) > 0.9:
#                 print(i, "count")
#                 tempL.append(i)
#             # print(np.dot(n3, n))
#         elif len(face) == 4:
#             n1 = np.cross(surface_points[face[1]] - surface_points[face[0]], S - surface_points[face[0]])
#             n1 = n1 / np.linalg.norm(n1)
#             n2 = np.cross(surface_points[face[2]] - surface_points[face[1]], S - surface_points[face[1]])
#             n2 = n2 / np.linalg.norm(n2)
#             n3 = np.cross(surface_points[face[3]] - surface_points[face[2]], S - surface_points[face[2]])
#             n3 = n3 / np.linalg.norm(n3)
#             n4 = np.cross(surface_points[face[0]] - surface_points[face[3]], S - surface_points[face[3]])
#             n4 = n4 / np.linalg.norm(n4)
#             # print("ho")
#             if np.dot(n1, n2) > 0.9 and np.dot(n2, n3) > 0.9 and np.dot(n3, n4) > 0.9:
#                 print(i, "count")
#                 tempL.append(i)
#     count.append(tempL)
#     temp += 1
# print(count)



# faces1 = surface_triangles.copy()
# faces1 = [faces1]
# faces1.extend(surface_quads)
# # print(faces1)
# mesh = trimesh.Trimesh(vertices=surface_points, faces = surface_quads, process=False)
# print(mesh)
# x = trimesh.ray.ray_util.contains_points(mesh, all_points, check_direction=None)
# print(x)
# def contains_points(intersector, points, check_direction=None):
#     # placeholder result with no hits we'll fill in later
#     contains = np.zeros(len(points), dtype=np.bool)
# mesh1 = trimesh.load('./sphere.obj', use_embree=False)
# print(mesh1)
