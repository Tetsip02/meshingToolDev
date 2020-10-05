import meshio
import numpy as np
# from shapely.geometry import Point, Polygon
# from vtk.util import numpy_support
# from pyevtk.hl import gridToVTK
import pickle

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
    n1 = np.cross(hex_points[1] - hex_points[0], hex_points[2] - hex_points[0])
    x1 = np.dot(x - hex_points[0], n1)
    if x1 < 0:
        return False
    n2 = np.cross(hex_points[7] - hex_points[4], hex_points[6] - hex_points[4])
    x2 = np.dot(x - hex_points[4], n2)
    if x2 < 0:
        return False
    n3 = np.cross(hex_points[3] - hex_points[0], hex_points[7] - hex_points[0])
    x3 = np.dot(x - hex_points[0], n3)
    if x3 < 0:
        return False
    n4 = np.cross(hex_points[5] - hex_points[1], hex_points[6] - hex_points[1])
    x4 = np.dot(x - hex_points[1], n4)
    if x4 < 0:
        return False
    n5 = np.cross(hex_points[4] - hex_points[0], hex_points[5] - hex_points[0])
    x5 = np.dot(x - hex_points[0], n5)
    if x5 < 0:
        return False
    n6 = np.cross(hex_points[2] - hex_points[3], hex_points[6] - hex_points[3])
    x6 = np.dot(x - hex_points[3], n6)
    if x6 < 0:
        return False
    return True


surface_triangles, surface_quads, surface_points = read_surface("./sphere.obj")

nPoints = len(surface_points)
nTriFaces = len(surface_triangles)
nQuadFaces = len(surface_quads)
nFaces = nTriFaces + nQuadFaces

# NumPy_data_shape = NumPy_data.shape
# VTK_data = numpy_support.numpy_to_vtk(num_array=surface_points, deep=True, array_type=vtk.VTK_FLOAT)
# VTK_data = numpy_support.numpy_to_vtk(num_array=surface_points, deep=True)

# get list of face normals
triangleNormals, quadNormals = get_face_normals(surface_points, surface_triangles, surface_quads)

# get list of face centroids
faceCentroids = []
faceArea = []
sizeField = []
targetLevel = []
triSizeFieldFactor = 2 * 3 ** -0.25
maxCellSize = 4.0
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

# set up base mesh
bBox_origin = np.array([0.0, 0.0, 0.0])
xmin = 10.0
xmax = 10.0
ymin = 10.0
ymax = 10.0
zmin = 10.0
zmax = 10.0
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



# targetPoints = []
# targetPointsComplete = []
# for leaf in z_hexas:
#     for j, target in enumerate(faceCentroids):
#         if hex_contains(l0_points[leaf], target):
#             targetPoints.append(j)
#     targetPointsComplete.append(targetPoints)
#     targetPoints = []
# with open("./targetPointsComplete.txt", "wb") as fp:   #Pickling
#     pickle.dump(targetPointsComplete, fp)
with open("./targetPointsComplete.txt", "rb") as fp:   # Unpickling
    targetPointsComplete = pickle.load(fp)


# level1_refinement

n_base_points = (nDivisions[0] + 1) * (nDivisions[1] + 1) * (nDivisions[2] + 1)
n_base_cells = nDivisions[0] * nDivisions[1] * nDivisions[2]

newPointsTemplate = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [1, 2, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 1], [1, 1, 1], [2, 1, 1], [0, 2, 1], [1, 2, 1], [2, 2, 1], [1, 0, 2], [0, 1, 2], [1, 1, 2], [2, 1, 2], [1, 2, 2]])

newLevelTemplate = np.array([[0, 0, 2, 1, 5, 6, 9, 8], [0, 0, 3, 2, 6, 7, 10, 9], [2, 3, 0, 4, 9, 10, 13, 12], [1, 2, 4, 0, 8, 9, 12, 11], [5, 6, 9, 8, 0, 14, 16, 15], [6, 7, 10, 9, 14, 0, 17, 16], [9, 10, 13, 12, 16, 17, 0, 18], [8, 9, 12, 11, 15, 16, 18, 0]])

level0_mesh = z_hexas.copy()
level0_mesh_reduced = z_hexas.copy() # level0_mesh minus the refined hexas
level1_mesh = []
# refined_mesh = z_hexas.copy()
all_points = l0_points.copy()
level1_targetCells = []
# print(len(targetPointsComplete))
for i, targets in enumerate(targetPointsComplete):
    if len(targets) != 0:
        if np.max(targetLevel[targets]) > 0 and np.max(targetLevel[targets]) < maxLevel:
            level1_targetCells.append(i)
            new_points = newPointsTemplate * 0.5 * maxCellSize + all_points[level0_mesh[i][0]] # add new points of refined hex (19 points)
            all_points = np.concatenate((all_points, new_points), axis = 0)
            new_level = np.diag(level0_mesh[i])
            template = newLevelTemplate + n_base_points
            np.fill_diagonal(template, 0)
            new_level += template
            level1_mesh.extend(new_level)
            # refined_mesh = np.delete(refined_mesh, i, axis=0)

for refined in level1_targetCells:
    level0_mesh_reduced = np.delete(level0_mesh_reduced, refined, axis=0)

reduced_voxels = [("hexahedron", level0_mesh_reduced)]
reduced_mesh = meshio.Mesh(points = all_points, cells = reduced_voxels)
meshio.write("./reduced_voxels.vtk", reduced_mesh, file_format="vtk", binary=False)

level1_mesh = np.array(level1_mesh)
level1_voxels = [("hexahedron", level1_mesh)]

level1_cells = meshio.Mesh(points = all_points, cells = level1_voxels)
meshio.write("./level1_cells.vtk", level1_cells, file_format="vtk", binary=False)

refined_mesh = np.concatenate((level0_mesh_reduced, level1_mesh), axis = 0)
all_voxels = [("hexahedron", refined_mesh)]

level1_refinement = meshio.Mesh(points = all_points, cells = all_voxels)
meshio.write("./level1_refinement.vtk", level1_refinement, file_format="vtk", binary=False)
meshio.write("./level1_refinement.msh", level1_refinement, file_format="gmsh22", binary=False)

# level2_refinement

# level1_targetPoints = []
# level1_targetPointsTemp = []
# for i, level1_cell in enumerate(level1_targetCells):
#     # print(level1_cell)
#     children = level1_mesh[i:i+8]
#     for child in children:
#         for j, targetPoint in enumerate(targetPointsComplete[level1_cell]):
#             if hex_contains(all_points[child], faceCentroids[targetPoint]):
#                 level1_targetPointsTemp.append(j)
#         level1_targetPoints.append(level1_targetPointsTemp)
#         level1_targetPointsTemp = []
# with open("./level1_targetPoints.txt", "wb") as fp:   #Pickling
#     pickle.dump(level1_targetPoints, fp)
with open("./level1_targetPoints.txt", "rb") as fp:   # Unpickling
    level1_targetPoints = pickle.load(fp)

level2_mesh = []
for i, targets in enumerate(level1_targetPoints):
    if len(targets) != 0:
        if np.max(targetLevel[targets]) > 0 and np.max(targetLevel[targets]) < maxLevel:
    #         level1_targetCells.append(i)
            # new_points = newPointsTemplate * 0.5 ** 2 * maxCellSize + all_points[level1_mesh[np.int(i / 8) * 8][0]] # add new points of refined hex (19 points)
            new_points = newPointsTemplate * 0.5 ** 2 * maxCellSize + all_points[level1_mesh[i][0]]
            # print(new_points)
            all_points = np.concatenate((all_points, new_points), axis = 0)
            new_level = np.diag(level1_mesh[i])
            template = newLevelTemplate + n_base_points + 19 * (i + 1)
            np.fill_diagonal(template, 0)
            new_level += template
            level2_mesh.extend(new_level)
            # print(refined_mesh.shape)
            # refined_mesh = np.delete(refined_mesh, i + n_base_cells, axis=0)

level2_mesh = np.array(level2_mesh)
level2_voxels = [("hexahedron", level2_mesh)]
level2_cells = meshio.Mesh(points = all_points, cells = level2_voxels)
meshio.write("./level2_cells.vtk", level2_cells, file_format="vtk", binary=False)


refined_mesh = np.concatenate((level0_mesh_reduced, level2_mesh), axis = 0)
all_voxels = [("hexahedron", refined_mesh)]

level2_refinement = meshio.Mesh(points = all_points, cells = all_voxels)
meshio.write("./level2_refinement.vtk", level2_refinement, file_format="vtk", binary=False)
meshio.write("./level2_refinement.msh", level2_refinement, file_format="gmsh22", binary=False)

# # refinement
#
# refineTemplate = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [1, 2, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 1], [1, 1, 1], [2, 1, 1], [0, 2, 1], [1, 2, 1], [2, 2, 1], [1, 0, 2], [0, 1, 2], [1, 1, 2], [2, 1, 2], [1, 2, 2]])
#
# refineMeshTemplate = np.array([[0, 0, 2, 1, 5, 6, 9, 8], [0, 0, 3, 2, 6, 7, 10, 9], [2, 3, 0, 4, 9, 10, 13, 12], [1, 2, 4, 0, 8, 9, 12, 11], [5, 6, 9, 8, 0, 14, 16, 15], [6, 7, 10, 9, 14, 0, 17, 16], [9, 10, 13, 12, 16, 17, 0, 18], [8, 9, 12, 11, 15, 16, 18, 0]])
#
# # for each point
#     # for each leaf, check if leaf contains point, if yes, refine
#
# # print(z_hexas[0])
# l0_mesh = z_hexas.copy()
# all_points = l0_points.copy()
# # l1_points = []
# # l1_mesh = []
# index = 0
# n_points_l0 = 216
# point0 = faceCentroids[index]
# # print(targetLevel[index])
# for i, leaf in enumerate(z_hexas):
#     if hex_contains(l0_points[leaf], point0): # extract hex that contains point0
#         if targetLevel[index] > 0: # if target level of point0 is larger than 0, refine
#             l0_mesh = np.delete(l0_mesh, i, axis=0) # remove hex in question
#             l1_points = refineTemplate * 0.5 * maxCellSize + l0_points[leaf[0]] # add new points of refined hex (19 points)
#             all_points = np.concatenate((all_points, l1_points), axis = 0)
#             l1_mesh = np.diag(leaf)
#             template = refineMeshTemplate + n_points_l0
#             np.fill_diagonal(template, 0)
#             l1_mesh += template
#             children = l1_mesh.copy()
#             for j, leaf_1 in enumerate(children):
#                 if hex_contains(all_points[leaf_1], point0):
#                     if targetLevel[index] > 1:
#                         l1_mesh = np.delete(l1_mesh, j, axis=0)
#                         l2_points = refineTemplate * 0.5 ** 2 * maxCellSize + all_points[leaf_1[0]]
#                         all_points = np.concatenate((all_points, l2_points), axis = 0)
#                         l2_mesh = np.diag(leaf_1)
#                         template = refineMeshTemplate + n_points_l0 + 19
#                         np.fill_diagonal(template, 0)
#                         l2_mesh += template
#                         children = l2_mesh.copy()
#                         for k, leaf_2 in enumerate(children):
#                             if hex_contains(all_points[leaf_2], point0):
#                                 if targetLevel[index] > 2:
#                                     l2_mesh = np.delete(l2_mesh, k, axis=0)
#                                     l3_points = refineTemplate * 0.5 ** 3 * maxCellSize + all_points[leaf_2[0]]
#                                     all_points = np.concatenate((all_points, l3_points), axis = 0)
#                                     l3_mesh = np.diag(leaf_2)
#                                     template = refineMeshTemplate + n_points_l0 + 19 + 19
#                                     np.fill_diagonal(template, 0)
#                                     l3_mesh += template
#                                     children = l3_mesh.copy()
#                                     for l, leaf_3 in enumerate(children):
#                                         if hex_contains(all_points[leaf_3], point0):
#                                             if targetLevel[index] > 3:
#                                                 l3_mesh = np.delete(l3_mesh, l, axis=0)
#                                                 l4_points = refineTemplate * 0.5 ** 4 * maxCellSize + all_points[leaf_3[0]]
#                                                 all_points = np.concatenate((all_points, l4_points), axis = 0)
#                                                 l4_mesh = np.diag(leaf_3)
#                                                 template = refineMeshTemplate + n_points_l0 + 19 + 19 + 19
#                                                 np.fill_diagonal(template, 0)
#                                                 l4_mesh += template
#                                                 children = l4_mesh.copy()
#                                                 for m, leaf_4 in enumerate(children):
#                                                     if hex_contains(all_points[leaf_4], point0):
#                                                         if targetLevel[index] > 4:
#                                                             l4_mesh = np.delete(l4_mesh, m, axis=0)
#                                                             l5_points = refineTemplate * 0.5 ** 5 * maxCellSize + all_points[leaf_4[0]]
#                                                             all_points = np.concatenate((all_points, l5_points), axis = 0)
#                                                             l5_mesh = np.diag(leaf_4)
#                                                             template = refineMeshTemplate + n_points_l0 + 19 + 19 + 19 + 19
#                                                             np.fill_diagonal(template, 0)
#                                                             l5_mesh += template
#                                                             children = l5_mesh.copy()
#
# l0_voxel = [("hexahedron", l0_mesh)]
# mesh = meshio.Mesh(points = l0_points, cells = l0_voxel)
# meshio.write("./l0_mesh.vtk", mesh, file_format="vtk", binary=False)
#
# l1_voxel = [("hexahedron", l1_mesh)]
# l1_mesh = meshio.Mesh(points = all_points, cells = l1_voxel)
# meshio.write("./l1_mesh.vtk", l1_mesh, file_format="vtk", binary=False)
#
# l2_voxel = [("hexahedron", l2_mesh)]
# l2_mesh = meshio.Mesh(points = all_points, cells = l2_voxel)
# meshio.write("./l2_mesh.vtk", l2_mesh, file_format="vtk", binary=False)
#
# l3_voxel = [("hexahedron", l3_mesh)]
# l3_mesh = meshio.Mesh(points = all_points, cells = l3_voxel)
# meshio.write("./l3_mesh.vtk", l3_mesh, file_format="vtk", binary=False)
#
# l4_voxel = [("hexahedron", l4_mesh)]
# l4_mesh = meshio.Mesh(points = all_points, cells = l4_voxel)
# meshio.write("./l4_mesh.vtk", l4_mesh, file_format="vtk", binary=False)
#
# l5_voxel = [("hexahedron", l5_mesh)]
# l5_mesh = meshio.Mesh(points = all_points, cells = l5_voxel)
# meshio.write("./l5_mesh.vtk", l5_mesh, file_format="vtk", binary=False)
#
# all_voxel = l0_voxel.copy()
# all_voxel.extend(l1_voxel)
# all_voxel.extend(l2_voxel)
# all_voxel.extend(l3_voxel)
# all_voxel.extend(l4_voxel)
# all_voxel.extend(l5_voxel)
# all_mesh = meshio.Mesh(points = all_points, cells = all_voxel)
# meshio.write("./all_mesh.vtk", all_mesh, file_format="vtk", binary=False)
# meshio.write("./all_mesh.msh", all_mesh, file_format="gmsh22", binary=False)
