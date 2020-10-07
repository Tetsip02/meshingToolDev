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

newPointsTemplate = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [1, 2, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 1], [1, 1, 1], [2, 1, 1], [0, 2, 1], [1, 2, 1], [2, 2, 1], [1, 0, 2], [0, 1, 2], [1, 1, 2], [2, 1, 2], [1, 2, 2]])

newLevelTemplate = np.array([[0, 0, 2, 1, 5, 6, 9, 8], [0, 0, 3, 2, 6, 7, 10, 9], [2, 3, 0, 4, 9, 10, 13, 12], [1, 2, 4, 0, 8, 9, 12, 11], [5, 6, 9, 8, 0, 14, 16, 15], [6, 7, 10, 9, 14, 0, 17, 16], [9, 10, 13, 12, 16, 17, 0, 18], [8, 9, 12, 11, 15, 16, 18, 0]])


level = 0
# level0_targetPoints = []
# level0_targetCells = []
# # level0_targetPointsTemp = []
# for i, leaf in enumerate(z_hexas):
#     tempList = []
#     for j, target in enumerate(faceCentroids):
#         if hex_contains(l0_points[leaf], target):
#             tempList.append(j)
#     level0_targetPoints.append(tempList)
#     if len(tempList) != 0 and np.max(targetLevel[tempList]) > level and np.max(targetLevel[tempList]) < maxLevel:
#         level0_targetCells.append(i)
# with open("./level0_targetPoints.txt", "wb") as fp:   #Pickling
#     pickle.dump(level0_targetPoints, fp)
# with open("./level0_targetCells.txt", "wb") as fp:   #Pickling
#     pickle.dump(level0_targetCells, fp)
with open("./level0_targetPoints.txt", "rb") as fp:   # Unpickling
    level0_targetPoints = pickle.load(fp)
with open("./level0_targetCells.txt", "rb") as fp:   # Unpickling
    level0_targetCells = pickle.load(fp)

all_points = l0_points.copy()
level0_mesh = z_hexas.copy()

level = 1
# level1_points = [] # contains corner points as well
level1_mesh = []
level1_targetPoints = [] # length = number of level1 cells, each entry contains list of target points in side each level1 cell
level1_targetCells = [] # indices in level1_mesh of cells that need refining
for i, targetCell in enumerate(level0_targetCells):
    # assemble refined cells
    new_level = np.diag(level0_mesh[targetCell])
    template = newLevelTemplate + all_points.shape[0] #- 19
    np.fill_diagonal(template, 0)
    new_level += template # [8 x 8]
    level1_mesh.extend(new_level)
    # compute new points of children (19 new points per refinement)
    new_points = newPointsTemplate * 0.5 ** level * maxCellSize + all_points[level0_mesh[targetCell][0]]
    all_points = np.concatenate((all_points, new_points), axis = 0)
    # compute targetCells and targetPoints among Level1 cells:
    for j, child in enumerate(new_level):
        tempList = []
        for targetPoint in level0_targetPoints[targetCell]:
            if hex_contains(all_points[child], faceCentroids[targetPoint]):
                tempList.append(targetPoint)
        level1_targetPoints.append(tempList)
        if len(tempList) != 0 and np.max(targetLevel[tempList]) > level and np.max(targetLevel[tempList]) < maxLevel:
            level1_targetCells.append((i * 8) + j)

level1_mesh = np.array(level1_mesh)

# ##############
# temp_voxels = [("hexahedron", level1_mesh)]
# tempMesh = meshio.Mesh(points = all_points, cells = temp_voxels)
# meshio.write("./test.vtk", tempMesh, file_format="vtk", binary=False)
# ##############

#create refined mesh
level0_mesh_reduced = np.delete(level0_mesh, level0_targetCells, axis=0)
refined_mesh = np.concatenate((level0_mesh_reduced, level1_mesh), axis = 0)
all_voxels = [("hexahedron", refined_mesh)]
level1_refinement = meshio.Mesh(points = all_points, cells = all_voxels)
meshio.write("./level1_refinement.vtk", level1_refinement, file_format="vtk", binary=False)
meshio.write("./level1_refinement.msh", level1_refinement, file_format="gmsh22", binary=False)

level = 2
level2_mesh = []
level2_targetPoints = [] # length = number of level2 cells, each entry contains list of target points in side each level2 cell
level2_targetCells = [] # indices in level2_mesh of cells that need refining
for i, targetCell in enumerate(level1_targetCells):
    # assemble refined cells
    new_level = np.diag(level1_mesh[targetCell])
    template = newLevelTemplate + all_points.shape[0] #- 19
    np.fill_diagonal(template, 0)
    new_level += template # [8 x 8]
    level2_mesh.extend(new_level)
    # compute new points of children (19 new points per refinement)
    new_points = newPointsTemplate * 0.5 ** level * maxCellSize + all_points[level1_mesh[targetCell][0]]
    all_points = np.concatenate((all_points, new_points), axis = 0)
    # compute targetCells and targetPoints among Level1 cells:
    for j, child in enumerate(new_level):
        tempList = []
        for targetPoint in level1_targetPoints[targetCell]:
            if hex_contains(all_points[child], faceCentroids[targetPoint]):
                tempList.append(targetPoint)
        level2_targetPoints.append(tempList)
        if len(tempList) != 0 and np.max(targetLevel[tempList]) > level and np.max(targetLevel[tempList]) < maxLevel:
            level2_targetCells.append((i * 8) + j)
level2_mesh = np.array(level2_mesh)

#create refined mesh
level1_mesh_reduced = np.delete(level1_mesh, level1_targetCells, axis=0)
refined_mesh = np.concatenate((level0_mesh_reduced, level1_mesh_reduced, level2_mesh), axis = 0)
all_voxels = [("hexahedron", refined_mesh)]
level2_refinement = meshio.Mesh(points = all_points, cells = all_voxels)
meshio.write("./level2_refinement.vtk", level2_refinement, file_format="vtk", binary=False)
meshio.write("./level2_refinement.msh", level2_refinement, file_format="gmsh22", binary=False)

# ##################################################################################
# transLayers = 1
# level2_midpoints = all_points[level2_mesh[:, 0]] + maxCellSize * 0.5 ** (level + 1)
# rInf = transLayers * maxCellSize * 0.5 ** (level - 1)
# ###################
# # l0_leaf_targets = []
# # for midpoint in level2_midpoints:
# #     distToMidpoint = np.sum((all_points - midpoint) ** 2, axis = 1) ** 0.5
# #     mask = np.array(np.where(distToMidpoint < rInf)).ravel() # indices of points inside area of influence
# #     for i, leaf in enumerate(level0_mesh_reduced):
# #         if np.any(np.in1d(leaf, mask)):
# #             l0_leaf_targets.append(i)
# # l0_leaf_targets = np.array(l0_leaf_targets)
# # l0_leaf_targets = np.unique(l0_leaf_targets, axis = 0)
# # with open("./l0_leaf_targets.txt", "wb") as fp:   #Pickling
# #     pickle.dump(l0_leaf_targets, fp)
# with open("./l0_leaf_targets.txt", "rb") as fp:   # Unpickling
#     l0_leaf_targets = pickle.load(fp)
# ########################################
#
# # refine trans targets
# l1_trans_mesh = []
# for i, trans_target in enumerate(l0_leaf_targets):
#     # assemble refined cells
#     new_level = np.diag(level0_mesh_reduced[trans_target])
#     template = newLevelTemplate + all_points.shape[0]
#     np.fill_diagonal(template, 0)
#     new_level += template # [8 x 8]
#     l1_trans_mesh.extend(new_level)
#     # compute new points of children (19 new points per refinement)
#     new_points = newPointsTemplate * 0.5 ** (level - 1) * maxCellSize + all_points[level0_mesh_reduced[trans_target][0]]
#     all_points = np.concatenate((all_points, new_points), axis = 0)
# l1_trans_mesh = np.array(l1_trans_mesh)
#
# trans_voxels2 = [("hexahedron", l1_trans_mesh)]
# trans_targets2 = meshio.Mesh(points = all_points, cells = trans_voxels2)
# meshio.write("./trans_targets2.vtk", trans_targets2, file_format="vtk", binary=False)
#
# #create refined mesh
# level0_mesh_reduced2 = np.delete(level0_mesh_reduced, l0_leaf_targets, axis=0)
# refined_mesh = np.concatenate((level0_mesh_reduced2, l1_trans_mesh, level1_mesh_reduced, level2_mesh), axis = 0)
# all_voxels = [("hexahedron", refined_mesh)]
# level2_refinement2 = meshio.Mesh(points = all_points, cells = all_voxels)
# meshio.write("./level2_refinement_withTransition.vtk", level2_refinement2, file_format="vtk", binary=False)
# meshio.write("./level2_refinement_withTransition.msh", level2_refinement2, file_format="gmsh22", binary=False)
# ############################################################################################

###################################################

level = 3
level3_mesh = []
level3_targetPoints = []
level3_targetCells = []
for i, targetCell in enumerate(level2_targetCells):
    # assemble refined cells
    new_level = np.diag(level2_mesh[targetCell])
    template = newLevelTemplate + all_points.shape[0] #- 19
    np.fill_diagonal(template, 0)
    new_level += template # [8 x 8]
    level3_mesh.extend(new_level)
    # compute new points of children (19 new points per refinement)
    new_points = newPointsTemplate * 0.5 ** level * maxCellSize + all_points[level2_mesh[targetCell][0]]
    all_points = np.concatenate((all_points, new_points), axis = 0)
    # compute targetCells and targetPoints among Level1 cells:
    for j, child in enumerate(new_level):
        tempList = []
        for targetPoint in level2_targetPoints[targetCell]:
            if hex_contains(all_points[child], faceCentroids[targetPoint]):
                tempList.append(targetPoint)
        level3_targetPoints.append(tempList)
        if len(tempList) != 0 and np.max(targetLevel[tempList]) > level and np.max(targetLevel[tempList]) < maxLevel:
            level3_targetCells.append((i * 8) + j)
level3_mesh = np.array(level3_mesh)

#create refined mesh
level2_mesh_reduced = np.delete(level2_mesh, level2_targetCells, axis=0)
refined_mesh = np.concatenate((level0_mesh_reduced, level1_mesh_reduced, level2_mesh_reduced, level3_mesh), axis = 0)
all_voxels = [("hexahedron", refined_mesh)]
level3_refinement = meshio.Mesh(points = all_points, cells = all_voxels)
meshio.write("./level3_refinement.vtk", level3_refinement, file_format="vtk", binary=False)
meshio.write("./level3_refinement.msh", level3_refinement, file_format="gmsh22", binary=False)

level = 4
level4_mesh = []
level4_targetPoints = []
level4_targetCells = []
for i, targetCell in enumerate(level3_targetCells):
    # assemble refined cells
    new_level = np.diag(level3_mesh[targetCell])
    template = newLevelTemplate + all_points.shape[0] #- 19
    np.fill_diagonal(template, 0)
    new_level += template # [8 x 8]
    level4_mesh.extend(new_level)
    # compute new points of children (19 new points per refinement)
    new_points = newPointsTemplate * 0.5 ** level * maxCellSize + all_points[level3_mesh[targetCell][0]]
    all_points = np.concatenate((all_points, new_points), axis = 0)
    # compute targetCells and targetPoints among Level4 cells:
    for j, child in enumerate(new_level):
        tempList = []
        for targetPoint in level3_targetPoints[targetCell]:
            if hex_contains(all_points[child], faceCentroids[targetPoint]):
                tempList.append(targetPoint)
        level4_targetPoints.append(tempList)
        if len(tempList) != 0 and np.max(targetLevel[tempList]) > level and np.max(targetLevel[tempList]) < maxLevel:
            level4_targetCells.append((i * 8) + j)
level4_mesh = np.array(level4_mesh)

#create refined mesh
level3_mesh_reduced = np.delete(level3_mesh, level3_targetCells, axis=0)
refined_mesh = np.concatenate((level0_mesh_reduced, level1_mesh_reduced, level2_mesh_reduced, level3_mesh_reduced, level4_mesh), axis = 0)
all_voxels = [("hexahedron", refined_mesh)]
level4_refinement = meshio.Mesh(points = all_points, cells = all_voxels)
meshio.write("./level4_refinement.vtk", level4_refinement, file_format="vtk", binary=False)
meshio.write("./level4_refinement.msh", level4_refinement, file_format="gmsh22", binary=False)

###########################################################################
# strategy:
# set currentLevel to 4
# compute level 4 midpoints
# set rInf to: transLayers * maxCellSize * 0.5 ** (currentLevel - 1)
# cycle through level 2 cells and refine to level3 if one of its points is inside influence area
# set currentLevel to 3
# compute level 3 midpoints
# set rInf to: transLayers * maxCellSize * 0.5 ** (currentLevel - 1)
# cycle through level 1 cells and refine to level2 if one of its points is inside influence area
# set currentLevel to 2
# compute level 2 midpoints
# set rInf to: transLayers * maxCellSize * 0.5 ** (currentLevel - 1)
# cycle through level 0 cells and refine to level1 if one of its points is inside influence area
transLayers = 1
currentLevel = 4
level4_midpoints = all_points[level4_mesh[:, 0]] + maxCellSize * 0.5 ** (currentLevel + 1)
rInf = transLayers * maxCellSize * 0.5 ** (currentLevel - 1)
############
# l2_trans_targets = []
# for midpoint in level4_midpoints:
#     distToMidpoint = np.sum((all_points - midpoint) ** 2, axis = 1) ** 0.5
#     mask = np.array(np.where(distToMidpoint < rInf)).ravel() # indices of points inside area of influence
#     for i, leaf in enumerate(level2_mesh_reduced):
#         if np.any(np.in1d(leaf, mask)):
#             l2_trans_targets.append(i)
# l2_trans_targets = np.array(l2_trans_targets)
# l2_trans_targets = np.unique(l2_trans_targets, axis = 0)
# with open("./l2_trans_targets.txt", "wb") as fp:   #Pickling
#     pickle.dump(l2_trans_targets, fp)
with open("./l2_trans_targets.txt", "rb") as fp:   # Unpickling
    l2_trans_targets = pickle.load(fp)
#############
# trans_voxels = [("hexahedron", level2_mesh_reduced[l2_trans_targets])]
# trans_targets = meshio.Mesh(points = all_points, cells = trans_voxels)
# meshio.write("./trans_targets.vtk", trans_targets, file_format="vtk", binary=False)
# refine trans targets
l2_trans_mesh = []
for trans_target in l2_trans_targets:
    # assemble refined cells
    new_level = np.diag(level2_mesh_reduced[trans_target])
    template = newLevelTemplate + all_points.shape[0]
    np.fill_diagonal(template, 0)
    new_level += template # [8 x 8]
    l2_trans_mesh.extend(new_level)
    # compute new points of children (19 new points per refinement)
    new_points = newPointsTemplate * 0.5 ** (currentLevel - 1) * maxCellSize + all_points[level2_mesh_reduced[trans_target][0]]
    all_points = np.concatenate((all_points, new_points), axis = 0)
l2_trans_mesh = np.array(l2_trans_mesh)
# trans_voxels2 = [("hexahedron", l2_trans_mesh)]
# trans_targets2 = meshio.Mesh(points = all_points, cells = trans_voxels2)
# meshio.write("./trans_targets2.vtk", trans_targets2, file_format="vtk", binary=False)

currentLevel = 3
level3_midpoints = all_points[level3_mesh[:, 0]] + maxCellSize * 0.5 ** (currentLevel + 1)
rInf = transLayers * maxCellSize * 0.5 ** (currentLevel - 1)
############
# l1_trans_targets = []
# for midpoint in level3_midpoints:
#     distToMidpoint = np.sum((all_points - midpoint) ** 2, axis = 1) ** 0.5
#     mask = np.array(np.where(distToMidpoint < rInf)).ravel() # indices of points inside area of influence
#     for i, leaf in enumerate(level1_mesh_reduced):
#         if np.any(np.in1d(leaf, mask)):
#             l1_trans_targets.append(i)
# l1_trans_targets = np.array(l1_trans_targets)
# l1_trans_targets = np.unique(l1_trans_targets, axis = 0)
# with open("./l1_trans_targets.txt", "wb") as fp:   #Pickling
#     pickle.dump(l1_trans_targets, fp)
with open("./l1_trans_targets.txt", "rb") as fp:   # Unpickling
    l1_trans_targets = pickle.load(fp)
#############
print(l1_trans_targets)
trans_voxels = [("hexahedron", level1_mesh_reduced[l1_trans_targets])]
trans_targets = meshio.Mesh(points = all_points, cells = trans_voxels)
meshio.write("./trans_targets.vtk", trans_targets, file_format="vtk", binary=False)




# ##################################################################################
# script in this section build transition layer after refinement number 2
# transLayers = 1
# level2_midpoints = all_points[level2_mesh[:, 0]] + maxCellSize * 0.5 ** (level + 1)
# rInf = transLayers * maxCellSize * 0.5 ** (level - 1)
# ###################
# # l0_leaf_targets = []
# # for midpoint in level2_midpoints:
# #     distToMidpoint = np.sum((all_points - midpoint) ** 2, axis = 1) ** 0.5
# #     mask = np.array(np.where(distToMidpoint < rInf)).ravel() # indices of points inside area of influence
# #     for i, leaf in enumerate(level0_mesh_reduced):
# #         if np.any(np.in1d(leaf, mask)):
# #             l0_leaf_targets.append(i)
# # l0_leaf_targets = np.array(l0_leaf_targets)
# # l0_leaf_targets = np.unique(l0_leaf_targets, axis = 0)
# # with open("./l0_leaf_targets.txt", "wb") as fp:   #Pickling
# #     pickle.dump(l0_leaf_targets, fp)
# with open("./l0_leaf_targets.txt", "rb") as fp:   # Unpickling
#     l0_leaf_targets = pickle.load(fp)
# ########################################
#
# # refine trans targets
# l1_trans_mesh = []
# for i, trans_target in enumerate(l0_leaf_targets):
#     # assemble refined cells
#     new_level = np.diag(level0_mesh_reduced[trans_target])
#     template = newLevelTemplate + all_points.shape[0]
#     np.fill_diagonal(template, 0)
#     new_level += template # [8 x 8]
#     l1_trans_mesh.extend(new_level)
#     # compute new points of children (19 new points per refinement)
#     new_points = newPointsTemplate * 0.5 ** (level - 1) * maxCellSize + all_points[level0_mesh_reduced[trans_target][0]]
#     all_points = np.concatenate((all_points, new_points), axis = 0)
# l1_trans_mesh = np.array(l1_trans_mesh)
#
# trans_voxels2 = [("hexahedron", l1_trans_mesh)]
# trans_targets2 = meshio.Mesh(points = all_points, cells = trans_voxels2)
# meshio.write("./trans_targets2.vtk", trans_targets2, file_format="vtk", binary=False)
#
# #create refined mesh
# level0_mesh_reduced2 = np.delete(level0_mesh_reduced, l0_leaf_targets, axis=0)
# refined_mesh = np.concatenate((level0_mesh_reduced2, l1_trans_mesh, level1_mesh_reduced, level2_mesh), axis = 0)
# all_voxels = [("hexahedron", refined_mesh)]
# level2_refinement2 = meshio.Mesh(points = all_points, cells = all_voxels)
# meshio.write("./level2_refinement_withTransition.vtk", level2_refinement2, file_format="vtk", binary=False)
# meshio.write("./level2_refinement_withTransition.msh", level2_refinement2, file_format="gmsh22", binary=False)
# ############################################################################################
















# transition Layers
# transLayers = 1
# level4_midpoints = all_points[level4_mesh[:, 0]] + maxCellSize * 0.5 ** 5
# rInf = transLayers * maxCellSize * 0.5 ** 3 # 0.15m
# # midpoint = level4_midpoints[500]
# # distToMidpoint = np.sum((all_points - midpoint) ** 2, axis = 1) ** 0.5
# # mask = np.array(np.where(distToMidpoint < rInf)).ravel() # indices of points inside area of influence
# # print(mask)
# # l0_leaf_targets = []
# # for leaf in level1_mesh_reduced:
# #     if np.any(np.in1d(leaf, mask)):
# #         l0_leaf_targets.append
# # print(l0_leaf_targets)

# level3_midpoints = all_points[level3_mesh[:, 0]] + maxCellSize * 0.5 ** 4
# rInf = transLayers * maxCellSize * 0.5 ** 2 # 0.15m
# midpoint = level3_midpoints[0]
# distToMidpoint = np.sum((all_points - midpoint) ** 2, axis = 1) ** 0.5
# mask = np.array(np.where(distToMidpoint < rInf)).ravel() # indices of points inside area of influence
# print(mask)
# l0_leaf_targets = []
# for leaf in level0_mesh_reduced:
#     if np.any(np.in1d(leaf, mask)):
#         l0_leaf_targets.append
# print(l0_leaf_targets)

# l0_leaf_targets = []
# for midpoint in level4_midpoints:
#     distToMidpoint = np.sum((all_points - midpoint) ** 2, axis = 1) ** 0.5
#     mask = np.array(np.where(distToMidpoint < rInf)).ravel() # indices of points inside area of influence
#     # l0_leaf_targets = []
#     for leaf in level1_mesh_reduced:
#         # print(leaf)
#         if np.any(np.in1d(leaf, mask)):
#             l0_leaf_targets.append
# print(l0_leaf_targets)
# print(level1_mesh_reduced)

# print(mask)
# print(mask.shape)
# print("midpoint", midpoint)
# print("node", all_points[0])
# # print(distToMidpoint)
# print("dist", distToMidpoint[0])
