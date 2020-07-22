import meshio
import numpy as np
from helpers import getNormals, getAbsNormals, getNeighbours, extrudePoints, getNeighboursV3

# testP1 = np.array([[0.5, 0.5, 0.5], [0.1, 0.7, 0.4], [0.2, 1.1, 0.6], [0.6, 1.2, 0.7]])
# testP2 = np.array([[0.5, 0.5, 0.5], [0.6, 1.2, 0.7], [0.9, 0.9, 0.8], [1.1, 0.6, 0.6]])
# testP3 = np.array([[0.5, 0.5, 0.5], [1.1, 0.6, 0.6], [1.2, 0.2, 0.4], [0.7, 0.0, 0.3]])
# testP4 = np.array([[0.5, 0.5, 0.5], [0.7, 0.0, 0.3], [0.0, -0.1, 0.7], [0.1, 0.7, 0.4]])
# nP = getNormals2(testP4)
# print(nP)


# surfaceMesh = meshio.read("PWsurface.obj")
surfaceMesh = meshio.read("./testGeometries/sphere.obj")

##layer 1 thickness##
t = 0.1

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



## get relevant points (negihbours and also diagonals in case M < 5) ##
P1 = 10
neighbours = np.array([], dtype=np.int)
directNeighbours = np.array([], dtype=np.int)
diagonals = np.array([], dtype=np.int)
## check valence ##
v = 0
for cell in triangle_cells:
    if P1 in cell:
        v+=1
for cell in quad_cells:
    if P1 in cell:
        v+=1
if v > 4:
    neighbours = getNeighbours(P1, triangle_cells, quad_cells)
elif v == 3:
    neighbours = getNeighbours(P1, triangle_cells, quad_cells)
elif v == 4:
    ## get direct neigbours and diagonals seperately ##
    for cell in quad_cells:
        if P1 in cell:
            cellIndex = np.where(cell == P)
            cellIndex = cellIndex[0][0]
            directNeighbours = np.append(directNeighbours, [cell[(cellIndex-1)%4], cell[(cellIndex+1)%4]])
            diagonals = np.append(diagonals, cell[(cellIndex+2)%4])
    for cell in triangle_cells:
        if P1 in cell:
            directNeighbours = np.append(directNeighbours, cell)


# cell1 = np.array([0, 1, 2])
# cell2 = np.array([3, 2, 5])
# mask = np.in1d(cell1, cell2)
# print(mask)
# print(np.sum(mask))


# P10neighbours = getNeighbours(30, triangle_cells, quad_cells)
# print(P10neighbours)

#
# P1 = 10
# P1neighbours = np.array([])
# for cell in triangle_cells:
#     if P1 in cell:
#         print(cell)
#         P1neighbours = np.append(P1neighbours, cell)
#
# for cell in quad_cells:
#     # print(cell)
#     if P1 in cell:
#         print(cell)
#         cellIndex = np.where(cell == P1)
#         cellIndex = cellIndex[0][0]
#         # print(cellIndex, (cellIndex-1)%4, (cellIndex+1)%4)
#         # print(cell[cellIndex])
#         # P1neighbours = np.append(P1neighbours, cell[(cellIndex-1)%4])
#         # P1neighbours = np.append(P1neighbours, cell[(cellIndex+1)%4])
#         P1neighbours = np.append(P1neighbours, [cell[(cellIndex-1)%4], cell[(cellIndex+1)%4]])
#
# print(P1neighbours)
# P1neighbours = np.unique(P1neighbours)
# print(P1neighbours)
# P1index = np.where(P1neighbours == P1)
# # print(P1index)
# P1neighbours = np.delete(P1neighbours, P1index)
# print(P1neighbours)
# # print(P1index[0][0])
# # print(P1neighbours2)
#

#
# P10neighbours = getNeighbours(1, triangle_cells, quad_cells)
# print(P10neighbours)
#

#
# ##get layer 1 points##
# l1_points = extrudePoints(surfacePoints, triangle_cells, quad_cells, t)
#
# ##assemble voxel points##
# l1_voxelPoints = np.concatenate((surfacePoints, l1_points))
#
# ##assemble layer 1 cells##
# nSurfacePoints = len(surfacePoints)
# l1_triangle_cells = triangle_cells + nSurfacePoints
# l1_quad_cells = quad_cells + nSurfacePoints
#
# ##assemble voxels##
# l1_voxels = [["wedge", np.concatenate((triangle_cells, l1_triangle_cells),axis=1)], ["hexahedron", np.concatenate((quad_cells, l1_quad_cells),axis=1)]]
#
# ##export mesh##
# l1_mesh = meshio.Mesh(points = l1_voxelPoints, cells = l1_voxels)
# meshio.write("./output/normalExtrudeAll2.vtk", l1_mesh, file_format="vtk", binary=False)
#
