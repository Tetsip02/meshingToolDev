import meshio
import numpy as np
# from getNormals import getNormals

def getNormals(shape):
    n = np.cross(shape[1,:] - shape[0,:], shape[2,:] - shape[0,:])
    norm = np.linalg.norm(n)
    if norm==0:
        raise ValueError('zero norm')
    else:
        normalised = n/norm

    return normalised

def getNeighbours(P, triangle_cells, quad_cells):
    ## initialize neighbour array ##
    neighbours = np.array([])
    ## add points of triangle cells that include P ##
    for cell in triangle_cells:
        if P in cell:
            neighbours = np.append(neighbours, cell)
    ## add points of quad cells that include P, except point located opposite of P, and P itself ##
    for cell in quad_cells:
        if P in cell:
            cellIndex = np.where(cell == P)
            cellIndex = cellIndex[0][0]
            neighbours = np.append(neighbours, [cell[(cellIndex-1)%4], cell[(cellIndex+1)%4]])
    ## remove duplicate points ##
    neighbours = np.unique(neighbours)
    ## remove P itself from list of neighbours ##
    Pindex = np.where(neighbours == P)
    neighbours = np.delete(neighbours, Pindex)

    return neighbours

def extrudePoints(surfacePoints, triangle_cells, quad_cells, t):
    ##initialize layer 1 points##
    l1_points = np.empty(surfacePoints.shape)
    ##get layer 1 points##
    for cell in triangle_cells:
        cellPoints = np.array([surfacePoints[cell[0]], surfacePoints[cell[1]], surfacePoints[cell[2]]])
        cellNormal = getNormals(cellPoints)
        l1_cellPoints = cellPoints + t * cellNormal
        l1_points[cell[0]] = l1_cellPoints[0]
        l1_points[cell[1]] = l1_cellPoints[1]
        l1_points[cell[2]] = l1_cellPoints[2]
    for cell in quad_cells:
        cellPoints = np.array([surfacePoints[cell[0]], surfacePoints[cell[1]], surfacePoints[cell[2]], surfacePoints[cell[3]]])
        cellNormal = getNormals(cellPoints)
        l1_cellPoints = cellPoints + t * cellNormal
        l1_points[cell[0]] = l1_cellPoints[0]
        l1_points[cell[1]] = l1_cellPoints[1]
        l1_points[cell[2]] = l1_cellPoints[2]
        l1_points[cell[3]] = l1_cellPoints[3]

    return l1_points

# def getNormals2(shape):
#     n = np.cross(shape[1,:] - shape[0,:], shape[2,:] - shape[0,:])
#     norm = np.linalg.norm(n)
#     if norm==0:
#         raise ValueError('zero norm')
#     else:
#         normalised = n/norm
#
#     return n


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


##get layer 1 points##
l1_points = extrudePoints(surfacePoints, triangle_cells, quad_cells, t)

##assemble voxel points##
l1_voxelPoints = np.concatenate((surfacePoints, l1_points))

##assemble layer 1 cells##
nSurfacePoints = len(surfacePoints)
l1_triangle_cells = triangle_cells + nSurfacePoints
l1_quad_cells = quad_cells + nSurfacePoints

##assemble voxels##
l1_voxels = [["wedge", np.concatenate((triangle_cells, l1_triangle_cells),axis=1)], ["hexahedron", np.concatenate((quad_cells, l1_quad_cells),axis=1)]]

##export mesh##
l1_mesh = meshio.Mesh(points = l1_voxelPoints, cells = l1_voxels)
meshio.write("./output/normalExtrudeAll2.vtk", l1_mesh, file_format="vtk", binary=False)
