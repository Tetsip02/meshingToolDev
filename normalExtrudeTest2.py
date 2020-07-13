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



# surfaceMesh = meshio.read("PWsurface.obj")
surfaceMesh = meshio.read("sphere.obj")

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
meshio.write("normalExtrudeAll2.vtk", l1_mesh, file_format="vtk", binary=False)
