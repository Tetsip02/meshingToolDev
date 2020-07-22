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

def getAbsNormals(shape):
    n = np.cross(shape[1,:] - shape[0,:], shape[2,:] - shape[0,:])

    return n

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
