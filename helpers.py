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
    ## get direct neighbours ##
    ## initialize neighbour array ##
    neighbours = np.array([], dtype=np.int)
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

def getNeighboursV3(P, triangle_cells, quad_cells):
    ## get relevant neighbours for vertex with valence 3 ##
    neighbours = np.array([], dtype=np.int)
    for cell in quad_cells:
        if P in cell:
            neighbours = np.append(neighbours, cell)
    for cell in triangle_cells:
        if P in cell:
            neighbours = np.append(neighbours, cell)
            ## get cells that share two vertices with 'cell' ##
            for cell2 in triangle_cells:
                mask = np.in1d(cell2, cell) # check if elements in arg1 are present in arg2, returns vector of length arg1
                if np.sum(mask) == 2:
                    neighbours = np.append(neighbours, cell2)
            for cell2 in quad_cells:
                mask = np.in1d(cell2, cell)
                if np.sum(mask) == 2:
                    neighbours = np.append(neighbours, cell2)
    ## remove duplicate points ##
    neighbours = np.unique(neighbours)
    ## remove P itself from list of neighbours ##
    Pindex = np.where(neighbours == P)
    neighbours = np.delete(neighbours, Pindex)

    return neighbours

def getNeighboursV4(P, triangle_cells, quad_cells):
    ## get direct neighbours and diagonals for vertex with valence 4 ##
    directNeighbours = np.array([], dtype=np.int)
    diagonals = np.array([], dtype=np.int)
    for cell in quad_cells:
        if P in cell:
            cellIndex = np.where(cell == P)
            cellIndex = cellIndex[0][0]
            directNeighbours = np.append(directNeighbours, [cell[(cellIndex-1)%4], cell[(cellIndex+1)%4]])
            diagonals = np.append(diagonals, cell[(cellIndex+2)%4])
    for cell in triangle_cells:
        if P in cell:
            directNeighbours = np.append(directNeighbours, cell)
            ## get cells that share two vertices with 'cell' ##
            for cell2 in triangle_cells:
                mask = np.in1d(cell2, cell) # check if elements in arg1 are present in arg2, returns vector of length arg1
                if np.sum(mask) == 2 and P not in cell2:
                    diagonals = np.append(diagonals, cell2)
            for cell2 in quad_cells:
                mask = np.in1d(cell2, cell)
                if np.sum(mask) == 2 and P not in cell2:
                    diagonals = np.append(diagonals, cell2)
    ## remove duplicate points ##
    directNeighbours = np.unique(directNeighbours)
    diagonals = np.unique(diagonals)
    ## remove P itself from list of direct neighbours ##
    Pindex = np.where(directNeighbours == P)
    directNeighbours = np.delete(directNeighbours, Pindex)
    ## remove direct neighbours from list of diagonals ##
    diagonals = np.setdiff1d(diagonals, directNeighbours)

    return directNeighbours, diagonals

def getVertexNormal(P, surfacePoints, triangle_cells, quad_cells):
    nFace = np.array([])
    for cell in triangle_cells:
        if P in cell:
            cellPoints = np.array([surfacePoints[cell[0]], surfacePoints[cell[1]], surfacePoints[cell[2]]])
            cellNormal = getAbsNormals(cellPoints)
            if nFace.size == 0:
                nFace = cellNormal
            else:
                nFace = np.vstack((nFace, cellNormal))
    for cell in quad_cells:
        if P in cell:
            cellPoints = np.array([surfacePoints[cell[0]], surfacePoints[cell[1]], surfacePoints[cell[2]], surfacePoints[cell[3]]])
            cellNormal = getAbsNormals(cellPoints)
            if nFace.size == 0:
                nFace = cellNormal
            else:
                nFace = np.vstack((nFace, cellNormal))
    nVertex = np.sum(nFace, axis=0)
    norm = np.linalg.norm(nVertex)
    nVertex = nVertex / norm

    return nVertex
