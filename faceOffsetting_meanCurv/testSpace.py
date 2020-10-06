import meshio
import numpy as np
# import math
# from ../helpers import *
# import json
import pickle
from scipy.linalg import null_space

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

# A = np.array([[-1, 3, 2], [1, 1, 0], [1, 1, 0]])
A = np.array([[ 0.04100598,  0.00242317, -0.00214077], [ 0.00242317,  0.00019063, -0.00013373], [-0.00214077, -0.00013373,  0.00017863]], dtype = np.float64)
# A = np.array([[2,3,5],[-4,2,3],[0,0,0]], dtype = np.float64)
print(A)
Z = null_space(A, rcond=1)
print(Z.shape)
print(Z)
# print(A @ null_space(A))

def centroid(vertexes):
     _x_list = [vertex [0] for vertex in vertexes]
     _y_list = [vertex [1] for vertex in vertexes]
     _z_list = [vertex [2] for vertex in vertexes]
     _len = len(vertexes)
     _x = sum(_x_list) / _len
     _y = sum(_y_list) / _len
     _z = sum(_z_list) / _len
     return(_x, _y, _z)

#determinant of matrix a
def det(a):
    return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]

#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

#dot product of vectors a and b
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#cross product of vectors a and b
def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)

#area of polygon poly
def area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

def get_face_normal(points, faces):
    normal = np.ndarray((len(faces), 3), dtype = np.float64)
    for i, face in enumerate(faces):
        coor = points[face, :]
        normal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
    # triNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])

    normal_norm = normal / np.linalg.norm(normal, axis = 1)[: , None]

    return normal_norm

surface_triangles_ini, surface_quads_ini, surface_points = read_surface("../testGeometries/sphere.obj")

print(surface_points[surface_triangles_ini[2]])

tri2 = np.array([[-0.12168533,  0.70228616, -0.70141773], [-0.04152985,  0.69270963, -0.72001989], [-0.06029736,  0.7255067,  -0.68556857]])
print(centroid(tri2))
print(area(tri2))

# print(get_face_normal(surface_points, surface_triangles_ini[2]))
a = surface_triangles_ini[2]
print(a)
coor = surface_points[a, :]
print(coor)
normal = np.cross(coor[1] - coor[0], coor[2] - coor[0])
print(normal)
# print(get_face_normal(surface_points, a))
n = get_face_normal(surface_points, surface_quads_ini)
print(n)
