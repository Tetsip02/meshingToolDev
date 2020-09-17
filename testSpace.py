import meshio
import numpy as np
from helpers import *
from sympy import *
import math

# print(sorted(pts, key=clockwiseangle_and_distance))
neigh = np.array([[1.191133e-01, -2.917435e-17, -9.928807e-01], [1.781393e-01, -4.363155e-17, -9.840053e-01], [0.08214438, -0.05612776, -0.9950387], [0.2052759, -0.05346072, -0.9772429], [0.1081761, -0.1180818, -0.987094], [0.1728758, -0.1143321, -0.9782853]])
# 3 [ 1.191133e-01 -2.917435e-17 -9.928807e-01]
# 4 [ 1.781393e-01 -4.363155e-17 -9.840053e-01]
# 135 [ 0.08214438 -0.05612776 -0.9950387 ]
# 136 [ 0.2052759  -0.05346072 -0.9772429 ]
# 280 [ 0.1081761 -0.1180818 -0.987094 ]
# 253 [ 0.1728758 -0.1143321 -0.9782853]
# correct order 3, 135, 280, 253, 136, 4
# neigh = 3, 4, 135, 136, 280, 253

centroid = np.array([0.1445386, -0.05449224, -0.9879976]) # vertex in question
plane = np.array([0.1436343578644257, -0.0590173541520779, -0.987869486900857]) #vertex normal

neighInd = [3, 4, 135, 136, 280, 253] # vertex neighbour indices
# neighInd = np.array([3, 4, 135, 136, 280, 253])

neigh_proj = neigh - np.dot(neigh - centroid, plane)[:, None] * plane # vertex neighbours projected onto plane

start = neigh_proj[0] # start ordering from neighbour with index 0
orient = np.cross(start, neigh_proj[1:])
orient = orient / np.linalg.norm(orient, axis = 1)[: , None]
# print("orient", orient)
orient = np.dot(orient, plane) # positive if point is in clockwise direction, negative if in CCW direction
orient /= np.abs(orient)
# print("orient", orient)

vector = neigh_proj - centroid
vector = vector / np.linalg.norm(vector, axis = 1)[: , None]

refvec = vector[0] # vector from which angles are measured

# angle = neigh_proj[1:] - centroid
# angle = angle / np.linalg.norm(angle, axis = 1)[: , None]
angle = np.dot(vector[1:], refvec)
angle = np.arccos(angle)
# print("angle", angle)
for i, orient in enumerate(orient):
    if orient > 0:
        angle[i] *= 180 / np.pi
    elif orient < 0:
        angle[i] = (2 * np.pi - angle[i]) * (180 / np.pi)
print(angle)
# print(np.sort(angle))
# print(np.sort(neighInd[1:], order = angle))
# comb = np.array([neighInd[1:], angle])
# print(comb)
# print(comb.shape)
# print(np.sort(comb))
key = np.argsort(angle)
key += 1
print(key)
neighInd = np.array([3, 4, 135, 136, 280, 253])
neighInd_sorted = neighInd.copy()
# neighInd_sorted = np.array(neighInd)
# print(neighInd[[0, 1]])
neighInd_sorted[[1, 2, 3, 4, 5]] = neighInd_sorted[[1, 3, 5, 4, 2]]
print(neighInd_sorted)
print(neighInd_sorted.shape)
print("pause")
print("")
# for i, (orient, angle) in enumerate(zip(orient[1:], angle)):
#     if orient > 0:
#         angle[0] =
    # print(i)
    # print(orient)
    # print(angle)

# P4_angle = P4_proj - P5
# P4_angle = P4_angle / np.linalg.norm(P4_angle)
# P4_angle = np.dot(P4_angle, refvec)
# P4_angle = np.arccos(P4_angle)
# print("P4_angle", (P4_angle * 180)/np.pi)
# P4_angle *= 180 / np.pi
# P135_angle = (2 * np.pi - P135_angle) * (180 / np.pi)

###############################################################################################
P5 = np.array([0.1445386, -0.05449224, -0.9879976])
P5n = np.array([0.1436343578644257, -0.0590173541520779, -0.987869486900857])
P3 = np.array([1.191133e-01, -2.917435e-17, -9.928807e-01])
P4 = np.array([1.781393e-01, -4.363155e-17, -9.840053e-01])
P135 = np.array([0.08214438, -0.05612776, -0.9950387])
P136 = np.array([0.2052759, -0.05346072, -0.9772429])
P280 = np.array([0.1081761, -0.1180818, -0.987094])
P253 = np.array([0.1728758, -0.1143321, -0.9782853])

P3_proj = P3 - np.dot(P3 - P5, P5n) * P5n
P4_proj = P4 - np.dot(P4 - P5, P5n) * P5n
P135_proj = P135 - np.dot(P135 - P5, P5n) * P5n
P136_proj = P136 - np.dot(P136 - P5, P5n) * P5n
P280_proj = P280 - np.dot(P280 - P5, P5n) * P5n
P253_proj = P253 - np.dot(P253 - P5, P5n) * P5n
# print(P3_proj)
# print(P4_proj)
crosstest4 = np.cross(P3_proj, P4_proj)
# print("orient4", crosstest4)
crosstest4 = crosstest4 / np.linalg.norm(crosstest4)
print("orient4", crosstest4)
print("4", np.dot(crosstest4, P5n))
crosstest135 = np.cross(P3_proj, P135_proj)
# print("orient135", crosstest135)
crosstest135 = crosstest135 / np.linalg.norm(crosstest135)
print("orient135", crosstest135)
print("135", np.dot(crosstest135, P5n))
crosstest136 = np.cross(P3_proj, P136_proj)
crosstest136 = crosstest136 / np.linalg.norm(crosstest136)
# print(crosstest136)
print("136", np.dot(crosstest136, P5n))
crosstest280 = np.cross(P3_proj, P280_proj)
crosstest280 = crosstest280 / np.linalg.norm(crosstest280)
print("280", np.dot(crosstest280, P5n))
crosstest253 = np.cross(P3_proj, P253_proj)
crosstest253 = crosstest253 / np.linalg.norm(crosstest253)
print("253", np.dot(crosstest253, P5n))

start = P3_proj
refvec = P3_proj - P5
refvec = refvec / np.linalg.norm(refvec)
# print(refvec)
P4_angle = P4_proj - P5
P4_angle = P4_angle / np.linalg.norm(P4_angle)
# print(np.cross(P4_angle, refvec))
# diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
P4_angle = np.dot(P4_angle, refvec)
# print(P4_angle)
P4_angle = np.arccos(P4_angle)
print("P4_angle", (P4_angle * 180)/np.pi)
P135_angle = P135_proj - P5
P135_angle = P135_angle / np.linalg.norm(P135_angle)
P135_angle = np.dot(P135_angle, refvec)
# print(P135_angle)
P135_angle = np.arccos(P135_angle)
print("P135_angle", (P135_angle * 180)/np.pi)
P136_angle = P136_proj - P5
P136_angle = P136_angle / np.linalg.norm(P136_angle)
P136_angle = np.dot(P136_angle, refvec)
# print(P136_angle)
P136_angle = np.arccos(P136_angle)
print("P136_angle", (P136_angle * 180)/np.pi)
P280_angle = P280_proj - P5
P280_angle = P280_angle / np.linalg.norm(P280_angle)
P280_angle = np.dot(P280_angle, refvec)
# print(P280_angle)
P280_angle = np.arccos(P280_angle)
print("P280_angle", (P280_angle * 180)/np.pi)
P253_angle = P253_proj - P5
P253_angle = P253_angle / np.linalg.norm(P253_angle)
P253_angle = np.dot(P253_angle, refvec)
# print(P253_angle)
P253_angle = np.arccos(P253_angle)
print("P253_angle", (P253_angle * 180)/np.pi)
print("correct order: 135, 280, 253, 136, 4")

P4_angle *= 180 / np.pi
P135_angle = (2 * np.pi - P135_angle) * (180 / np.pi)
P136_angle *= 180 / np.pi
P280_angle = (2 * np.pi - P280_angle) * (180 / np.pi)
P253_angle *= 180 / np.pi
print("4", P4_angle)
print("135", P135_angle)
print("136", P136_angle)
print("280", P280_angle)
print("253", P253_angle)

####################################################################################

# z = np.array([0, 0, 1])
# rot = np.eye(3) * (z / refvec)
# # rot2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
# print(rot)
# print(np.dot(refvec, rot))
# # print(np.dot(P3_proj, rot))
# test1 = P3_proj - P5
# test2 = np.dot(test1, rot)
# print(test2)
# print(P5)

####################################
# neigh_proj = neigh - np.dot(neigh - P5, refvec) * refvec
# vector = neigh - P5
# dist = np.dot(vector, refvec)
# diff = dist[:, None] * refvec
# print(neigh_proj)
# print(dist)
# print(vector)
# print(dist)
# print(diff)
# neigh_proj = neigh - diff

# #################################################
# neigh_proj = neigh - np.dot(neigh - P5, P5n)[:, None] * P5n
# print(neigh_proj)
# start = neigh_proj[0]
# print(start)
# refvec = start - P5
# refvec = refvec / np.linalg.norm(refvec)
# print(refvec)
# vector = neigh_proj - start
# print(vector)
# vector[1:] = vector[1:] / np.linalg.norm(vector[1:], axis = 1)[: , None]
# print(vector)
# dot = np.dot(vector, refvec)
# print(dot)
# angle = np.arccos(dot)
# print( 90 - ((angle*180) / np.pi) )
# ######################################

# P3_proj = P3 - np.dot(P3, refvec) * refvec
# print(P3_proj)
# diffprod = refvec x vector_normalized
# vector_norm = vector / np.linalg.norm(vector)
# print(vector)
# print(vector_norm)


# def clockwiseangle_and_distance(point):
#     # Vector between point and the origin: v = p - o
#     vector = [point[0]-origin[0], point[1]-origin[1]]
#     # Length of vector: ||v||
#     lenvector = math.hypot(vector[0], vector[1])
#     # If length is zero there is no angle
#     if lenvector == 0:
#         return -math.pi, 0
#     # Normalize vector: v/||v||
#     normalized = [vector[0]/lenvector, vector[1]/lenvector]
#     dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
#     diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
#     angle = math.atan2(diffprod, dotprod)
#     # Negative angles represent counter-clockwise angles so we need to subtract them
#     # from 2*pi (360 degrees)
#     if angle < 0:
#         return 2*math.pi+angle, lenvector
#     # I return first the angle because that's the primary sorting criterium
#     # but if two vectors have the same angle then the shorter distance should come first.
#     return angle, lenvector
# print(sorted(neigh, key=clockwiseangle_and_distance))
# r_P = r_O + t_1*e_1 + t_2*e_2 + s*n
