# test ray tracing

import numpy as np
# from shapely.geometry import Point, Polygon
# import pyny3d.geoms as pyny

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


#######################################################

AABB = np.array([[1, 1, 1], [2, 1, 1], [2, 2, 1], [1, 2, 1], [1, 1, 2], [2, 1, 2], [2, 2, 2], [1, 2, 2]])
triangle = np.array([[2.1, 2.1, 1], [3, 3, 1.2], [2.8, 2.9, 1.8]])
# triangle = np.array([[1.9, 1.9, 1], [3, 3, 1.2], [2.8, 2.9, 1.8]])

t = triInBox(AABB, triangle)
print(t)

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
    print("False, no overlap") # no overlap
else:
    print("True, possibly overlap") # possibly overlap

# test 2, a10
p0 = triangle[0, 0] * triangle[1, 2] - triangle[0, 2] * triangle[1, 0]
p2 = triangle[2, 0] * (triangle[1, 2] - triangle[0, 2]) - triangle[2, 2] * (triangle[1, 0] - triangle[0, 0])
min_p = min(p0, p2)
max_p = max(p0, p2)
r = (np.abs(triangle[1, 2] - triangle[0, 2]) + np.abs(triangle[1, 0] - triangle[0, 0])) * h
if r < min_p or -r > max_p:
    print("False, no overlap") # no overlap
else:
    print("True, possibly overlap") # possibly overlap

# test 3, a20
p0 = triangle[0, 1] * triangle[1, 0] - triangle[0, 0] * triangle[1, 1]
p2 = triangle[2, 1] * (triangle[1, 0] - triangle[0, 0]) - triangle[2, 0] * (triangle[1, 1] - triangle[0, 1])
min_p = min(p0, p2)
max_p = max(p0, p2)
r = (np.abs(triangle[1, 1] - triangle[0, 1]) + np.abs(triangle[1, 0] - triangle[0, 0])) * h
if r < min_p or -r > max_p:
    print("False, no overlap") # no overlap
else:
    print("True, possibly overlap") # possibly overlap

# test 4, a01
p0 = triangle[0, 2] * (triangle[2, 1] - triangle[1, 1]) - triangle[0, 1] * (triangle[2, 2] - triangle[1, 2])
p1 = triangle[1, 2] * triangle[2, 1] - triangle[1, 1] * triangle[2, 2]
min_p = min(p0, p1)
max_p = max(p0, p1)
r = (np.abs(triangle[2, 2] - triangle[1, 2]) + np.abs(triangle[2, 1] - triangle[1, 1])) * h
if r < min_p or -r > max_p:
    print("False, no overlap") # no overlap
else:
    print("True, possibly overlap") # possibly overlap

# test 5, a11
p0 = triangle[0, 0] * (triangle[2, 2] - triangle[1, 2]) - triangle[0, 2] * (triangle[2, 0] - triangle[1, 0])
p1 = triangle[1, 0] * triangle[2, 2] - triangle[1, 2] * triangle[2, 0]
min_p = min(p0, p1)
max_p = max(p0, p1)
r = (np.abs(triangle[2, 2] - triangle[1, 2]) + np.abs(triangle[2, 0] - triangle[1, 0])) * h
if r < min_p or -r > max_p:
    print("False, no overlap") # no overlap
else:
    print("True, possibly overlap") # possibly overlap

# test 6, a21
p0 = triangle[0, 1] * (triangle[2, 0] - triangle[1, 0]) - triangle[0, 0] * (triangle[2, 1] - triangle[1, 1])
p1 = triangle[1, 1] * triangle[2, 0] - triangle[1, 0] * triangle[2, 1]
min_p = min(p0, p1)
max_p = max(p0, p1)
r = (np.abs(triangle[2, 1] - triangle[1, 1]) + np.abs(triangle[2, 0] - triangle[1, 0])) * h
if r < min_p or -r > max_p:
    print("False, no overlap") # no overlap
else:
    print("True, possibly overlap") # possibly overlap

# test 7, a02
p0 = triangle[0, 1] * triangle[2, 2] - triangle[0, 2] * triangle[2, 1]
p1 = triangle[1, 2] * (triangle[0, 1] - triangle[2, 1]) - triangle[1, 1] * (triangle[0, 2] - triangle[2, 2])
min_p = min(p0, p1)
max_p = max(p0, p1)
r = (np.abs(triangle[0, 2] - triangle[2, 2]) + np.abs(triangle[0, 1] - triangle[2, 1])) * h
if r < min_p or -r > max_p:
    print("False, no overlap") # no overlap
else:
    print("True, possibly overlap") # possibly overlap

# test 8, a12
p0 = triangle[0, 2] * triangle[2, 0] - triangle[0, 0] * triangle[2, 2]
p1 = triangle[1, 0] * (triangle[0, 2] - triangle[2, 2]) - triangle[1, 2] * (triangle[0, 0] - triangle[2, 0])
min_p = min(p0, p1)
max_p = max(p0, p1)
r = (np.abs(triangle[0, 2] - triangle[2, 2]) + np.abs(triangle[0, 0] - triangle[2, 0])) * h
if r < min_p or -r > max_p:
    print("False, no overlap") # no overlap
else:
    print("True, possibly overlap") # possibly overlap

# test 9, a22
p0 = triangle[0, 0] * triangle[2, 1] - triangle[0, 1] * triangle[2, 0]
p1 = triangle[1, 1] * (triangle[0, 0] - triangle[2, 0]) - triangle[1, 0] * (triangle[0, 1] - triangle[2, 1])
min_p = min(p0, p1)
max_p = max(p0, p1)
r = (np.abs(triangle[0, 1] - triangle[2, 1]) + np.abs(triangle[0, 0] - triangle[2, 0])) * h
if r < min_p or -r > max_p:
    print("False, no overlap") # no overlap
else:
    print("True, possibly overlap") # possibly overlap

# test 10, x-plane
min_e = min(triangle[:, 0])
max_e = max(triangle[:, 0])
if h < min_e or -h > max_e:
    print("False, no overlap") # no overlap
else:
    print("True, possibly overlap") # possibly overlap

# test 11, y-plane
min_e = min(triangle[:, 1])
max_e = max(triangle[:, 1])
if h < min_e or -h > max_e:
    print("False, no overlap") # no overlap
else:
    print("True, possibly overlap") # possibly overlap

# test 12, z-plane
min_e = min(triangle[:, 2])
max_e = max(triangle[:, 2])
if h < min_e or -h > max_e:
    print("False, no overlap") # no overlap
else:
    print("True, possibly overlap") # possibly overlap

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
    print("N:False, no overlap") # no overlap
elif res2 > 0:
    print("N:True, possibly overlap") # possibly overlap
else:
    print("N:False, no overlap") # no overlap

# def triBox(AABB, triangle):
#     h = AABB[]
