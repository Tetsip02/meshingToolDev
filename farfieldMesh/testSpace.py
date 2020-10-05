# test ray tracing

import numpy as np
from shapely.geometry import Point, Polygon
import pyny3d.geoms as pyny

# # Create Point objects
# p1 = Point(24.952242, 60.1696017, 0.5)
# p2 = Point(24.976567, 60.1612500, 0.5)
#
# # Create a Polygon
# # coords = [(24.950899, 60.169158), (24.953492, 60.169158), (24.953510, 60.170104), (24.950958, 60.169990)]
# coords = np.array([(24.950899, 60.169158, 0), (24.953492, 60.169158, 0), (24.953510, 60.170104, 0), (24.950958, 60.169990, 0), (24.950899, 60.169158, 1), (24.953492, 60.169158, 1), (24.953510, 60.170104, 1), (24.950958, 60.169990, 1)])
# poly = Polygon(coords)
#
# # Check if p1 is within the polygon using the within function
# print(p1.within(poly))
#
# # Check if p2 is within the polygon
# print(p2.within(poly))
#
# # Does polygon contain p1?
# print(poly.contains(p1))
#
# # Does polygon contain p2?
# print(poly.contains(p2))

# poly1 = pyny.Polygon(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]))
# poly2 = pyny.Polygon(np.array([[0, 0, 3], [0.5, 0, 3], [0.5, 0.5, 3], [0, 0.5, 3]]))
# polyhedron = pyny.Polyhedron.by_two_polygons(poly1, poly2)
# polyhedron.plot('b')

plane1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
point1 = np.array([20000000, 30000, 0.000000001])

planeN = np.cross(plane1[1] - plane1[0], plane1[2] - plane1[0])

pointDash = np.dot(point1 - np.array([0, 0, 0]), planeN)
# print(plane1[1])
# print(plane1.shape)
print(planeN)
print(pointDash)
