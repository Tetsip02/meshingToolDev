import meshio
import numpy as np
# import math
from helpers import *


surface_triangles, surface_quads, surface_points = read_surface("./testGeometries/sphere.obj")
d1 = 0.1
d2 = 0.12
level1, level2, level3 = build_ref_mesh_points(surface_points, surface_triangles, surface_quads, d1, d2, 1)
nSurfacePoints = len(surface_points)
level1_triangles = surface_triangles
level1_quads = surface_quads
level2_triangles = level1_triangles + nSurfacePoints
level2_quads = level1_quads + nSurfacePoints
level3_triangles = level2_triangles + nSurfacePoints
level3_quads = level2_quads + nSurfacePoints
allPoints = np.concatenate((level1, level2, level3))
# voxels = [["triangle", triangle_cells], ["quad", quad_cells]]
voxels1 = [["wedge", np.concatenate((level1_triangles, level2_triangles),axis=1)], ["hexahedron", np.concatenate((level1_quads, level2_quads),axis=1)]]
voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)], ["hexahedron", np.concatenate((level2_quads, level3_quads),axis=1)]]
voxels1and2 = voxels1
voxels1and2.extend(voxels2)
mesh_test = meshio.Mesh(points = allPoints, cells = voxels1and2)
# meshio.write("./output/ref_mesh_test2.vtk", mesh_test, file_format="vtk", binary=False)
meshio.write("./output/ref_mesh_test2.msh", mesh_test, file_format="gmsh22", binary=False)
