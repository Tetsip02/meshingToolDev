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

# fix 1: change orientation
level2_triangles_fix = level2_triangles.copy()
level2_triangles_fix[:,[0, 2]] = level2_triangles_fix[:,[2, 0]]
# level2_triangles_fix[:, 0], level2_triangles_fix[:, 2] = level2_triangles_fix[:, 2], level2_triangles_fix[:, 0].copy()
# level2_triangles_fix.T[[0, 2]] = level2_triangles_fix.T[[2, 0]]
print(level2_triangles[1:10, :])
print(level2_triangles_fix[1:10, :])
# my_array[:,[0, 1]] = my_array[:,[1, 0]] #switch column 0 with column 1


level2_quads_fix = level2_quads.copy()
level2_quads_fix[:,[0, 3]] = level2_quads_fix[:,[3, 0]]
level2_quads_fix[:,[1, 2]] = level2_quads_fix[:,[2, 1]]

voxels1_fix = [["wedge", np.concatenate((level1_triangles, level2_triangles_fix),axis=1)], ["hexahedron", np.concatenate((level1_quads, level2_quads_fix),axis=1)]]
mesh_test4 = meshio.Mesh(points = allPoints, cells = voxels1_fix)
meshio.write("./output/ref_mesh_test4.msh", mesh_test4, file_format="gmsh22", binary=False)
