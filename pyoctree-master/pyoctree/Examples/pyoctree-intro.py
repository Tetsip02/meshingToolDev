#!/usr/bin/env python
# coding: utf-8

# # pyoctree introduction
# ---
# Requirements:
# * pyoctree
# * vtk >= 6.2.0

# In[1]:


# Imports
from __future__ import print_function
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sys, vtk
sys.path.append('../')
import pyoctree
from pyoctree import pyoctree as ot


# In[2]:


print('pyoctree version = ', pyoctree.__version__)
print('vtk version = ', vtk.vtkVersion.GetVTKVersion())


# ## Load 3D model geometry (stl file)

# In[3]:


# Read in stl file using vtk
reader = vtk.vtkSTLReader()
reader.SetFileName("knot.stl")
reader.MergingOn()
reader.Update()
stl = reader.GetOutput()
print("Number of points    = %d" % stl.GetNumberOfPoints())
print("Number of triangles = %d" % stl.GetNumberOfCells())


# In[4]:


# Extract polygon info from stl

# 1. Get array of point coordinates
numPoints   = stl.GetNumberOfPoints()
pointCoords = np.zeros((numPoints,3),dtype=float)
for i in range(numPoints):
    pointCoords[i,:] = stl.GetPoint(i)
    
# 2. Get polygon connectivity
numPolys     = stl.GetNumberOfCells()
connectivity = np.zeros((numPolys,3),dtype=np.int32)
for i in range(numPolys):
    atri = stl.GetCell(i)
    ids = atri.GetPointIds()
    for j in range(3):
        connectivity[i,j] = ids.GetId(j)


# In[5]:


# Show format of pointCoords
pointCoords


# In[6]:


# Show format of connectivity
connectivity


# ## Generate octree

# In[7]:


# Create octree structure containing stl poly mesh
tree = ot.PyOctree(pointCoords,connectivity)


# In[8]:


# Print out basic Octree data
print("Size of Octree               = %.3fmm" % tree.root.size)
print("Number of Octnodes in Octree = %d" % tree.getNumberOfNodes())
print("Number of polys in Octree    = %d" % tree.numPolys)


# ## Explore tree nodes

# In[9]:


# Get the root node
print(tree.root)


# In[10]:


# Get the branches of the root node, OctNode 0
tree.root.branches


# In[11]:


# Get OctNode 0-0 (first branch of the root node)
print(tree.root.branches[0])


# In[12]:


# Get the branches of OctNode 0-0 (first branch of the root node) using getNodeFromId function
tree.getNodeFromId('0-0').branches


# In[13]:


# An octnode that has no branches is a leaf
print('OctNode 0-0-1 is a leaf = ', tree.getNodeFromId('0-3-1').isLeaf)
print(tree.getNodeFromId('0-3-1'))


# ## Explore polygons in tree

# In[14]:


# Get list of first 10 polygons stored in tree
tree.polyList[0:10]


# In[15]:


# Get details of the first tri in the tree
tri = tree.polyList[0]
s = []
for i in range(3):
    s.append('[%.3f, %.3f, %.3f]' % tuple(tri.vertices[i][:3]))
s = ','.join(s)

print('Tri label = %d' % tri.label)
print('Vertex coordinates = %s' % s)
print('Face normal direction = [%.2f, %.2f, %.2f]' % tuple(tri.N))
print('Perp. distance from origin = %.3f' % tri.D)


# In[16]:


# Find the OctNode that the tri with label=0 lies within
tree.getNodesFromLabel(0)


# In[17]:


# Test if OctNode contains a particular tri label
tree.getNodeFromId('0-3-1').hasPolyLabel(0)


# ## Find intersections between 3D object and ray

# In[18]:


# Create a list containing a single ray
xs,xe,ys,ye,zs,ze = stl.GetBounds()
x = 0.5*np.mean([xs,xe])
y = np.mean([ys,ye])
rayPointList = np.array([[[x,y,zs],[x,y,ze]]],dtype=np.float32)


# In[19]:


# Find if an intersection occurred
for i in tree.rayIntersections(rayPointList):
    print(i==1)


# In[20]:


# Get intersection points for a single ray
ray = rayPointList[0]
for i in tree.rayIntersection(ray):
    print('Intersection coords = [%.2f, %.2f, %.2f]' % tuple(i.p), ',  Parametric. dist. along ray = %.2f' % i.s)


# ## Write out a vtk file of the octree structure 

# In[21]:


# Create a vtk representation of Octree
tree.getOctreeRep()


# In[ ]:




