#!/usr/bin/env python
# coding: utf-8

# # pyoctree introduction
# ---
# Requirements:
# * pyoctree
# * vtk >= 6.2.0

# In[8]:


# Imports
from __future__ import print_function
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sys, vtk
sys.path.append('../')
import pyoctree
from pyoctree import pyoctree as ot


# In[1]:


print('pyoctree version = ', pyoctree.__version__)
print('vtk version = ', vtk.vtkVersion.GetVTKVersion())


# ## Load 3D model geometry (stl file)

# In[2]:


# Read in stl file using vtk
reader = vtk.vtkSTLReader()
reader.SetFileName("knot.stl")
reader.MergingOn()
reader.Update()
stl = reader.GetOutput()
print("Number of points    = %d" % stl.GetNumberOfPoints())
print("Number of triangles = %d" % stl.GetNumberOfCells())


# In[3]:


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


# In[4]:


# Show format of pointCoords
pointCoords


# In[5]:


# Show format of connectivity
connectivity


# ## Generate octree

# In[6]:


# Create octree structure containing stl poly mesh
tree = ot.PyOctree(pointCoords,connectivity)


# In[7]:


# Print out basic Octree data
print("Size of Octree               = %.3fmm" % tree.root.size)
print("Number of Octnodes in Octree = %d" % tree.getNumberOfNodes())
print("Number of polys in Octree    = %d" % tree.numPolys)


# ## Find intersections between 3D object and ray

# In[9]:


# Create a list containing a single ray
xs,xe,ys,ye,zs,ze = stl.GetBounds()
x = 0.5*np.mean([xs,xe])
y = np.mean([ys,ye])
rayPointList = np.array([[[x,y,zs],[x,y,ze]]],dtype=np.float32)


# In[10]:


# Find if an intersection occurred
for i in tree.rayIntersections(rayPointList):
    print(i==1)


# In[11]:


# Get intersection points for a single ray
ray = rayPointList[0]
for i in tree.rayIntersection(ray):
    print('Intersected tri = %d,' % i.triLabel, 'Intersection coords = [%.2f, %.2f, %.2f]' % tuple(i.p), ',  Parametric. dist. along ray = %.2f' % i.s)


# In[12]:


# Get list of intersected triangles
triLabelList = [i.triLabel for i in tree.rayIntersection(ray)]
triLabelList


# In[13]:


# Get tris
triList = [tree.polyList[i] for i in triLabelList]
triList


# In[14]:


# Convert ray from start/end points into unit vector
from numpy.linalg import norm
rayVect = ray[1]-ray[0]
rayVect /= norm(rayVect)
rayVect


# In[15]:


# Find if tri face normal is in the same direction as the ray
for tri in triList:
    if np.dot(tri.N,rayVect)>0:
        print("Tri %d face normal is in the same direction as ray i.e. This is an exit point." % tri.label)
    else:
        print("Tri %d face normal is in the opposite direction as the ray i.e. This is an entry point" % tri.label)


# In[ ]:




