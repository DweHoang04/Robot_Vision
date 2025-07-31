
# -*- coding: utf-8 -*-
"""
Neighborhoods Module for 3D Point Cloud Processing

This module provides neighborhood computation functions for 3D point clouds,
essential for local surface analysis in the Harris 3D keypoint detection pipeline.

Original source: https://github.com/rengotj/projet_NPM
Modified for LEGO Duplo bin picking system.

Functions:
- k_ring_delaunay_adaptive(): Adaptive neighborhood selection using Delaunay triangulation
- k_ring_delaunay(): Fixed k-ring neighborhood selection using Delaunay triangulation

Unused functions (brute_force_KNN, brute_force_spherical) removed for code cleanup.
"""
import numpy as np
from scipy.spatial import Delaunay


def k_ring_delaunay(points, k):
    """
    Select all neighbors in the k first rings using Delaunay triangulation.
    
    This function creates a neighborhood graph based on Delaunay triangulation
    and extends it to k-ring neighborhoods for local surface analysis.
    
    Args:
        points: 3D point cloud array (N x 3)
        k: Number of rings to include in neighborhood
        
    Returns:
        neighborhood: Dictionary mapping point indices to their k-ring neighbors
        
    Used in: Step 1 - 3D Harris Keypoint Detection (fixed neighborhood method)
    """
    triangulation = Delaunay(points) # Compute structure

    neighborhood_direct = {}
    for f in triangulation.simplices :
        for v in range(f.shape[0]) :
            faces = list(f.copy())
            faces.pop(v)
            if f[v] in neighborhood_direct.keys():
                neighborhood_direct[f[v]] = list(np.unique(neighborhood_direct[f[v]]+faces))
            else:
                neighborhood_direct[f[v]] = faces

    neighborhood = {}                              
    for i in range(1, k):
        for v in neighborhood_direct.keys():
            for neighbor in neighborhood_direct[v]:
                if v in neighborhood.keys():
                    neighborhood[v] = list(np.unique(neighborhood[v] + neighborhood_direct[neighbor]))
                else :
                    neighborhood[v] = list(np.unique(neighborhood_direct[v] + neighborhood_direct[neighbor]))
    
    return(neighborhood)

def k_ring_delaunay_adaptive(points, delta, max_iter=5):
    """
    Choose the best k value according to delta and select neighbors in k rings.
    
    This adaptive method automatically determines the neighborhood size based on
    the delta parameter (fraction of the diagonal of the object bounding rectangle).
    It ensures consistent neighborhood coverage regardless of point cloud density.
    
    Args:
        points: 3D point cloud array (N x 3)
        delta: Distance threshold for neighborhood expansion
        max_iter: Maximum number of ring expansions (default: 5)
        
    Returns:
        neighborhood: Dictionary mapping point indices to their adaptive neighbors
        
    Used in: Step 1 - 3D Harris Keypoint Detection (adaptive neighborhood method)
    """
    triangulation = Delaunay(points) # Compute structure

    neighborhood_direct = {}
    for f in triangulation.simplices :
        for v in range(f.shape[0]) :
            faces = list(f.copy())
            faces.pop(v)
            if f[v] in neighborhood_direct.keys():
                neighborhood_direct[f[v]] = list(np.unique(neighborhood_direct[f[v]]+faces))
            else:
                neighborhood_direct[f[v]] = faces
    
    neighborhood = {}
    for v in neighborhood_direct.keys():
        query = points[v]
        #compute the distance of the query and its ring
        dist = np.max(np.linalg.norm(query-points[neighborhood_direct[v]], axis=1))
        if dist >= delta :
            bigger_ring = False
            neighborhood[v] = neighborhood_direct[v]
        else :
            bigger_ring = True
        
        iteration = 1
        while bigger_ring and iteration <= max_iter:
            iteration += 1
            for neighbor in neighborhood_direct[v]:
                if v in neighborhood.keys():
                    neighborhood[v] = list(np.unique(neighborhood[v] + neighborhood_direct[neighbor]))
                else :
                    neighborhood[v] = list(np.unique(neighborhood_direct[v] + neighborhood_direct[neighbor]))

            #compute the distance of the query and its ring
            dist = np.max(np.linalg.norm(query-points[neighborhood[v]], axis=1))
            if dist >= delta :
                bigger_ring = False
            else :
                bigger_ring = True
            
    return(neighborhood)