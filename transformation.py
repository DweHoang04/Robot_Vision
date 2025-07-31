# -*- coding: utf-8 -*-
"""
Transformation Module for 3D Point Cloud Processing

This module provides point cloud transformation utilities for the 3D bin picking pipeline.
Currently used primarily for centering point clouds during Harris keypoint detection.

Original source: https://github.com/rengotj/projet_NPM
Modified and simplified for LEGO Duplo bin picking system.

Functions:
- centering_centroid(): Centers point clouds around their centroid (USED in Harris detection)
"""
import numpy as np
import math

from utils.ply import write_ply, read_ply


def centering_centroid(points):
    """
    Center the point cloud on its centroid.
    
    This function translates the point cloud so that its centroid is at the origin,
    which is essential for proper Harris corner response computation.
    
    Args:
        points: Input point cloud array (N x 3)
        
    Returns:
        centred_points: Points translated to center around origin
        centroid: Original centroid coordinates
        
    Used in: Step 1 - 3D Harris Keypoint Detection
    """
    centred_points = points.copy()
    centroid = np.mean(centred_points, axis=0)
    centred_points = centred_points - centroid
    return(centred_points, centroid)