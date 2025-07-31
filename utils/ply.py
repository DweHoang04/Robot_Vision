# -*- coding: utf-8 -*-
"""
PLY file I/O utilities
Compatible with the existing 3D_harris.py code
"""
import numpy as np
import open3d as o3d

def read_ply(filename):
    """
    Read PLY file and return data dictionary
    
    Args:
        filename: Path to PLY file
        
    Returns:
        data: Dictionary with 'x', 'y', 'z' keys containing point coordinates
    """
    try:
        # Use Open3D to read PLY file
        pcd = o3d.io.read_point_cloud(filename)
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            raise ValueError(f"No points found in PLY file: {filename}")
        
        # Return data in expected format
        data = {
            'x': points[:, 0],
            'y': points[:, 1], 
            'z': points[:, 2]
        }
        
        # Include colors if available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            data['red'] = colors[:, 0]
            data['green'] = colors[:, 1]
            data['blue'] = colors[:, 2]
        
        return data
        
    except Exception as e:
        print(f"Error reading PLY file {filename}: {e}")
        # Return empty data structure
        return {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}

def write_ply(filename, data_list, field_names):
    """
    Write PLY file from data
    
    Args:
        filename: Output PLY filename
        data_list: List of data arrays [points, labels1, labels2, ...]
        field_names: List of field names ['x', 'y', 'z', 'label1', 'label2', ...]
    """
    try:
        # Extract points (first item should be points)
        points = data_list[0]
        
        if len(points) == 0:
            print(f"Warning: No points to write to {filename}")
            return
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add colors if available in field names
        if len(data_list) > 1 and any(name in field_names for name in ['red', 'green', 'blue']):
            # Look for color fields
            try:
                if len(data_list) >= 4:  # points + 3 color channels
                    red_idx = field_names.index('red') if 'red' in field_names else 1
                    green_idx = field_names.index('green') if 'green' in field_names else 2  
                    blue_idx = field_names.index('blue') if 'blue' in field_names else 3
                    
                    colors = np.column_stack([
                        data_list[red_idx],
                        data_list[green_idx], 
                        data_list[blue_idx]
                    ])
                    pcd.colors = o3d.utility.Vector3dVector(colors)
            except (IndexError, ValueError):
                pass  # Skip colors if not properly formatted
        
        # Write PLY file
        success = o3d.io.write_point_cloud(filename, pcd)
        
        if success:
            print(f"PLY file written successfully: {filename}")
        else:
            print(f"Failed to write PLY file: {filename}")
            
    except Exception as e:
        print(f"Error writing PLY file {filename}: {e}")

def points_to_ply_format(points):
    """
    Convert points array to PLY-compatible format
    
    Args:
        points: Point cloud array (N x 3)
        
    Returns:
        data: Dictionary with 'x', 'y', 'z' keys
    """
    if len(points) == 0:
        return {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}
    
    return {
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2]
    }

def ply_format_to_points(data):
    """
    Convert PLY format data to points array
    
    Args:
        data: Dictionary with 'x', 'y', 'z' keys
        
    Returns:
        points: Point cloud array (N x 3)
    """
    if 'x' not in data or 'y' not in data or 'z' not in data:
        return np.array([])
    
    points = np.column_stack([data['x'], data['y'], data['z']])
    return points
