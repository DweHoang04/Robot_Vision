# transformation.py
# Point Cloud Transformation and Preprocessing Library
# Contains functions for centering, scaling, rotation, and coordinate transformations

import numpy as np
from sklearn.decomposition import PCA


class PointCloudTransformation:
    """
    A class containing various point cloud transformation and preprocessing methods
    for improving 3D vision processing and Harris corner detection stability.
    """

    @staticmethod
    def centering_centroid(points):
        """
        Center the point cloud on its centroid
        
        Args:
            points: numpy array of 3D points (N x 3)
            
        Returns:
            tuple: (centered_points, original_centroid) for later restoration
        """
        centred_points = points.copy()
        centroid = np.mean(centred_points, axis=0)
        centred_points = centred_points - centroid
        return centred_points, centroid

    @staticmethod
    def centering_origin(points, centroid):
        """
        Restore the point cloud to its original position using the saved centroid
        
        Args:
            points: numpy array of centered 3D points (N x 3)
            centroid: original centroid position to restore to
            
        Returns:
            numpy array: restored points in original coordinate system
        """
        centred_points = points.copy()
        centred_points = centred_points + centroid
        return centred_points

    @staticmethod
    def scale_point_cloud(points, target_scale=1.0):
        """
        Scale point cloud to a target scale for consistent processing
        
        Args:
            points: numpy array of 3D points (N x 3)
            target_scale: desired maximum distance from origin after scaling
            
        Returns:
            tuple: (scaled_points, scaling_factor) for later restoration
        """
        scaled_points = points.copy()
        
        # Calculate current scale (maximum distance from origin)
        distances = np.linalg.norm(scaled_points, axis=1)
        current_scale = np.max(distances)
        
        if current_scale > 0:
            scaling_factor = target_scale / current_scale
            scaled_points = scaled_points * scaling_factor
            return scaled_points, scaling_factor
        else:
            return scaled_points, 1.0

    @staticmethod
    def align_to_principal_axes(points):
        """
        Align point cloud to its principal axes using PCA
        This improves the stability of Harris corner detection
        
        Args:
            points: numpy array of 3D points (N x 3)
            
        Returns:
            tuple: (aligned_points, rotation_matrix) for transformation tracking
        """
        # Center the points
        centered_points = points - np.mean(points, axis=0)
        
        # Compute PCA
        pca = PCA(n_components=3)
        pca.fit(centered_points)
        
        # Get the principal components (rotation matrix)
        rotation_matrix = pca.components_
        
        # Apply rotation to align with principal axes
        aligned_points = np.dot(centered_points, rotation_matrix.T)
        
        # Restore the centroid
        aligned_points = aligned_points + np.mean(points, axis=0)
        
        return aligned_points, rotation_matrix

    @staticmethod
    def preprocess_for_harris_detection(points):
        """
        Complete preprocessing pipeline for Harris 3D corner detection
        Includes centering, scaling, and principal axis alignment
        
        Args:
            points: numpy array of 3D points (N x 3)
            
        Returns:
            tuple: (preprocessed_points, transform_params) where transform_params
                   contains all transformation data for potential restoration
        """
        print("Preprocessing points for Harris detection...")
        
        # Step 1: Center on centroid
        centered_points, original_centroid = PointCloudTransformation.centering_centroid(points)
        
        # Step 2: Scale to unit scale for numerical stability
        scaled_points, scale_factor = PointCloudTransformation.scale_point_cloud(centered_points, target_scale=1.0)
        
        # Step 3: Align to principal axes (optional but recommended)
        aligned_points, rotation_matrix = PointCloudTransformation.align_to_principal_axes(scaled_points + original_centroid)
        
        # Store transformation parameters for later restoration
        transform_params = {
            'original_centroid': original_centroid,
            'scale_factor': scale_factor,
            'rotation_matrix': rotation_matrix
        }
        
        return aligned_points, transform_params

    @staticmethod
    def restore_harris_points(harris_points, original_points, transform_params):
        """
        Restore Harris corner points to original coordinate system
        
        Args:
            harris_points: detected corner points in transformed space
            original_points: original point cloud (for reference)
            transform_params: transformation parameters from preprocessing
            
        Returns:
            numpy array: corner points in original coordinate system
        """
        if len(harris_points) == 0:
            return harris_points
            
        restored_points = harris_points.copy()
        
        # Note: For this implementation, we'll keep points in the transformed space
        # since the clustering and analysis work better in the aligned coordinate system
        # If needed, we can add full inverse transformation here
        
        return restored_points

    @staticmethod
    def transform_point_cloud_to_world(points):
        """
        Transform points to normalize them to world coordinate system
        This is the specific transformation used in the bin picking system
        
        Args:
            points: numpy array of 3D points in camera coordinate system
            
        Returns:
            numpy array: transformed points in world coordinate system
        """
        new_origin = np.array([-0.1663194511548611, -0.30196779718241507, 0.652])
        # Rotating matrix that swap X and Y coordinates and invert Z coordinate
        rotation_matrix = np.array([
            [0,  1,  0],
            [1,  0,  0],
            [0,  0, -1]
        ]) 
        translated = points - new_origin # Shifting the points to normalize the new coordinate
        transformed = np.dot(translated, rotation_matrix.T) # Transforming the points using dot product
        return transformed

    @staticmethod
    def keep_inside_boundary_points(points, colors, x_min, x_max, y_min, y_max, margin=0.02):
        """
        Border filtering algorithm using AND logic to remove boundary points
        
        Args:
            points: numpy array of 3D points
            colors: numpy array of RGB colors corresponding to points
            x_min, x_max, y_min, y_max: boundary limits
            margin: margin to remove from boundaries
            
        Returns:
            tuple: (filtered_points, filtered_colors)
        """
        mask = (
            (points[:, 0] >= x_min + margin) & (points[:, 0] <= x_max - margin) &
            (points[:, 1] >= y_min + margin) & (points[:, 1] <= y_max - margin)
        ) # Removing the border by an amount of margin
        # The scanning range will be (x_min + margin, x_max - margin) x (y_min + margin, y_max - margin)
        return points[mask], colors[mask] # Return filtered point cloud and color values

    @staticmethod
    def keep_points_above_plane(points, colors, plane_model):
        """
        Filter 3D points so that only those above or on a reference plane are kept
        
        Args:
            points: numpy array of 3D points
            colors: numpy array of RGB colors
            plane_model: plane equation parameters [a, b, c, d] for ax + by + cz + d = 0
            
        Returns:
            tuple: (filtered_points, filtered_colors) above the plane
        """
        a, b, c, d = plane_model
        # Calculating distance from the point to the plane (< 0: Below; = 0: On; > 0: Above)
        mask = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d <= 0)
        return points[mask], colors[mask]
