# -*- coding: utf-8 -*-
"""
Bin Picking Filter Module
Implementation of Step 0: Closest brick cluster filtering
"""
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from collections import Counter

class BinPickingFilter:
    def __init__(self):
        """Initialize bin picking filter with LEGO color definitions"""
        # Define LEGO colors in HSV ranges
        self.lego_colors = {
            'red': {'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
            'red2': {'lower': np.array([170, 50, 50]), 'upper': np.array([180, 255, 255])},  # Red wraps around
            'orange': {'lower': np.array([10, 50, 50]), 'upper': np.array([25, 255, 255])},
            'yellow': {'lower': np.array([25, 50, 50]), 'upper': np.array([35, 255, 255])},
            'light_green': {'lower': np.array([35, 50, 50]), 'upper': np.array([85, 255, 255])},
            'light_blue': {'lower': np.array([85, 50, 50]), 'upper': np.array([125, 255, 255])},
            'dark_blue': {'lower': np.array([100, 50, 50]), 'upper': np.array([130, 255, 255])},
        }
    
    def rgb_to_hsv_vectorized(self, rgb_colors):
        """
        Convert RGB colors to HSV using vectorized operations
        
        Args:
            rgb_colors: RGB color array (N x 3) with values in [0, 1]
            
        Returns:
            hsv_colors: HSV color array (N x 3)
        """
        # Convert to 0-255 range and uint8
        rgb_255 = (rgb_colors * 255).astype(np.uint8)
        
        # Reshape for OpenCV
        rgb_reshaped = rgb_255.reshape(-1, 1, 3)
        
        # Convert to HSV
        hsv_reshaped = cv2.cvtColor(rgb_reshaped, cv2.COLOR_RGB2HSV)
        
        # Reshape back
        hsv_colors = hsv_reshaped.reshape(-1, 3)
        
        return hsv_colors
    
    def classify_lego_color(self, hsv_color):
        """
        Classify a single HSV color as one of the LEGO colors
        
        Args:
            hsv_color: HSV color values (3,)
            
        Returns:
            color_name: Name of the closest LEGO color or None
        """
        for color_name, color_range in self.lego_colors.items():
            if color_name == 'red2':  # Skip the second red range in iteration
                continue
                
            lower = color_range['lower']
            upper = color_range['upper']
            
            # Check if color is within range
            if np.all(hsv_color >= lower) and np.all(hsv_color <= upper):
                return color_name.replace('2', '')  # Remove '2' suffix from red2
        
        # Special case for red (check both ranges)
        red_range1 = self.lego_colors['red']
        red_range2 = self.lego_colors['red2']
        
        if ((np.all(hsv_color >= red_range1['lower']) and np.all(hsv_color <= red_range1['upper'])) or
            (np.all(hsv_color >= red_range2['lower']) and np.all(hsv_color <= red_range2['upper']))):
            return 'red'
        
        return None
    
    def filter_by_lego_colors(self, points, colors):
        """
        Filter points to keep only those with valid LEGO colors
        
        Args:
            points: Point cloud (N x 3)
            colors: RGB colors (N x 3) in [0, 1] range
            
        Returns:
            filtered_points: Points with valid LEGO colors
            filtered_colors: Corresponding colors
            color_labels: Color classification for each point
        """
        print(f"Filtering {len(points)} points by LEGO colors...")
        
        # Convert RGB to HSV
        hsv_colors = self.rgb_to_hsv_vectorized(colors)
        
        # Classify each color
        valid_mask = []
        color_labels = []
        
        for hsv_color in hsv_colors:
            color_class = self.classify_lego_color(hsv_color)
            if color_class is not None:
                valid_mask.append(True)
                color_labels.append(color_class)
            else:
                valid_mask.append(False)
                color_labels.append('unknown')
        
        valid_mask = np.array(valid_mask)
        
        # Filter points and colors
        filtered_points = points[valid_mask]
        filtered_colors = colors[valid_mask]
        filtered_color_labels = [label for i, label in enumerate(color_labels) if valid_mask[i]]
        
        print(f"Kept {len(filtered_points)} points with valid LEGO colors")
        if len(filtered_color_labels) > 0:
            color_counts = Counter(filtered_color_labels)
            print(f"Color distribution: {dict(color_counts)}")
        
        return filtered_points, filtered_colors, filtered_color_labels
    
    def find_closest_cluster(self, points, colors, eps=0.01, min_samples=10):
        """
        Find the cluster containing the closest points to the camera
        
        Args:
            points: Point cloud (N x 3)
            colors: RGB colors (N x 3)
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples parameter
            
        Returns:
            closest_cluster_points: Points from closest cluster
            closest_cluster_colors: Colors from closest cluster
            cluster_info: Dictionary with cluster information
        """
        print(f"Finding closest cluster from {len(points)} points...")
        
        if len(points) < min_samples:
            print("Insufficient points for clustering")
            return points, colors, {}
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)
        
        # Find unique clusters (excluding noise points labeled as -1)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        if len(unique_labels) == 0:
            print("No valid clusters found")
            return np.array([]), np.array([]), {}
        
        print(f"Found {len(unique_labels)} clusters")
        
        # Find the cluster with the closest point to camera (smallest Z value)
        closest_cluster_label = None
        min_distance = float('inf')
        cluster_stats = {}
        
        for label in unique_labels:
            cluster_mask = (labels == label)
            cluster_points = points[cluster_mask]
            
            # Find closest point in this cluster (minimum Z distance)
            min_z = np.min(cluster_points[:, 2])
            cluster_size = len(cluster_points)
            mean_z = np.mean(cluster_points[:, 2])
            
            cluster_stats[label] = {
                'size': cluster_size,
                'min_z': min_z,
                'mean_z': mean_z,
                'centroid': np.mean(cluster_points, axis=0)
            }
            
            print(f"  Cluster {label}: {cluster_size} points, min_z={min_z:.3f}, mean_z={mean_z:.3f}")
            
            if min_z < min_distance:
                min_distance = min_z
                closest_cluster_label = label
        
        if closest_cluster_label is None:
            print("No closest cluster found")
            return np.array([]), np.array([]), {}
        
        # Extract closest cluster
        closest_mask = (labels == closest_cluster_label)
        closest_cluster_points = points[closest_mask]
        closest_cluster_colors = colors[closest_mask]
        
        print(f"Selected cluster {closest_cluster_label} with {len(closest_cluster_points)} points")
        print(f"Closest point distance: {min_distance:.3f}")
        
        cluster_info = {
            'cluster_label': closest_cluster_label,
            'total_clusters': len(unique_labels),
            'cluster_stats': cluster_stats,
            'closest_distance': min_distance
        }
        
        return closest_cluster_points, closest_cluster_colors, cluster_info
    
    def extract_dominant_color_region(self, points, colors, color_labels=None, 
                                    top_points_ratio=0.3):
        """
        Extract the dominant color from the closest points in the cluster
        
        Args:
            points: Cluster points (N x 3)
            colors: Cluster colors (N x 3)
            color_labels: Pre-classified color labels
            top_points_ratio: Ratio of closest points to consider for dominant color
            
        Returns:
            filtered_points: Points with dominant color
            filtered_colors: Corresponding colors
            dominant_color: Name of dominant color
        """
        if len(points) == 0:
            return np.array([]), np.array([]), None
        
        print(f"Extracting dominant color from {len(points)} points...")
        
        # Sort points by Z distance (closest first)
        z_distances = points[:, 2]
        sorted_indices = np.argsort(z_distances)
        
        # Take top closest points for color analysis
        num_top_points = max(1, int(len(points) * top_points_ratio))
        top_indices = sorted_indices[:num_top_points]
        top_colors = colors[top_indices]
        
        # Classify colors if not provided
        if color_labels is None:
            hsv_colors = self.rgb_to_hsv_vectorized(top_colors)
            top_color_labels = []
            for hsv_color in hsv_colors:
                color_class = self.classify_lego_color(hsv_color)
                if color_class is not None:
                    top_color_labels.append(color_class)
        else:
            top_color_labels = [color_labels[i] for i in top_indices if color_labels[i] != 'unknown']
        
        if len(top_color_labels) == 0:
            print("No valid LEGO colors found in closest points")
            return points, colors, None
        
        # Find dominant color
        color_counts = Counter(top_color_labels)
        dominant_color = color_counts.most_common(1)[0][0]
        
        print(f"Dominant color: {dominant_color}")
        print(f"Color distribution in closest points: {dict(color_counts)}")
        
        # Filter all points by dominant color
        if color_labels is None:
            # Re-classify all colors
            hsv_all = self.rgb_to_hsv_vectorized(colors)
            all_color_labels = []
            for hsv_color in hsv_all:
                color_class = self.classify_lego_color(hsv_color)
                all_color_labels.append(color_class)
        else:
            all_color_labels = color_labels
        
        # Keep points with dominant color
        dominant_mask = np.array([label == dominant_color for label in all_color_labels])
        filtered_points = points[dominant_mask]
        filtered_colors = colors[dominant_mask]
        
        print(f"Filtered to {len(filtered_points)} points with {dominant_color} color")
        
        return filtered_points, filtered_colors, dominant_color
    
    def apply_bin_picking_filters(self, points, colors, dbscan_eps=0.01, 
                                dbscan_min_samples=10, top_points_ratio=0.3):
        """
        Apply complete bin picking filtering pipeline (Step 0)
        
        Args:
            points: Input point cloud (N x 3)
            colors: Input RGB colors (N x 3) in [0, 1] range
            dbscan_eps: DBSCAN epsilon parameter
            dbscan_min_samples: DBSCAN minimum samples parameter
            top_points_ratio: Ratio of closest points for dominant color extraction
            
        Returns:
            result: Dictionary containing filtered data and metadata
        """
        print("="*50)
        print("STEP 0: CLOSEST BRICK CLUSTER FILTERING")
        print("="*50)
        
        # Step 0.1: Filter by valid LEGO colors
        lego_points, lego_colors, color_labels = self.filter_by_lego_colors(points, colors)
        
        if len(lego_points) == 0:
            print("No valid LEGO colors found!")
            return {
                'points': np.array([]),
                'colors': np.array([]),
                'dominant_color': None,
                'cluster_info': {},
                'success': False
            }
        
        # Step 0.2: Find closest cluster
        closest_points, closest_colors, cluster_info = self.find_closest_cluster(
            lego_points, lego_colors, eps=dbscan_eps, min_samples=dbscan_min_samples)
        
        if len(closest_points) == 0:
            print("No closest cluster found!")
            return {
                'points': np.array([]),
                'colors': np.array([]),
                'dominant_color': None,
                'cluster_info': cluster_info,
                'success': False
            }
        
        # Step 0.3: Extract dominant color region
        filtered_points, filtered_colors, dominant_color = self.extract_dominant_color_region(
            closest_points, closest_colors, top_points_ratio=top_points_ratio)
        
        if len(filtered_points) == 0:
            print("No points with dominant color found!")
            return {
                'points': np.array([]),
                'colors': np.array([]),
                'dominant_color': dominant_color,
                'cluster_info': cluster_info,
                'success': False
            }
        
        print(f"Final result: {len(filtered_points)} points with {dominant_color} color")
        print("="*50)
        
        return {
            'points': filtered_points,
            'colors': filtered_colors,
            'dominant_color': dominant_color,
            'cluster_info': cluster_info,
            'success': True,
            'original_count': len(points),
            'lego_color_count': len(lego_points),
            'closest_cluster_count': len(closest_points),
            'final_count': len(filtered_points)
        }
