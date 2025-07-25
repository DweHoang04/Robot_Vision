import os
import socket
import time
import cv2
import numpy as np
import open3d as o3d
import struct
import math
import itertools
import scipy.spatial.distance
import copy
import pickle
from datetime import datetime
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.transform import Rotation as SciRotation

from pykinect2 import PyKinectRuntime, PyKinectV2
from pykinect2.PyKinectV2 import *

class BinPickingSystem:

    def __init__(self, wdf_path):
        self.output_dir = "C:\\Users\\FILAB\\Desktop\\DUY\\Results"
        # self.host = "192.168.1.23"
        # self.port = 9999

    def keep_inside_boundary_points(self, points, colors, x_min, x_max, y_min, y_max, margin=0.02):
        mask = (
            (points[:, 0] >= x_min + margin) & (points[:, 0] <= x_max - margin) &
            (points[:, 1] >= y_min + margin) & (points[:, 1] <= y_max - margin)
        )
        return points[mask], colors[mask]

    def apply_dbscan(self, points, colors, eps=0.05, min_samples=10):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = dbscan.labels_
        mask = (labels != -1)
        return points[mask], colors[mask]

    def keep_points_above_plane(self, points, colors, plane_model):
        a, b, c, d = plane_model
        mask = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d <= 0)
        return points[mask], colors[mask]

    def transform_point_cloud(self, points):
        new_origin = np.array([-0.1663194511548611, -0.30196779718241507, 0.652])
        rotation_matrix = np.array([
            [0,  1,  0],
            [1,  0,  0],
            [0,  0, -1]
        ])
        translated = points - new_origin
        transformed = np.dot(translated, rotation_matrix.T)
        return transformed

    def capture_and_preprocess_kinect_data(self, roi_x=195, roi_y=50, roi_w=245, roi_h=300,
                                           plane_dist_thresh=0.005, ransac_n=3, ransac_iter=1000,
                                           boundary_margin=0.005, dbscan_eps_pre=0.01, dbscan_min_samples_pre=50):
        kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

        while not (kinect.has_new_depth_frame() and kinect.has_new_color_frame()):
            time.sleep(0.01)

        depth_frame = kinect.get_last_depth_frame().reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width))
        color_frame = kinect.get_last_color_frame().reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4))[:, :, :3]
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

        depth_roi = depth_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        color_roi = color_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        intrinsics = kinect._mapper.GetDepthCameraIntrinsics()
        fx, fy = intrinsics.FocalLengthX, intrinsics.FocalLengthY
        cx, cy = intrinsics.PrincipalPointX, intrinsics.PrincipalPointY

        points, colors = [], []
        for i in range(depth_roi.shape[0]):
            for j in range(depth_roi.shape[1]):
                z = depth_roi[i, j] * 0.001
                if z > 0:
                    x = (j + roi_x - cx) * z / fx
                    y = -(i + roi_y - cy) * z / fy
                    depth_point = PyKinectV2._DepthSpacePoint()
                    depth_point.x, depth_point.y = j + roi_x, i + roi_y
                    color_point = kinect._mapper.MapDepthPointToColorSpace(depth_point, depth_roi[i, j])
                    cx_c, cy_c = int(color_point.x), int(color_point.y)
                    if 0 <= cx_c < color_frame.shape[1] and 0 <= cy_c < color_frame.shape[0]:
                        c = color_frame[cy_c, cx_c] / 255.0
                        points.append((x, y, z))
                        colors.append(c)

        kinect.close()
        points = np.array(points)
        colors = np.array(colors)

        if len(points) < 3:
            return np.array([]), np.array([])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        plane_model, inliers = pcd.segment_plane(distance_threshold=plane_dist_thresh,
                                                 ransac_n=ransac_n,
                                                 num_iterations=ransac_iter)
        non_plane_cloud = pcd.select_by_index(inliers, invert=True)
        above_points, above_colors = self.keep_points_above_plane(np.asarray(non_plane_cloud.points),
                                                                  np.asarray(non_plane_cloud.colors),
                                                                  plane_model)
        if len(above_points) == 0:
            return np.array([]), np.array([])

        x_min, y_min, _ = np.min(above_points, axis=0)
        x_max, y_max, _ = np.max(above_points, axis=0)
        margin_points, margin_colors = self.keep_inside_boundary_points(above_points, above_colors,
                                                                        x_min, x_max, y_min, y_max,
                                                                        margin=boundary_margin)
        if len(margin_points) == 0:
            return np.array([]), np.array([])

        denoised_points, denoised_colors = self.apply_dbscan(margin_points, margin_colors,
                                                             eps=dbscan_eps_pre,
                                                             min_samples=dbscan_min_samples_pre)
        if len(denoised_points) == 0:
            return np.array([]), np.array([])

        transformed = self.transform_point_cloud(denoised_points)
        return transformed, denoised_colors

    def save_transformed_point_cloud(self, points, colors, output_file):
        data = np.hstack([points, colors])
        np.savetxt(output_file, data, delimiter=' ', fmt='%f')

    def save_cloud_image(self, points, colors, image_path):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)  # Stabilize rendering
        vis.capture_screen_image(image_path)
        vis.destroy_window()
    '''    
    # Negative means turn left, positive means turn right.
    def calculate_y_axis_angle_xy(self, minor_axis):
        ref_axis = np.array([0, 1])
        v2d = minor_axis[:2] / np.linalg.norm(minor_axis[:2])
        angle_deg = np.degrees(np.arccos(np.clip(np.dot(v2d, ref_axis), -1.0, 1.0)))
        if v2d[0] < 0:
            angle_deg = -angle_deg
        return angle_deg
    '''
    def calculate_y_axis_angle_xy(self, minor_axis):
        # Normalize to unit vector
        v2d = minor_axis[:2] / np.linalg.norm(minor_axis[:2])

        def clockwise_angle_from_y(vec):
            # Clockwise rotation angle from Y-axis (0~360)
            angle = np.degrees(np.arctan2(vec[0], vec[1])) % 360
            return angle

        angle1 = clockwise_angle_from_y(v2d)
        angle2 = clockwise_angle_from_y(-v2d)

        # Since it's a line, remove directionality → the smaller rotation angle is the actual clockwise rotation of the line
        angle_deg = min(angle1, angle2)
        # If over 90°, convert to complementary angle
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
            angle_deg = - angle_deg
        return angle_deg  # Finally return with negative sign

    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space for better color discrimination"""
        rgb = np.array(rgb)
        if rgb.ndim == 1:
            rgb = rgb.reshape(1, -1)
        
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        # Hue calculation
        h = np.zeros_like(max_val)
        mask = diff != 0
        
        max_is_r = (max_val == r) & mask
        max_is_g = (max_val == g) & mask
        max_is_b = (max_val == b) & mask
        
        h[max_is_r] = (60 * ((g[max_is_r] - b[max_is_r]) / diff[max_is_r]) + 360) % 360
        h[max_is_g] = (60 * ((b[max_is_g] - r[max_is_g]) / diff[max_is_g]) + 120) % 360
        h[max_is_b] = (60 * ((r[max_is_b] - g[max_is_b]) / diff[max_is_b]) + 240) % 360
        
        # Saturation calculation
        s = np.zeros_like(max_val)
        s[max_val != 0] = diff[max_val != 0] / max_val[max_val != 0]
        
        # Value calculation
        v = max_val
        
        return np.column_stack([h, s, v])

    def identify_lego_color_category(self, rgb_color):
        """Identify LEGO brick color category based on RGB values"""
        r, g, b = rgb_color[0], rgb_color[1], rgb_color[2]
        
        # Define color ranges for 6 LEGO colors
        if r > 0.6 and g < 0.3 and b < 0.3:
            return "red"
        elif r > 0.7 and g > 0.4 and b < 0.3:
            return "orange"
        elif r > 0.7 and g > 0.7 and b < 0.4:
            return "yellow"
        elif r < 0.4 and g > 0.5 and b < 0.4:
            return "light_green"
        elif r < 0.4 and g > 0.4 and b > 0.5:
            # Use HSV to distinguish light blue vs dark blue
            hsv = self.rgb_to_hsv(rgb_color.reshape(1, -1))[0]
            h, s, v = hsv[0], hsv[1], hsv[2]
            
            # Dark blue: lower value (brightness), higher saturation
            # Light blue: higher value (brightness), potentially lower saturation
            if v < 0.6 or s > 0.7:
                return "dark_blue"
            else:
                return "light_blue"
        else:
            # Fallback: try to distinguish based on overall brightness
            brightness = (r + g + b) / 3
            if r < 0.5 and g < 0.5 and b > 0.3:
                if brightness < 0.4:
                    return "dark_blue"
                else:
                    return "light_blue"
            else:
                return "unknown"

    def get_dominant_color(self, colors):
        """Extract the most dominant color from a set of RGB values with improved precision"""
        if len(colors) == 0:
            return np.array([0.5, 0.5, 0.5])  # Default gray
        
        # Use higher precision for color analysis
        colors_precise = np.clip(colors, 0, 1)
        
        # Categorize colors first
        color_categories = []
        for color in colors_precise:
            category = self.identify_lego_color_category(color)
            color_categories.append(category)
        
        # Find most common category
        category_counter = Counter(color_categories)
        dominant_category = max(category_counter, key=category_counter.get)
        
        print(f"Dominant color category: {dominant_category}")
        
        # Get average color for the dominant category
        category_mask = np.array(color_categories) == dominant_category
        if np.sum(category_mask) > 0:
            dominant_color = np.mean(colors_precise[category_mask], axis=0)
        else:
            # Fallback to simple averaging
            dominant_color = np.mean(colors_precise, axis=0)
        
        return dominant_color

    # ========== STEP 2: FIND CLOSEST CLUSTER ==========
    def find_closest_brick_cluster(self, points, colors, dbscan_eps=0.01, dbscan_min_samples=10):
        """
        Find the cluster that has the closest point to the camera (highest Z coordinate)
        Returns the cluster points, colors, and cluster info
        """
        print("Step 2: Finding closest brick cluster...")
        
        # Perform DBSCAN clustering to identify individual brick clusters
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points)
        labels = dbscan.labels_

        # Find all valid clusters (exclude noise with label -1)
        valid_clusters = []
        for lbl in set(labels):
            if lbl == -1:  # Skip noise points
                continue
            
            cluster_mask = (labels == lbl)
            cluster_pts = points[cluster_mask]
            cluster_cols = colors[cluster_mask]
            
            # Filter small clusters that are likely noise
            if len(cluster_pts) < 500:
                continue
                
            # Find the highest point (closest to camera) in this cluster
            max_z = np.max(cluster_pts[:, 2])
            avg_z = np.mean(cluster_pts[:, 2])
            
            valid_clusters.append({
                'label': lbl,
                'points': cluster_pts,
                'colors': cluster_cols,
                'max_z': max_z,
                'avg_z': avg_z,
                'size': len(cluster_pts)
            })
        
        if len(valid_clusters) == 0:
            print("No valid brick clusters found")
            return None, None, None
        
        # Sort by highest point (max Z coordinate) - closest to camera
        valid_clusters.sort(key=lambda x: x['max_z'], reverse=True)
        closest_cluster = valid_clusters[0]
        
        print(f"Selected closest cluster {closest_cluster['label']} with max Z={closest_cluster['max_z']:.3f}m")
        print(f"Cluster size: {closest_cluster['size']} points")
        
        return closest_cluster['points'], closest_cluster['colors'], closest_cluster

    # ========== STEP 3: COLOR FILTERING ==========
    def filter_by_top_brick_color(self, cluster_points, cluster_colors, color_tolerance=0.15):
        """
        Filter points to keep only those with similar color to the top brick
        Uses the highest points to determine the dominant color of the top brick
        Applies adaptive tolerance for blue color variants
        """
        print("Step 3: Filtering by top brick color...")
        
        if len(cluster_points) == 0:
            return cluster_points, cluster_colors
        
        # Find points in the top 20% of the cluster (highest Z values)
        z_coords = cluster_points[:, 2]
        z_threshold = np.percentile(z_coords, 80)  # Top 20%
        top_mask = z_coords >= z_threshold
        
        if np.sum(top_mask) == 0:
            print("No top points found, using all cluster points")
            return cluster_points, cluster_colors
        
        top_colors = cluster_colors[top_mask]
        top_dominant_color = self.get_dominant_color(top_colors)
        
        print(f"Top brick dominant color: RGB({top_dominant_color[0]:.3f}, {top_dominant_color[1]:.3f}, {top_dominant_color[2]:.3f})")
        
        # Identify the color category for adaptive tolerance
        color_category = self.identify_lego_color_category(top_dominant_color)
        
        # Adaptive tolerance based on color category
        if color_category in ["dark_blue", "light_blue"]:
            # For blue variants, use more relaxed tolerance since they overlap in RGB
            adaptive_tolerance = color_tolerance * 1.3  # 30% more tolerance
            print(f"Blue color detected, using adaptive tolerance: {adaptive_tolerance:.3f}")
            
            # Additional HSV-based filtering for blue colors
            color_distances = np.linalg.norm(cluster_colors - top_dominant_color, axis=1)
            similar_color_mask = color_distances <= adaptive_tolerance
            
            # Secondary HSV filtering for better blue discrimination
            cluster_hsv = self.rgb_to_hsv(cluster_colors)
            top_hsv = self.rgb_to_hsv(top_dominant_color.reshape(1, -1))[0]
            
            # Focus on hue similarity for blues (hue around 240° for blues)
            hue_diff = np.abs(cluster_hsv[:, 0] - top_hsv[0])
            hue_diff = np.minimum(hue_diff, 360 - hue_diff)  # Handle circular hue
            hue_mask = hue_diff <= 30  # 30 degree hue tolerance
            
            # Combine RGB and HSV filtering
            combined_mask = similar_color_mask & hue_mask
            
            print(f"Blue filtering: RGB mask: {np.sum(similar_color_mask)}, HSV mask: {np.sum(hue_mask)}, Combined: {np.sum(combined_mask)}")
            
        else:
            # For other colors, use standard tolerance
            adaptive_tolerance = color_tolerance
            color_distances = np.linalg.norm(cluster_colors - top_dominant_color, axis=1)
            combined_mask = color_distances <= adaptive_tolerance
            
            print(f"Standard color filtering with tolerance: {adaptive_tolerance:.3f}")
        
        filtered_points = cluster_points[combined_mask]
        filtered_colors = cluster_colors[combined_mask]
        
        print(f"Color filtering: {len(cluster_points)} -> {len(filtered_points)} points")
        
        # If filtering removed too many points, fall back to less strict filtering
        if len(filtered_points) < len(cluster_points) * 0.1:  # Less than 10% remaining
            print("Warning: Aggressive filtering removed too many points, using relaxed tolerance")
            relaxed_tolerance = adaptive_tolerance * 1.5
            color_distances = np.linalg.norm(cluster_colors - top_dominant_color, axis=1)
            relaxed_mask = color_distances <= relaxed_tolerance
            filtered_points = cluster_points[relaxed_mask]
            filtered_colors = cluster_colors[relaxed_mask]
            print(f"Relaxed filtering: {len(cluster_points)} -> {len(filtered_points)} points")
        
        return filtered_points, filtered_colors

    # ========== STEP 4: HARRIS 3D CORNER DETECTION ==========
    def polyfit3d(self, x, y, z, order=2):
        """Fit a 3D polynomial surface to the data points"""
        ncols = (order + 1)**2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order+1), range(order+1))
        for k, (i, j) in enumerate(ij):
            G[:, k] = x**i * y**j
        m, _, _, _ = np.linalg.lstsq(G, z, rcond=None)
        return m

    def compute_knn_neighborhood(self, points, k=6):
        """Compute k-nearest neighbor neighborhoods"""
        neighborhoods = {}
        for i, query in enumerate(points):
            distances = np.linalg.norm(query - points, axis=1)
            neighbor_indices = np.argsort(distances)[1:k+1]  # Exclude the point itself
            neighborhoods[i] = neighbor_indices.tolist()
        return neighborhoods

    def compute_harris_3d_corners(self, points, delta=0.025, harris_k=0.04, num_corners=20):
        """
        Compute Harris 3D corner detection for LEGO brick feature points
        """
        print("Step 4a: Computing Harris 3D corners...")
        
        if len(points) < 10:
            print("Not enough points for Harris corner detection")
            return []

        # Compute neighborhoods
        neighborhood = self.compute_knn_neighborhood(points, k=8)
        
        # Initialize response array
        resp = np.zeros(len(points))
        
        # Compute Harris response for each point
        for point_idx in range(len(points)):
            try:
                if point_idx not in neighborhood or len(neighborhood[point_idx]) < 6:
                    resp[point_idx] = -np.inf
                    continue
                
                neighbors = points[neighborhood[point_idx], :]
                neighbors_centered = neighbors - np.mean(neighbors, axis=0)
                
                # PCA for local coordinate system
                pca = PCA(n_components=3)
                pca.fit(neighbors_centered)
                eigenvectors = pca.components_
                
                # Project to local 2D coordinate system
                neighbors_2d = neighbors_centered @ eigenvectors[:2].T
                neighbors_z = neighbors_centered @ eigenvectors[2]
                
                if len(neighbors_2d) >= 6:
                    # Fit polynomial surface
                    m = self.polyfit3d(neighbors_2d[:, 0], neighbors_2d[:, 1], neighbors_z, order=2)
                    m = m.reshape((3, 3))
                    
                    # Compute Harris response
                    fx2 = m[2, 0]**2 + 2*m[1, 1]**2 + 2*m[0, 2]**2
                    fy2 = m[0, 2]**2 + 2*m[1, 1]**2 + 2*m[2, 0]**2
                    fxfy = m[2, 0]*m[0, 2] + 2*m[1, 1]**2
                    
                    resp[point_idx] = fx2 * fy2 - fxfy * fxfy - harris_k * (fx2 + fy2) * (fx2 + fy2)
                else:
                    resp[point_idx] = -np.inf
                    
            except Exception as e:
                resp[point_idx] = -np.inf
                continue
        
        # Find corner candidates
        candidates = []
        for point_idx in range(len(points)):
            if point_idx in neighborhood:
                neighbor_responses = resp[neighborhood[point_idx]]
                if len(neighbor_responses) > 0 and resp[point_idx] >= np.max(neighbor_responses):
                    candidates.append([point_idx, resp[point_idx]])
        
        if len(candidates) == 0:
            print("No corner candidates found")
            return []
        
        # Sort by Harris response and select top corners
        candidates.sort(reverse=True, key=lambda x: x[1])
        candidates = np.array(candidates)
        
        # Select spatially distributed corners
        selected_corners = []
        corner_points = []
        min_distance = 0.01  # Minimum distance between corners
        
        for i, (point_idx, response) in enumerate(candidates):
            point_idx = int(point_idx)
            candidate_point = points[point_idx]
            
            # Check if this corner is far enough from already selected corners
            too_close = False
            for selected_point in corner_points:
                if np.linalg.norm(candidate_point - selected_point) < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                selected_corners.append(point_idx)
                corner_points.append(candidate_point)
                
                if len(selected_corners) >= num_corners:
                    break
        
        corner_points = np.array(corner_points) if corner_points else np.array([]).reshape(0, 3)
        print(f"Found {len(corner_points)} Harris corners")
        
        return corner_points

    # ========== STEP 4: SPIN IMAGE POSE ESTIMATION ==========
    def compute_surface_normals(self, points, k=6):
        """Compute surface normals for each point using local PCA"""
        normals = np.zeros_like(points)
        
        for i, point in enumerate(points):
            distances = np.linalg.norm(points - point, axis=1)
            neighbors_idx = np.argsort(distances)[1:k+1]
            neighbors = points[neighbors_idx]
            
            centered = neighbors - np.mean(neighbors, axis=0)
            
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                normal = vh[-1]
                
                # Ensure consistent orientation
                centroid_dir = point - np.mean(neighbors, axis=0)
                if np.dot(normal, centroid_dir) < 0:
                    normal = -normal
                    
                normals[i] = normal
            except:
                normals[i] = np.array([0, 0, 1])
                
        return normals

    def compute_spin_image(self, point, normal, points, spin_size=32, max_radius=0.03):
        """Compute spin image descriptor according to Johnson & Hebert"""
        vectors = points - point
        beta = np.dot(vectors, normal)
        
        projections = np.outer(beta, normal)
        perpendicular = vectors - projections
        alpha = np.linalg.norm(perpendicular, axis=1)
        
        alpha_max = max_radius
        beta_min, beta_max = -max_radius, max_radius
        
        alpha_bins = np.linspace(0, alpha_max, spin_size)
        beta_bins = np.linspace(beta_min, beta_max, spin_size)
        
        spin_image = np.zeros((spin_size, spin_size))
        
        valid_mask = (alpha <= alpha_max) & (beta >= beta_min) & (beta <= beta_max)
        valid_alpha = alpha[valid_mask]
        valid_beta = beta[valid_mask]
        
        if len(valid_alpha) > 0:
            alpha_indices = np.digitize(valid_alpha, alpha_bins) - 1
            beta_indices = np.digitize(valid_beta, beta_bins) - 1
            
            alpha_indices = np.clip(alpha_indices, 0, spin_size - 1)
            beta_indices = np.clip(beta_indices, 0, spin_size - 1)
            
            for a_idx, b_idx in zip(alpha_indices, beta_indices):
                spin_image[b_idx, a_idx] += 1
        
        if np.sum(spin_image) > 0:
            spin_image = spin_image / np.sum(spin_image)
            
        return spin_image

    def compute_spin_image_correlation(self, spin1, spin2):
        """Compute Pearson correlation coefficient between two spin images"""
        p = spin1.flatten()
        q = spin2.flatten()
        
        N = len(p)
        if N == 0:
            return 0.0
        
        sum_p = np.sum(p)
        sum_q = np.sum(q)
        sum_pq = np.sum(p * q)
        sum_p2 = np.sum(p * p)
        sum_q2 = np.sum(q * q)
        
        numerator = N * sum_pq - sum_p * sum_q
        denominator_p = N * sum_p2 - sum_p * sum_p
        denominator_q = N * sum_q2 - sum_q * sum_q
        denominator = np.sqrt(denominator_p * denominator_q)
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        return np.clip(correlation, -1.0, 1.0)

    # ========== STEP 4: UPDATED ALIGNMENT ALGORITHM ==========
    def estimate_rigid_transformation_3_points(self, template_pts, target_pts):
        """Estimate rigid transformation from 3 point correspondences"""
        if len(template_pts) != 3 or len(target_pts) != 3:
            return None, None
        
        template_centroid = np.mean(template_pts, axis=0)
        target_centroid = np.mean(target_pts, axis=0)
        
        template_centered = template_pts - template_centroid
        target_centered = target_pts - target_centroid
        
        H = template_centered.T @ target_centered
        
        try:
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            T = target_centroid - R @ template_centroid
            return R, T
        except:
            return None, None

    def updated_alignment_algorithm(self, Uc, Wc, U, W, epsilon1=0.005, epsilon2=0.01, 
                                  iter_max=1000, cmax=0.1, pmax=0.1):
        """
        Updated alignment algorithm from the paper:
        INPUT: Uc, Wc, U, W (correspondences and full point sets)
        OUTPUT: estimated pose
        """
        print("Step 4b: Running updated alignment algorithm...")
        
        if len(Uc) < 3 or len(Wc) < 3:
            print("Need at least 3 correspondences for pose estimation")
            return None, None
        
        Ce = cmax
        Pe = pmax
        best_R, best_T = None, None
        best_inlier_count = 0
        
        print(f"Starting alignment with {len(Uc)} correspondences...")
        
        while Ce > epsilon2:
            Pe = pmax
            
            while Pe > epsilon1:
                iteration = 1
                
                while iteration <= iter_max:
                    # Select 3 similar matching pair points randomly
                    if len(Uc) < 3:
                        break
                    
                    random_indices = np.random.choice(len(Uc), 3, replace=False)
                    sample_Uc = Uc[random_indices]
                    sample_Wc = Wc[random_indices]
                    
                    # Compute the rigid transformation
                    R, T = self.estimate_rigid_transformation_3_points(sample_Uc, sample_Wc)
                    
                    if R is None:
                        iteration += 1
                        continue
                    
                    # Transform Uc: U'c = R*Uc + T
                    Uc_transformed = (R @ Uc.T).T + T
                    
                    # Compute Pe = ||U'c - Wc|| / sqrt(n)
                    distances = np.linalg.norm(Uc_transformed - Wc, axis=1)
                    Pe = np.mean(distances)
                    
                    if Pe <= epsilon1:
                        # Count inliers for this transformation
                        inliers = distances < epsilon1
                        inlier_count = np.sum(inliers)
                        
                        if inlier_count > best_inlier_count:
                            best_inlier_count = inlier_count
                            best_R, best_T = R, T
                        break
                    
                    iteration += 1
                    
                    if iteration > iter_max:
                        break
                
                if Pe <= epsilon1:
                    break
                else:
                    # Relax Pe threshold slightly
                    epsilon1 *= 1.1
                    if epsilon1 > 0.02:  # Maximum tolerance
                        break
            
            if Pe > epsilon1:
                print("Failed to find valid transformation within Pe threshold")
                return None, None
            
            if best_R is not None:
                # Transform U: U' = R*U + T
                U_transformed = (best_R @ U.T).T + best_T
                
                # Compute the means for U' and W as U'm and Wm
                U_mean = np.mean(U_transformed, axis=0)
                W_mean = np.mean(W, axis=0)
                
                # Ce = ||U'm - Wm||
                Ce = np.linalg.norm(U_mean - W_mean)
                
                if Ce <= epsilon2:
                    break
                else:
                    # Relax Ce threshold slightly
                    epsilon2 *= 1.1
                    if epsilon2 > 0.05:  # Maximum tolerance
                        break
            else:
                break
        
        if best_R is not None:
            # ICP Refinement on U' and W
            print("Performing ICP refinement...")
            final_R, final_T = self.refine_pose_with_icp(U, W, best_R, best_T)
            
            print(f"Alignment completed with {best_inlier_count} inliers")
            print(f"Final Pe: {Pe:.6f}, Ce: {Ce:.6f}")
            
            return final_R, final_T
        else:
            print("Alignment algorithm failed to find valid pose")
            return None, None

    def refine_pose_with_icp(self, template_points, target_points, initial_R, initial_T, 
                           max_iterations=30, tolerance=1e-6):
        """ICP refinement for pose estimation"""
        R, T = initial_R.copy(), initial_T.copy()
        
        for iteration in range(max_iterations):
            transformed_template = (R @ template_points.T).T + T
            
            distances = scipy.spatial.distance.cdist(transformed_template, target_points)
            nearest_indices = np.argmin(distances, axis=1)
            nearest_target = target_points[nearest_indices]
            
            if len(transformed_template) >= 3:
                new_R, new_T = self.estimate_rigid_transformation_3_points(
                    transformed_template[:3], nearest_target[:3])
                
                if new_R is None:
                    break
                
                R = new_R @ R
                T = new_R @ T + new_T
                
                translation_change = np.linalg.norm(new_T)
                rotation_change = np.linalg.norm(new_R - np.eye(3))
                
                if translation_change < tolerance and rotation_change < tolerance:
                    break
        
        return R, T

    def estimate_pose_with_spin_images(self, template_points, target_points, 
                                     correlation_threshold=0.6):
        """Complete pose estimation using Harris corners and spin images"""
        print("Step 4: Starting pose estimation with spin images...")
        
        # Detect Harris corners
        template_corners = self.compute_harris_3d_corners(template_points, num_corners=15)
        target_corners = self.compute_harris_3d_corners(target_points, num_corners=15)
        
        if len(template_corners) < 3 or len(target_corners) < 3:
            print("Insufficient corners for pose estimation")
            return None, None
        
        # Compute surface normals
        template_normals = self.compute_surface_normals(template_points)
        target_normals = self.compute_surface_normals(target_points)
        
        # Get normals for corner points
        template_corner_normals = []
        for corner in template_corners:
            distances = np.linalg.norm(template_points - corner, axis=1)
            closest_idx = np.argmin(distances)
            template_corner_normals.append(template_normals[closest_idx])
        
        target_corner_normals = []
        for corner in target_corners:
            distances = np.linalg.norm(target_points - corner, axis=1)
            closest_idx = np.argmin(distances)
            target_corner_normals.append(target_normals[closest_idx])
        
        # Generate spin images and find correspondences
        correspondences = []
        
        for i, (corner, normal) in enumerate(zip(template_corners, template_corner_normals)):
            template_spin = self.compute_spin_image(corner, normal, template_points)
            
            best_correlation = -1
            best_match = -1
            
            for j, (target_corner, target_normal) in enumerate(zip(target_corners, target_corner_normals)):
                target_spin = self.compute_spin_image(target_corner, target_normal, target_points)
                correlation = self.compute_spin_image_correlation(template_spin, target_spin)
                
                if correlation > best_correlation and correlation > correlation_threshold:
                    best_correlation = correlation
                    best_match = j
            
            if best_match >= 0:
                correspondences.append((i, best_match, best_correlation))
        
        print(f"Found {len(correspondences)} correspondences")
        
        if len(correspondences) < 3:
            print("Insufficient correspondences for pose estimation")
            return None, None
        
        # Prepare correspondence points for alignment algorithm
        Uc = np.array([template_corners[i] for i, j, _ in correspondences])
        Wc = np.array([target_corners[j] for i, j, _ in correspondences])
        
        # Use updated alignment algorithm
        R, T = self.updated_alignment_algorithm(Uc, Wc, template_points, target_points)
        
        return R, T

    # ========== STEP 5: WORLD COORDINATE CALCULATION ==========
    def extract_world_coordinates(self, R, T, coordinate_system='euler_xyz'):
        """Extract world coordinates from pose estimation results"""
        print("Step 5: Extracting world coordinates...")
        
        x, y, z = T[0], T[1], T[2]
        
        if coordinate_system == 'euler_xyz':
            rotation_obj = SciRotation.from_matrix(R)
            euler_angles = rotation_obj.as_euler('xyz', degrees=True)
            rx, ry, rz = euler_angles[0], euler_angles[1], euler_angles[2]
        else:
            # Fallback manual extraction
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            
            if not singular:
                rx = math.atan2(R[2, 1], R[2, 2])
                ry = math.atan2(-R[2, 0], sy)
                rz = math.atan2(R[1, 0], R[0, 0])
            else:
                rx = math.atan2(-R[1, 2], R[1, 1])
                ry = math.atan2(-R[2, 0], sy)
                rz = 0
            
            rx = math.degrees(rx)
            ry = math.degrees(ry)
            rz = math.degrees(rz)
        
        coordinates = {
            'position': [x, y, z],
            'rotation': [rx, ry, rz],
            'rotation_matrix': R.tolist(),
            'translation': T.tolist(),
            'coordinate_system': coordinate_system
        }
        
        print(f"World coordinates extracted:")
        print(f"  Position (x, y, z): ({x:.3f}, {y:.3f}, {z:.3f}) meters")
        print(f"  Rotation (rx, ry, rz): ({rx:.1f}, {ry:.1f}, {rz:.1f}) degrees")
        
        return coordinates

    # ========== STEP 6: VISUALIZATION ==========
    def visualize_matched_template(self, original_points, original_colors, template_points, 
                                  R, T, highlight_color=[0, 0, 0]):  # Black highlight
        """
        Visualize the matched template overlaid on the original cluster
        """
        print("Step 6: Creating visualization with matched template...")
        
        # Transform template points to match the target
        if R is not None and T is not None:
            transformed_template = (R @ template_points.T).T + T
        else:
            transformed_template = template_points
        
        # Create point clouds for visualization
        # Original cluster
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(original_points)
        original_pcd.colors = o3d.utility.Vector3dVector(original_colors)
        
        # Matched template (highlighted in black)
        template_pcd = o3d.geometry.PointCloud()
        template_pcd.points = o3d.utility.Vector3dVector(transformed_template)
        template_colors = np.array([highlight_color] * len(transformed_template))
        template_pcd.colors = o3d.utility.Vector3dVector(template_colors)
        
        # Coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        
        # Visualize
        geometries = [original_pcd, template_pcd, coord_frame]
        
        try:
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Matched Template Visualization",
                width=800,
                height=600
            )
        except Exception as e:
            print(f"Visualization failed: {e}")
        
        return geometries

    def save_visualization_image(self, geometries, output_path):
        """Save the visualization as an image"""
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=800, height=600)
            
            for geometry in geometries:
                vis.add_geometry(geometry)
            
            vis.update_geometry(o3d.geometry.PointCloud())
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)
            
            vis.capture_screen_image(output_path)
            vis.destroy_window()
            
            print(f"Visualization saved to: {output_path}")
        except Exception as e:
            print(f"Failed to save visualization: {e}")

    # ========== TEMPLATE CREATION FROM CAD MODELS ==========
    def load_cad_model(self, stl_file_path):
        """Load STL CAD model and convert to point cloud"""
        try:
            mesh = o3d.io.read_triangle_mesh(stl_file_path)
            if len(mesh.vertices) == 0:
                print(f"Failed to load STL file: {stl_file_path}")
                return None
            
            # Normalize mesh size and center it
            mesh.scale(0.02, center=mesh.get_center())  # Scale to ~2cm size (typical LEGO brick)
            mesh.translate(-mesh.get_center())
            
            # Ensure mesh has normals
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            
            print(f"Loaded CAD model with {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
            return mesh
            
        except Exception as e:
            print(f"Error loading STL file {stl_file_path}: {e}")
            return None

    def generate_depth_image_from_viewpoint(self, mesh, x_angle, y_angle, image_size=512, depth_scale=1000):
        """
        Generate depth image from specific viewpoint (x_angle, y_angle)
        x_angle: rotation around x-axis (0-180 degrees)
        y_angle: rotation around y-axis (0-360 degrees)
        """
        # Create rotation matrices
        rx = np.radians(x_angle)
        ry = np.radians(y_angle)
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # Combined rotation matrix
        R_combined = Ry @ Rx
        
        # Apply rotation to mesh
        # Create a copy of the mesh using deepcopy to avoid Open3D version issues
        mesh_rotated = copy.deepcopy(mesh)
        mesh_rotated.rotate(R_combined, center=(0, 0, 0))
        
        # Create virtual camera setup
        # Position camera at distance to view the entire object
        bounds = mesh_rotated.get_axis_aligned_bounding_box()
        max_extent = np.max(bounds.get_extent())
        camera_distance = max_extent * 3.0  # Place camera at 3x the object's max extent
        
        # Camera positioned along positive Z-axis looking towards origin
        camera_position = np.array([0, 0, camera_distance])
        camera_target = np.array([0, 0, 0])
        camera_up = np.array([0, 1, 0])
        
        # Create visualization for depth rendering
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=image_size, height=image_size)
        vis.add_geometry(mesh_rotated)
        
        # Set camera parameters
        ctr = vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        
        # Set extrinsic parameters (camera pose)
        camera_params.extrinsic = np.eye(4)
        camera_params.extrinsic[:3, :3] = np.eye(3)  # Identity rotation (camera looks along -Z)
        camera_params.extrinsic[:3, 3] = camera_position
        
        # Set intrinsic parameters
        focal_length = image_size * 0.8  # Approximate focal length
        camera_params.intrinsic.set_intrinsics(
            image_size, image_size, focal_length, focal_length, 
            image_size/2, image_size/2
        )
        
        ctr.convert_from_pinhole_camera_parameters(camera_params)
        
        # Render and capture depth
        vis.poll_events()
        vis.update_renderer()
        
        depth_image = vis.capture_depth_float_buffer(do_render=True)
        vis.destroy_window()
        
        # Convert depth buffer to numpy array and process
        depth_array = np.asarray(depth_image)
        
        # Convert to meaningful depth values and handle invalid depths
        depth_array[depth_array == 1.0] = 0  # Set invalid depths to 0
        depth_array = depth_array * depth_scale  # Scale to millimeters
        
        return depth_array

    def depth_image_to_point_cloud(self, depth_image, x_angle, y_angle, focal_length=None):
        """Convert depth image back to 3D point cloud"""
        height, width = depth_image.shape
        
        if focal_length is None:
            focal_length = width * 0.8  # Same as used in rendering
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to camera coordinates
        cx, cy = width / 2, height / 2
        z = depth_image / 1000.0  # Convert back to meters
        x = (u - cx) * z / focal_length
        y = (v - cy) * z / focal_length
        
        # Filter out invalid points
        valid_mask = z > 0
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        
        if len(x_valid) == 0:
            return np.array([]).reshape(0, 3)
        
        points_camera = np.column_stack([x_valid, y_valid, z_valid])
        
        # Transform back from camera coordinates to world coordinates
        # Reverse the rotation applied during rendering
        rx = np.radians(x_angle)
        ry = np.radians(y_angle)
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        R_combined = Ry @ Rx
        R_inverse = R_combined.T  # Transpose for inverse rotation
        
        # Apply inverse rotation to get back to world coordinates
        points_world = (R_inverse @ points_camera.T).T
        
        return points_world

    def generate_cad_template_database(self, stl_file_path, x_step=30, y_step=30, 
                                     num_corners_per_view=10, spin_correlation_threshold=0.6):
        """
        Generate comprehensive template database from CAD model
        Following paper's suggestion: x-axis 0-180°, y-axis 0-360°
        
        Args:
            stl_file_path: Path to STL CAD model file
            x_step: Step size for x-axis rotation (degrees)
            y_step: Step size for y-axis rotation (degrees)
            num_corners_per_view: Number of Harris corners to extract per viewpoint
            spin_correlation_threshold: Minimum correlation for template matching
        """
        print("=== GENERATING CAD TEMPLATE DATABASE ===")
        
        # Load CAD model
        mesh = self.load_cad_model(stl_file_path)
        if mesh is None:
            return None
        
        template_database = {
            'templates': [],
            'viewpoints': [],
            'harris_corners': [],
            'spin_images': [],
            'metadata': {
                'source_file': stl_file_path,
                'x_step': x_step,
                'y_step': y_step,
                'total_views': 0
            }
        }
        
        print(f"Generating templates with x_step={x_step}°, y_step={y_step}°")
        
        total_views = 0
        
        # Generate views: x-axis 0-180°, y-axis 0-360°
        for x_angle in range(0, 181, x_step):  # 0 to 180 degrees
            for y_angle in range(0, 360, y_step):  # 0 to 360 degrees
                print(f"Processing viewpoint: x={x_angle}°, y={y_angle}°")
                
                try:
                    # Generate depth image from this viewpoint
                    depth_image = self.generate_depth_image_from_viewpoint(mesh, x_angle, y_angle)
                    
                    # Convert depth image to point cloud
                    points = self.depth_image_to_point_cloud(depth_image, x_angle, y_angle)
                    
                    if len(points) < 50:  # Skip if too few points
                        continue
                    
                    # Extract Harris 3D corners
                    harris_corners = self.compute_harris_3d_corners(points, num_corners=num_corners_per_view)
                    
                    if len(harris_corners) < 3:  # Need minimum corners for pose estimation
                        continue
                    
                    # Compute surface normals for corner points
                    normals = self.compute_surface_normals(points)
                    
                    # Generate spin images for each Harris corner
                    corner_spin_images = []
                    for corner in harris_corners:
                        # Find closest point in original cloud to get normal
                        distances = np.linalg.norm(points - corner, axis=1)
                        closest_idx = np.argmin(distances)
                        corner_normal = normals[closest_idx]
                        
                        # Compute spin image
                        spin_image = self.compute_spin_image(corner, corner_normal, points)
                        corner_spin_images.append(spin_image)
                    
                    # Store template data
                    template_data = {
                        'viewpoint': (x_angle, y_angle),
                        'points': points,
                        'harris_corners': harris_corners,
                        'spin_images': corner_spin_images,
                        'point_count': len(points),
                        'corner_count': len(harris_corners)
                    }
                    
                    template_database['templates'].append(template_data)
                    template_database['viewpoints'].append((x_angle, y_angle))
                    template_database['harris_corners'].append(harris_corners)
                    template_database['spin_images'].append(corner_spin_images)
                    
                    total_views += 1
                    
                except Exception as e:
                    print(f"Error processing viewpoint ({x_angle}°, {y_angle}°): {e}")
                    continue
        
        template_database['metadata']['total_views'] = total_views
        
        print(f"=== TEMPLATE DATABASE COMPLETED ===")
        print(f"Generated {total_views} valid templates from CAD model")
        print(f"Average corners per template: {np.mean([len(corners) for corners in template_database['harris_corners']]):.1f}")
        
        return template_database

    def match_with_cad_templates(self, target_points, template_database, 
                               correlation_threshold=0.6, max_templates_to_try=50):
        """
        Match target point cloud with CAD template database
        Returns best matching template and pose estimation
        """
        print("=== MATCHING WITH CAD TEMPLATE DATABASE ===")
        
        if template_database is None or len(template_database['templates']) == 0:
            print("No valid template database provided")
            return None, None, None
        
        # Extract Harris corners from target
        target_corners = self.compute_harris_3d_corners(target_points, num_corners=15)
        if len(target_corners) < 3:
            print("Insufficient Harris corners in target point cloud")
            return None, None, None
        
        # Compute target normals and spin images
        target_normals = self.compute_surface_normals(target_points)
        target_corner_normals = []
        target_spin_images = []
        
        for corner in target_corners:
            distances = np.linalg.norm(target_points - corner, axis=1)
            closest_idx = np.argmin(distances)
            corner_normal = target_normals[closest_idx]
            target_corner_normals.append(corner_normal)
            
            spin_image = self.compute_spin_image(corner, corner_normal, target_points)
            target_spin_images.append(spin_image)
        
        print(f"Target: {len(target_corners)} corners, {len(target_spin_images)} spin images")
        
        best_match_score = 0
        best_template = None
        best_correspondences = None
        
        # Limit number of templates to try for efficiency
        templates_to_try = min(max_templates_to_try, len(template_database['templates']))
        template_indices = np.random.choice(len(template_database['templates']), 
                                          templates_to_try, replace=False)
        
        print(f"Testing {templates_to_try} templates from database...")
        
        for template_idx in template_indices:
            template = template_database['templates'][template_idx]
            template_corners = template['harris_corners']
            template_spin_images = template['spin_images']
            
            # Find correspondences between target and template spin images
            correspondences = []
            total_correlation = 0
            
            for i, target_spin in enumerate(target_spin_images):
                best_correlation = -1
                best_match_idx = -1
                
                for j, template_spin in enumerate(template_spin_images):
                    correlation = self.compute_spin_image_correlation(target_spin, template_spin)
                    if correlation > best_correlation and correlation > correlation_threshold:
                        best_correlation = correlation
                        best_match_idx = j
                
                if best_match_idx >= 0:
                    correspondences.append((i, best_match_idx, best_correlation))
                    total_correlation += best_correlation
            
            if len(correspondences) >= 3:  # Need minimum correspondences
                match_score = total_correlation / len(correspondences)  # Average correlation
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_template = template
                    best_correspondences = correspondences
        
        if best_template is None:
            print("No suitable template match found")
            return None, None, None
        
        print(f"Best template match: viewpoint {best_template['viewpoint']}, score: {best_match_score:.3f}")
        print(f"Correspondences: {len(best_correspondences)}")
        
        # Perform pose estimation with best template
        Uc = np.array([target_corners[i] for i, j, _ in best_correspondences])
        Wc = np.array([best_template['harris_corners'][j] for i, j, _ in best_correspondences])
        
        R, T = self.updated_alignment_algorithm(Uc, Wc, best_template['points'], target_points)
        
        return R, T, best_template

    def create_simple_brick_template(self, brick_points, subsample_ratio=0.3):
        """
        Fallback: Create a simple template from detected brick points
        This is used when no CAD template database is available
        """
        if len(brick_points) == 0:
            return brick_points
        
        # Subsample points to create a simpler template
        n_template_points = max(100, int(len(brick_points) * subsample_ratio))
        if len(brick_points) > n_template_points:
            indices = np.random.choice(len(brick_points), n_template_points, replace=False)
            template_points = brick_points[indices]
        else:
            template_points = brick_points
        
        print(f"Created simple template with {len(template_points)} points")
        return template_points

    # ========== MAIN PIPELINE IMPLEMENTATION ==========
    def comprehensive_bin_picking_pipeline(self, transformed_file, summary_file,
                                         dbscan_eps=0.01, dbscan_min_samples=10,
                                         template_database=None):
        """
        Complete bin picking pipeline with all 6 steps using PRE-GENERATED template database:
        1. Data acquisition and preprocessing (already done)
        2. Find closest cluster
        3. Color filtering  
        4. Harris 3D + Spin image pose estimation (with pre-generated CAD templates)
        5. World coordinate extraction
        6. Visualization
        
        Args:
            template_database: PRE-GENERATED CAD template database (required for CAD-based pose estimation)
        """
        print("=== COMPREHENSIVE BIN PICKING PIPELINE ===")
        
        # Load preprocessed data
        data = np.loadtxt(transformed_file, delimiter=' ')
        points, colors = data[:, :3], data[:, 3:6]
        
        # Step 2: Find closest cluster
        cluster_points, cluster_colors, cluster_info = self.find_closest_brick_cluster(
            points, colors, dbscan_eps, dbscan_min_samples)
        
        if cluster_points is None:
            print("Pipeline failed: No valid clusters found")
            return []
        
        # Step 3: Color filtering
        filtered_points, filtered_colors = self.filter_by_top_brick_color(
            cluster_points, cluster_colors)
        
        # Step 4: Advanced pose estimation with PRE-GENERATED CAD templates
        R, T, matched_template = None, None, None
        
        if template_database is not None and len(template_database['templates']) > 0:
            print("Using PRE-GENERATED CAD template database for pose estimation...")
            print(f"Template database contains {len(template_database['templates'])} templates")
            R, T, matched_template = self.match_with_cad_templates(filtered_points, template_database)
        else:
            print("No template database provided, using simple template approach...")
        
        # Fallback to simple template if CAD matching fails
        if R is None:
            print("CAD template matching failed, using simple template approach...")
            simple_template = self.create_simple_brick_template(filtered_points)
            R, T = self.estimate_pose_with_spin_images(simple_template, filtered_points)
            matched_template = {'points': simple_template, 'viewpoint': 'simple_template'}
        
        # Step 5: Extract world coordinates
        if R is not None and T is not None:
            world_coordinates = self.extract_world_coordinates(R, T)
            pose_success = True
        else:
            print("All pose estimation methods failed, using cluster centroid as position")
            centroid = np.mean(filtered_points, axis=0)
            world_coordinates = {
                'position': centroid.tolist(),
                'rotation': [0, 0, 0],
                'coordinate_system': 'euler_xyz'
            }
            R, T = np.eye(3), centroid
            pose_success = False
        
        # Step 6: Visualization
        if matched_template is not None and 'points' in matched_template:
            template_points = matched_template['points']
        else:
            template_points = self.create_simple_brick_template(filtered_points)
        
        geometries = self.visualize_matched_template(
            filtered_points, filtered_colors, template_points, R, T)
        
        # Save visualization
        vis_path = summary_file.replace('.txt', '_visualization.png')
        self.save_visualization_image(geometries, vis_path)
        
        # Save results with template information
        results = [{
            'world_coordinates': world_coordinates,
            'cluster_info': cluster_info,
            'pose_estimation_success': pose_success,
            'filtered_points_count': len(filtered_points),
            'template_method': 'CAD' if matched_template and 'viewpoint' in matched_template and matched_template['viewpoint'] != 'simple_template' else 'simple',
            'matched_viewpoint': matched_template.get('viewpoint', None) if matched_template else None,
            'template_database_size': len(template_database['templates']) if template_database else 0
        }]
        
        # Save enhanced summary file
        with open(summary_file, "w") as f:
            coords = world_coordinates['position']
            rotation = world_coordinates['rotation']
            f.write(f"=== BIN PICKING RESULTS ===\n")
            f.write(f"Position (mm): {coords[0]*1000:.2f} {coords[1]*1000:.2f} {coords[2]*1000:.2f}\n")
            f.write(f"Rotation (deg): {rotation[0]:.2f} {rotation[1]:.2f} {rotation[2]:.2f}\n")
            f.write(f"Cluster size: {cluster_info['size']} points\n")
            f.write(f"Filtered points: {len(filtered_points)} points\n")
            f.write(f"Pose estimation: {'Success' if pose_success else 'Failed'}\n")
            f.write(f"Template method: {results[0]['template_method']}\n")
            if results[0]['matched_viewpoint']:
                f.write(f"Matched viewpoint: {results[0]['matched_viewpoint']}\n")
            if results[0]['template_database_size'] > 0:
                f.write(f"Template database size: {results[0]['template_database_size']} templates\n")
        
        print("=== PIPELINE COMPLETED ===")
        return results

    # def send_file_via_tcp(self, file_path):
    #     filename = os.path.basename(file_path).encode('utf-8')
    #     with open(file_path, "rb") as f:
    #         file_data = f.read()
    #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #         s.connect((self.host, self.port))
    #         s.sendall(struct.pack('>I', len(filename)))
    #         s.sendall(filename)
    #         s.sendall(file_data)

    def run_pipeline(self, template_database=None):
        """
        Main pipeline runner implementing the 6-step comprehensive bin picking system:
        1. Data acquisition and preprocessing
        2. Find closest cluster  
        3. Color filtering
        4. Harris 3D + Spin image pose estimation (with PRE-GENERATED CAD templates)
        5. World coordinate extraction
        6. Visualization
        
        Args:
            template_database: PRE-GENERATED template database (use generate_and_save_template_database() first)
        """
        print("=== STARTING COMPREHENSIVE BIN PICKING PIPELINE ===")
        
        if template_database is not None:
            print(f"Using provided template database with {len(template_database['templates'])} templates")
        else:
            print("No template database provided - will use simple template approach")
        
        # Step 1: Data acquisition and preprocessing
        print("Step 1: Data acquisition and preprocessing...")
        points, colors = self.capture_and_preprocess_kinect_data()
        if len(points) == 0:
            print("[PIPELINE] No valid points found after preprocessing. Exiting.")
            return
        
        print(f"Step 1 completed: {len(points)} points acquired and preprocessed")
        
        # Prepare output files
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        transformed_file = os.path.join(self.output_dir, f"Transformed_ROI_point_cloud_{timestamp_str}.txt")
        summary_file = os.path.join(self.output_dir, f"BinPicking_Results_{timestamp_str}.txt")
        image_file = os.path.join(self.output_dir, f"PointCloud_Img_{timestamp_str}.png")
        
        # Save preprocessed data
        print("Saving preprocessed data...")
        self.save_transformed_point_cloud(points, colors, transformed_file)
        self.save_cloud_image(points, colors, image_file)
        
        # Run comprehensive pipeline (Steps 2-6) with pre-generated template database
        print("Running comprehensive bin picking pipeline...")
        results = self.comprehensive_bin_picking_pipeline(
            transformed_file, summary_file,
            dbscan_eps=0.01, 
            dbscan_min_samples=10,
            template_database=template_database
        )
        
        if results:
            print("=== PIPELINE SUCCESS ===")
            world_coords = results[0]['world_coordinates']
            print(f"Final brick position: {world_coords['position']}")
            print(f"Final brick rotation: {world_coords['rotation']}")
            print(f"Template method used: {results[0]['template_method']}")
            if results[0]['matched_viewpoint']:
                print(f"Best matching viewpoint: {results[0]['matched_viewpoint']}")
            
            # Send results to server
            # try:
            #     self.send_file_via_tcp(summary_file)
            #     print("[PIPELINE] Results sent to server successfully.")
            # except Exception as e:
            #     print(f"[WARNING] Failed to send file to server: {e}")
        else:
            print("=== PIPELINE FAILED ===")
            print("No valid brick detected or pose estimation failed")
        
        print("[PIPELINE] All processes completed.")

    def generate_and_save_template_database(self, stl_file_path, output_path, 
                                          x_step=30, y_step=30, 
                                          num_corners_per_view=10):
        """
        STEP 1: Generate template database from CAD model and save it
        This should be run ONCE per brick type to create the template database
        
        Args:
            stl_file_path: Path to STL CAD model file
            output_path: Path to save the template database (e.g., "lego_brick_templates.pkl")
            x_step: Step size for x-axis rotation (degrees) - smaller = more templates, better accuracy
            y_step: Step size for y-axis rotation (degrees) - smaller = more templates, better accuracy
            num_corners_per_view: Number of Harris corners to extract per viewpoint
        
        Returns:
            template_database: The generated database (also saved to file)
        """
        print("=== GENERATING TEMPLATE DATABASE FROM CAD MODEL ===")
        print(f"Input STL file: {stl_file_path}")
        print(f"Output database file: {output_path}")
        print(f"Template resolution: x_step={x_step}°, y_step={y_step}°")
        
        # Generate comprehensive template database
        template_database = self.generate_cad_template_database(
            stl_file_path, 
            x_step=x_step, 
            y_step=y_step, 
            num_corners_per_view=num_corners_per_view
        )
        
        if template_database is None:
            print("Failed to generate template database")
            return None
        
        # Save template database to file
        try:
            with open(output_path, "wb") as f:
                pickle.dump(template_database, f)
            print(f"Template database saved successfully to: {output_path}")
            print(f"Database contains {template_database['metadata']['total_views']} templates")
            print("You can now use this database file in the main bin picking pipeline")
            
        except Exception as e:
            print(f"Error saving template database: {e}")
            return None
        
        return template_database

    def load_template_database(self, database_path):
        """
        Load a pre-generated template database from file
        
        Args:
            database_path: Path to the saved template database file
            
        Returns:
            template_database: The loaded template database
        """
        try:
            with open(database_path, "rb") as f:
                template_database = pickle.load(f)
            
            print(f"Template database loaded successfully from: {database_path}")
            print(f"Database contains {len(template_database['templates'])} templates")
            print(f"Source CAD file: {template_database['metadata']['source_file']}")
            print(f"Template resolution: {template_database['metadata']['x_step']}° x {template_database['metadata']['y_step']}°")
            
            return template_database
            
        except Exception as e:
            print(f"Error loading template database from {database_path}: {e}")
            return None

    def run_pipeline_simple(self):
        """
        Simplified pipeline runner without CAD templates (backward compatibility)
        """
        return self.run_pipeline(template_database=None)
        
if __name__ == "__main__":
    # =============================================================================
    # RECOMMENDED WORKFLOW: Two-Step Process
    # =============================================================================
    
    # STEP 1: Generate Template Database from CAD Model (Run this ONCE per brick type)
    
    # system = BinPickingSystem(wdf_path="")
    # stl_path = "C:\\Users\\FILAB\\Desktop\\DUY\\LegoBrick_4_2.stl" 
    # database_path = "C:\\Users\\FILAB\\Desktop\\DUY\\Templates" 
    
    # Generate template database (this may take several minutes)
    # template_db = system.generate_and_save_template_database(
    #     stl_file_path=stl_path,
    #     output_path=database_path,
    #     x_step=30,  # 30° steps: faster generation, good accuracy
    #     y_step=30   # Use smaller steps (15°, 20°) for higher accuracy but longer generation time
    # )
    
    # STEP 2: Run Main Bin Picking Pipeline with Pre-Generated Templates
    # This is what you run for each bin picking operation:
    
    system = BinPickingSystem(wdf_path="")
    
    # Option A: Load pre-generated template database and run pipeline
    # database_path = "lego_brick_templates.pkl"  # Path to your saved template database
    # template_db = system.load_template_database(database_path)
    # system.run_pipeline(template_database=template_db)
    
    # Option B: Run without CAD templates (simple mode for testing)
    system.run_pipeline_simple()
    
    # =============================================================================
    # EXAMPLE: Complete workflow for first-time setup
    # =============================================================================
    
    # # First time setup (generate templates once)
    # system = BinPickingSystem(wdf_path="")
    # stl_path = "C:/path/to/your/lego_brick.stl"
    # database_path = "C:/path/to/templates/lego_brick_templates.pkl"
    # 
    # print("=== FIRST TIME SETUP: GENERATING TEMPLATE DATABASE ===")
    # template_db = system.generate_and_save_template_database(
    #     stl_file_path=stl_path,
    #     output_path=database_path,
    #     x_step=20,  # Higher resolution templates
    #     y_step=20
    # )
    # 
    # if template_db is not None:
    #     print("=== RUNNING BIN PICKING WITH GENERATED TEMPLATES ===")
    #     system.run_pipeline(template_database=template_db)
    # else:
    #     print("Template generation failed, running simple mode")
    #     system.run_pipeline_simple()
    
    # =============================================================================
    # EXAMPLE: Production usage (templates already generated)
    # =============================================================================
    
    # # Production usage (templates already exist)
    # system = BinPickingSystem(wdf_path="")
    # database_path = "C:/path/to/templates/lego_brick_templates.pkl"
    # 
    # print("=== PRODUCTION MODE: LOADING EXISTING TEMPLATES ===")
    # template_db = system.load_template_database(database_path)
    # 
    # if template_db is not None:
    #     system.run_pipeline(template_database=template_db)
    # else:
    #     print("Failed to load template database, running simple mode")
    #     system.run_pipeline_simple()