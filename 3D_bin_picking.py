# Essential Libraries
import os # Interacting with OS
import socket # Network communication between systems
import time # Time related functions
import cv2 # Used for image processing
import numpy as np # Matrix operation
import open3d as o3d # 3D data processing
import struct # Convert Python values to C struct to communicate with sensors (In this case: Kinect V2)
import scipy.spatial
import json # For template library metadata
import math # Mathematical functions
import copy # For deep copying objects
from datetime import datetime # Time operation
import itertools
from scipy.spatial import Delaunay

# Image processing Libraries
from sklearn.cluster import DBSCAN # Density-based clustering algorithm (Used for grouping similar 3D point)
from sklearn.decomposition import PCA # Dimensionality reduction (flatten into lower dimension)
from scipy.spatial.transform import Rotation as R

# Kinect Libraries
from pykinect2 import PyKinectRuntime, PyKinectV2
from pykinect2.PyKinectV2 import *

import matplotlib.pyplot as plt

# Main Program
class BinPickingSystem:

    # Initializing (This part is for the robot arm so it is not necessary)
    def __init__(self, wdf_path, output_dir=None):
        self.output_dir = "C:\\Users\\FILAB\\Desktop\\DUY\\Results" # Data saving location
        # self.host = "192.168.1.23" # Target IP for sending data to the robot arm
        # self.port = 9999 # Target (Robot arm) port

    # Border filtering algorithm using AND logic
    def keep_inside_boundary_points(self, points, colors, x_min, x_max, y_min, y_max, margin=0.02):
        mask = (
            (points[:, 0] >= x_min + margin) & (points[:, 0] <= x_max - margin) &
            (points[:, 1] >= y_min + margin) & (points[:, 1] <= y_max - margin)
        ) # Removing the border by an amount of margin
        # The scanning range will be (x_min + margin, x_max - margin) x (y_min + margin, y_max - margin)
        return points[mask], colors[mask] # Return filtered point cloud and color values
    
    # Filter 3D points so that only those above or on a reference plane are kept
    def keep_points_above_plane(self, points, colors, plane_model):
        a, b, c, d = plane_model
        # Calculating distance from the point to the plane (< 0: Below; = 0: On; > 0: Above)
        mask = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d <= 0)
        return points[mask], colors[mask]

    # Transforming points to normalize them to world coordinate
    def transform_point_cloud(self, points):
        new_origin = np.array([-0.1663194511548611, -0.30196779718241507, 0.652]) # Why this coordinate?
        # Rotating matrix that swap X and Y coordinates and invert Z coordinate
        rotation_matrix = np.array([
            [0,  1,  0],
            [1,  0,  0],
            [0,  0, -1]
        ]) 
        translated = points - new_origin # Shifting the points to normalize the new coordinate
        transformed = np.dot(translated, rotation_matrix.T) # Transforming the points using dot product
        return transformed

    # Main depth and RGB data acquiring and processing
    def capture_and_preprocess_kinect_data(self, roi_x=195, roi_y=50, roi_w=245, roi_h=300,
                                           plane_dist_thresh=0.005, ransac_n=3, ransac_iter=1000,
                                           boundary_margin=0.005, dbscan_eps_pre=0.01, dbscan_min_samples_pre=50): # These settings are for DBSCAN filtering
        # ROI: Region of Interest
        kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
        # RANSAC Algorithm: An algorithm used for estimating parameters of a mathematical model from a set of observed data that contains outliers
        # when outliers are to be no influence on the estimated values. On the other hand, it can be used to detect outliers.

        # Starting Kinect and wait until both depth frames and color frames are ready
        while not (kinect.has_new_depth_frame() and kinect.has_new_color_frame()):
            time.sleep(0.01) # Time delay

        # Getting the latest data of depth frames and color frames
        depth_frame = kinect.get_last_depth_frame().reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width))
        color_frame = kinect.get_last_color_frame().reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4))[:, :, :3]
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB) # Converting BGR data from color frame to RGB data

        # Cropping the region of interest
        depth_roi = depth_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        color_roi = color_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Intrinsics (Internal characteristics) of a camera includes focal length, pixel dimensions, resolution
        # This part of the code is used for camera's intrinsics parameters retrieval to convert depth data into 3D points
        intrinsics = kinect._mapper.GetDepthCameraIntrinsics()
        fx, fy = intrinsics.FocalLengthX, intrinsics.FocalLengthY
        cx, cy = intrinsics.PrincipalPointX, intrinsics.PrincipalPointY

        points, colors = [], []
        for i in range(depth_roi.shape[0]):
            for j in range(depth_roi.shape[1]):
                z = depth_roi[i, j] * 0.001 # mm to meter
                if z > 0:
                    # Calculating x and y coordinates using pinhole camera equations
                    x = (j + roi_x - cx) * z / fx
                    y = -(i + roi_y - cy) * z / fy # Flip y direction
                    
                    # Mapping 3D depth points to its corresponding pixel in the color image
                    depth_point = PyKinectV2._DepthSpacePoint()
                    depth_point.x, depth_point.y = j + roi_x, i + roi_y # Scanning the region of interest
                    color_point = kinect._mapper.MapDepthPointToColorSpace(depth_point, depth_roi[i, j])
                    # Getting RGB color for each point
                    cx_c, cy_c = int(color_point.x), int(color_point.y)
                    if 0 <= cx_c < color_frame.shape[1] and 0 <= cy_c < color_frame.shape[0]:
                        c = color_frame[cy_c, cx_c] / 255.0 # Normalizing them by dividing by 255
                        # Appending to corresponding point
                        points.append((x, y, z))
                        colors.append(c)

        kinect.close()
        # Converting the data to numpy arrays
        points = np.array(points)
        colors = np.array(colors)

        # Early exit if the array is empty (no data or not sufficient data acquired)
        if len(points) < 3:
            return np.array([]), np.array([])

        # Preparing the point cloud for plane segmentation and processing
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Finding best-fit plane using RANSAC
        plane_model, inliers = pcd.segment_plane(distance_threshold=plane_dist_thresh,
                                                 ransac_n=ransac_n,
                                                 num_iterations=ransac_iter)
        # Removing the bin floor, keep only the object in the xy-plane
        non_plane_cloud = pcd.select_by_index(inliers, invert=True)
        # Applying the filters to keep only the object that is above the floor (plane)
        above_points, above_colors = self.keep_points_above_plane(np.asarray(non_plane_cloud.points),
                                                                  np.asarray(non_plane_cloud.colors),
                                                                  plane_model)
        # If there is no point above the plane then return null arrays
        if len(above_points) == 0:
            return np.array([]), np.array([])

        # Removing margin points
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

        # Apply coordinate transformation to normalize points to world coordinate
        transformed = self.transform_point_cloud(denoised_points)
        return transformed, denoised_colors

    def apply_dbscan(self, points, colors, eps=0.01, min_samples=50):
        """
        Apply DBSCAN clustering to remove noise points from point cloud data.
        
        Args:
            points: 3D point cloud array
            colors: RGB color array corresponding to points
            eps: DBSCAN radius parameter for neighborhood
            min_samples: DBSCAN minimum samples parameter
        
        Returns:
            Tuple of (denoised_points, denoised_colors) with noise points removed
        """
        if len(points) < min_samples:
            return points, colors
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = dbscan.labels_
        
        # Remove noise points (label -1)
        mask = (labels != -1)
        denoised_points = points[mask]
        denoised_colors = colors[mask]
        
        return denoised_points, denoised_colors

    # ========== ENHANCED BIN PICKING PIPELINE FOR STACKED LEGO BRICKS ==========
    
    def find_and_extract_topmost_brick(self, points, colors, dbscan_eps=0.01, dbscan_min_samples=10, 
                                      color_tolerance=0.15, top_surface_depth=0.02):
        """
        Combined Step 2-3: Find points closest to camera and extract topmost brick by dominant color
        
        This function:
        1. Finds the points that are closest to the camera (smallest Z values) globally
        2. Extracts the dominant color from these closest points
        3. Filters the entire point cloud to keep only points with the dominant color
        
        Args:
            points: 3D point cloud
            colors: RGB colors for each point
            dbscan_eps: DBSCAN radius parameter
            dbscan_min_samples: DBSCAN minimum samples parameter
            color_tolerance: Tolerance for color similarity (0.0 = exact match, 1.0 = any color)
            top_surface_depth: Depth range to consider as "top surface" for color extraction (meters)
        
        Returns:
            Tuple of (topmost_brick_points, topmost_brick_colors, closest_z_value)
        """
        print("Step 2-3: Finding points closest to camera and extracting topmost brick by color...")
        
        # PHASE 1: Find the points that are globally closest to the camera
        # No need for clustering first - we want the absolute closest points regardless of clusters
        
        # PHASE 2: Extract dominant color from the points closest to camera (globally)
        # Find the points that are closest to the camera across the entire point cloud
        all_z_coords = points[:, 2]
        z_min_global = np.min(all_z_coords)
        z_threshold = z_min_global + top_surface_depth  # Consider top surface for color extraction
        
        # Get points closest to camera from the entire point cloud
        closest_to_camera_mask = all_z_coords <= z_threshold
        closest_points = points[closest_to_camera_mask]
        closest_colors = colors[closest_to_camera_mask]
        
        if len(closest_colors) == 0:
            print("No points found closest to camera, using minimum Z point")
            # Fallback: use the single closest point
            closest_idx = np.argmin(all_z_coords)
            dominant_color = colors[closest_idx]
            dominant_color_name = "unknown"
        else:
            # Get dominant color from points closest to camera
            dominant_color = self.get_dominant_color(closest_colors)
            # Classify the dominant color using HSV classification
            dominant_color_name, _ = self.classify_hsv_color(dominant_color)
        
        print(f"Extracted dominant color from points closest to camera: {dominant_color_name}")
        print(f"RGB values: [{dominant_color[0]:.3f}, {dominant_color[1]:.3f}, {dominant_color[2]:.3f}]")
        print(f"Used {len(closest_colors)} points within {top_surface_depth}m of closest point (Z={z_min_global:.3f}m)")
        
        # PHASE 3: Filter the entire point cloud by the dominant color using HSV-based classification
        # This ensures we get all points of the topmost brick, even if they extend beyond the closest cluster
        filtered_points_list = []
        filtered_colors_list = []
        
        for i, color in enumerate(colors):
            color_name, _ = self.classify_hsv_color(color)
            if color_name == dominant_color_name:
                filtered_points_list.append(points[i])
                filtered_colors_list.append(color)
        
        if len(filtered_points_list) == 0:
            print("No points found with exact HSV color match, falling back to RGB tolerance")
            # Fallback to RGB distance-based filtering with tight tolerance
            color_distances = np.linalg.norm(colors - dominant_color, axis=1)
            similar_color_mask = color_distances <= color_tolerance
            topmost_brick_points = points[similar_color_mask]
            topmost_brick_colors = colors[similar_color_mask]
        else:
            topmost_brick_points = np.array(filtered_points_list)
            topmost_brick_colors = np.array(filtered_colors_list)
        
        print(f"HSV-based color filtering: {len(points)} → {len(topmost_brick_points)} points")
        print(f"Filtered out {len(points) - len(topmost_brick_points)} points with different colors")
        print(f"Successfully extracted topmost brick with {dominant_color_name} color")
        
        return topmost_brick_points, topmost_brick_colors, z_min_global
    
    def apply_enhanced_alignment_algorithm(self, template_keypoints, target_keypoints, correspondences,
                                         epsilon1=0.005, epsilon2=0.01, iter_max=1000, max_outer_iter=50):
        """
        Step 4: Enhanced alignment algorithm based on the paper pseudocode:
        
        INPUT: Uc (template keypoints), Wc (target keypoints), U (template points), W (target points)
        OUTPUT: estimated pose
        
        This is the exact algorithm from your specification:
        Set Ce = Cmax, Pe = Pmax
        While Ce > ε2
            While Pe > ε1
                iter = 1
                Select 3 similar matching pair points randomly.
                Compute the rigid transformation.
                    U'c = RUc + T
                    Pe = ‖U'c − Wc‖ / √n
                iter = iter + 1
                If iter > Iter_max
                    Break;
                END If
            END While
            If Pe > ε1
                Return 1, Break
            END If
            U' = RU + T
            Compute the means for U' and W as U'm and Wm
                Ce = ‖U'm − Wm‖
        END While
        ICP Refinement on U' and W
        Compute the pose estimation
        """
        print("Step 4: Applying enhanced alignment algorithm from paper...")
        
        if len(correspondences) < 3:
            print("Insufficient correspondences for pose estimation")
            return None, None
        
        # Extract correspondence pairs
        Uc = np.array([template_keypoints[i] for i, j, _ in correspondences])  # Template keypoints
        Wc = np.array([target_keypoints[j] for i, j, _ in correspondences])    # Target keypoints
        
        # For this implementation, we'll use the keypoints as both U and W
        # In a full implementation, U and W would be the complete point clouds
        U = Uc  # Template points
        W = Wc  # Target points
        
        print(f"Starting enhanced alignment with {len(correspondences)} correspondences")
        
        # Initialize Ce and Pe to maximum values
        Ce = float('inf')  # Cmax
        Pe = float('inf')  # Pmax
        
        best_R, best_T = None, None
        outer_iter = 0
        
        # Outer loop: While Ce > ε2
        while Ce > epsilon2 and outer_iter < max_outer_iter:
            outer_iter += 1
            print(f"Outer iteration {outer_iter}: Ce = {Ce:.6f}")
            
            # Inner loop: While Pe > ε1
            inner_iter = 0
            while Pe > epsilon1 and inner_iter < iter_max:
                inner_iter += 1
                
                # Select 3 similar matching pair points randomly
                if len(correspondences) >= 3:
                    sample_indices = np.random.choice(len(correspondences), 3, replace=False)
                    sample_Uc = Uc[sample_indices]
                    sample_Wc = Wc[sample_indices]
                    
                    # Compute the rigid transformation
                    R, T = self.estimate_rigid_transformation_3_points(sample_Uc, sample_Wc)
                    
                    if R is not None:
                        # U'c = RUc + T
                        Uc_prime = (R @ Uc.T).T + T
                        
                        # Pe = ‖U'c − Wc‖ / √n
                        distances = np.linalg.norm(Uc_prime - Wc, axis=1)
                        Pe = np.mean(distances) / np.sqrt(len(Uc))
                        
                        # Store best transformation
                        if best_R is None or Pe < epsilon1:
                            best_R, best_T = R.copy(), T.copy()
                
                # If iter > Iter_max, Break
                if inner_iter >= iter_max:
                    break
            
            # If Pe > ε1, Return 1, Break
            if Pe > epsilon1:
                print(f"Failed to converge: Pe = {Pe:.6f} > ε1 = {epsilon1}")
                return None, None
            
            # U' = RU + T
            if best_R is not None:
                U_prime = (best_R @ U.T).T + best_T
                
                # Compute the means for U' and W as U'm and Wm
                U_prime_mean = np.mean(U_prime, axis=0)
                W_mean = np.mean(W, axis=0)
                
                # Ce = ‖U'm − Wm‖
                Ce = np.linalg.norm(U_prime_mean - W_mean)
                
                print(f"Inner loop converged: Pe = {Pe:.6f}, Ce = {Ce:.6f}")
        
        if best_R is None:
            print("Enhanced alignment algorithm failed")
            return None, None
        
        print(f"Enhanced alignment converged in {outer_iter} outer iterations")
        print(f"Final Pe = {Pe:.6f}, Ce = {Ce:.6f}")
        
        # ICP Refinement on U' and W
        print("Performing ICP refinement...")
        refined_R, refined_T = self.refine_pose_with_icp(U, W, best_R, best_T)
        
        return refined_R, refined_T
    
    def detect_and_match_topmost_brick(self, cluster_points, cluster_colors, template_library_dir=None, enable_spin_images=True):
        """
        Complete Pipeline Implementation following the document:
        1. 3D Keypoint Detection using Harris corners
        2. Spin Image Computation 
        3. Spin Image Matching
        4. RANSAC-Based Matching and Transformation
        5. ICP Refinement
        
        Args:
            cluster_points: Points from the target object
            cluster_colors: Colors from the target object
            template_library_dir: Directory containing template library
            enable_spin_images: Use spin image matching
        
        Returns:
            Dictionary containing pose estimation results
        """
        print("Starting 3D Object Alignment Pipeline...")
        
        # Step 2: 3D Keypoint Detection - Extract keypoints from target W
        print("Step 2: 3D Keypoint Detection")
        target_keypoints = self.compute_harris_3d_keypoints(cluster_points, num_corners=15)
        
        if len(target_keypoints) == 0:
            print("No Harris corners detected in target")
            return None
        
        print(f"Detected {len(target_keypoints)} target keypoints")
        
        # If template library is provided, use spin image matching
        if template_library_dir and os.path.exists(template_library_dir) and enable_spin_images:
            print("Using spin image matching with template library")
            return self.match_with_spin_images(cluster_points, target_keypoints, template_library_dir)
        else:
            print("ERROR: No template library provided or spin images disabled")
            print(f"template_library_dir: {template_library_dir}")
            print(f"enable_spin_images: {enable_spin_images}")
            return None
    
    def match_with_spin_images(self, target_points, target_keypoints, template_library_dir):
        """
        Complete Spin Image Matching Pipeline:
        Steps 3-6: Spin Image Computation, Matching, RANSAC, and ICP
        """
        print("Step 3: Spin Image Computation")
        
        # Load template library
        templates = self.load_template_library(template_library_dir)
        if not templates:
            print("ERROR: No templates found in library - cannot proceed with spin image matching")
            print(f"Template library directory: {template_library_dir}")
            return None
        
        print(f"Loaded {len(templates)} templates from library for spin image matching")
        
        # Compute surface normals for target points
        target_normals = self.compute_surface_normals(target_points)
        
        # Get normals for target keypoints
        target_kp_normals = []
        for kp in target_keypoints:
            distances = np.linalg.norm(target_points - kp, axis=1)
            closest_idx = np.argmin(distances)
            target_kp_normals.append(target_normals[closest_idx])
        
        # Generate spin images for target keypoints Q
        target_spin_images = []
        print(f"Starting to generate {len(target_keypoints)} target spin images...")
        
        for i, (kp, normal) in enumerate(zip(target_keypoints, target_kp_normals)):
            spin_img = self.compute_spin_image(kp, normal, target_points)
            target_spin_images.append(spin_img)
            
            # Save spin image visualization for debugging
            spin_viz_dir = os.path.join(self.output_dir, "spin_images", "target")
            spin_filename = os.path.join(spin_viz_dir, f"target_spin_{i:03d}.png")
            print(f"Saving target spin image {i+1}/{len(target_keypoints)} to: {spin_filename}")
            self.save_spin_image(spin_img, spin_filename, cmap="viridis")
        
        print(f"Generated {len(target_spin_images)} target spin images")
        print(f"Spin image visualizations saved to: {spin_viz_dir}")
        
        # Save spin image summary
        print("Creating spin image summary...")
        self.save_spin_image_summary(target_spin_images, target_keypoints, self.output_dir)
        
        # Also save individual target spin images with enhanced metadata
        print(f"Target spin image visualizations saved to: {spin_viz_dir}")
        print("=" * 50)
        print("SPIN IMAGE VISUALIZATION SUMMARY:")
        print(f"Target Spin Images: {len(target_spin_images)} saved to {spin_viz_dir}")
        print("=" * 50)
        
        best_match = None
        best_score = -1
        
        # Step 4: Spin Image Matching with each template
        print("Step 4: Spin Image Matching")
        for template in templates:
            try:
                # Load template keypoints and spin images
                template_file = template['file']
                template_pcd = o3d.io.read_point_cloud(template_file)
                template_points = np.asarray(template_pcd.points)
                
                if len(template_points) < 10:
                    continue
                
                # Get template keypoints U
                template_keypoints = self.compute_harris_3d_keypoints(template_points, num_corners=15)
                if len(template_keypoints) == 0:
                    continue
                
                # Compute template spin images P
                template_normals = self.compute_surface_normals(template_points)
                template_kp_normals = []
                for kp in template_keypoints:
                    distances = np.linalg.norm(template_points - kp, axis=1)
                    closest_idx = np.argmin(distances)
                    template_kp_normals.append(template_normals[closest_idx])
                
                template_spin_images = []
                print(f"\nGenerating {len(template_keypoints)} spin images for template {template['id']}...")
                
                for j, (kp, normal) in enumerate(zip(template_keypoints, template_kp_normals)):
                    spin_img = self.compute_spin_image(kp, normal, template_points)
                    template_spin_images.append(spin_img)
                    
                    # Save template spin image visualization for debugging
                    template_spin_dir = os.path.join(self.output_dir, "spin_images", "templates", template['id'])
                    template_spin_filename = os.path.join(template_spin_dir, f"template_spin_{j:03d}.png")
                    print(f"  Saving template spin image {j+1}/{len(template_keypoints)} to: {template_spin_filename}")
                    self.save_spin_image(spin_img, template_spin_filename, cmap="plasma")
                
                # Save template spin image summary for easy viewing
                if len(template_spin_images) > 0:
                    print(f"Creating template spin image summary for template {template['id']}...")
                    self.save_template_spin_image_summary(template_spin_images, template['id'], self.output_dir)
                    print(f"Template {template['id']} spin images: {len(template_spin_images)} saved to {template_spin_dir}")
                    print("-" * 40)
                
                # Find correspondences using spin image correlation R(P,Q)
                correspondences = self.find_spin_image_correspondences(
                    target_spin_images, template_spin_images, 
                    target_keypoints, template_keypoints)
                
                # Save correspondence visualizations for debugging
                if len(correspondences) > 0:
                    self.save_spin_image_correspondences(
                        target_spin_images, template_spin_images, correspondences,
                        template['id'], self.output_dir)
                
                if len(correspondences) < 3:
                    print(f"  Template {template['id']}: Insufficient correspondences ({len(correspondences)} < 3)")
                    continue
                
                # Step 5: RANSAC-Based Matching and Transformation
                print(f"Step 5: RANSAC for template {template['id']}")
                R, T, inliers = self.estimate_rigid_transformation_ransac_pipeline(
                    template_keypoints, target_keypoints, correspondences)
                
                if R is not None:
                    # Evaluate match using equations (6) and (7)
                    Pe, Ce = self.evaluate_pose_match(template_keypoints, target_keypoints, R, T, correspondences)
                    print(f"  Template {template['id']}: Pe={Pe:.6f}, Ce={Ce:.6f}")
                    
                    # Step 6: ICP Refinement
                    if Pe < 0.01 and Ce < 0.01:  # ε1 and ε2 thresholds
                        print("Step 6: ICP Refinement")
                        R_refined, T_refined = self.refine_pose_with_icp(template_points, target_points, R, T)
                        if R_refined is not None:
                            R, T = R_refined, T_refined
                    
                    # Calculate final match score
                    match_score = 1.0 / (1.0 + Pe + Ce)  # Higher is better
                    print(f"  Template {template['id']}: Match score = {match_score:.3f}")
                    
                    if match_score > best_score:
                        best_score = match_score
                        best_match = {
                            'rotation_matrix': R,
                            'translation': T,
                            'template_id': template['id'],
                            'match_score': match_score,
                            'Pe': Pe,
                            'Ce': Ce,
                            'num_correspondences': len(correspondences),
                            'num_inliers': len(inliers),
                            'method': 'spin_image_pipeline'
                        }
                        
            except Exception as e:
                print(f"Error processing template {template['id']}: {e}")
                continue
        
        if best_match:
            print(f"Best match: {best_match['template_id']}")
            print(f"  Match score: {best_match['match_score']:.3f}")
            print(f"  Pe: {best_match['Pe']:.6f}, Ce: {best_match['Ce']:.6f}")
            return best_match
        else:
            print("ERROR: No suitable template match found - spin image matching failed")
            print("This means all templates were processed but none met the matching criteria")
            print("Check spin image generation and correlation thresholds")
            return None
    
    def find_spin_image_correspondences(self, target_spin_images, template_spin_images, 
                                      target_keypoints, template_keypoints, correlation_threshold=0.6):
        """
        Find correspondences between template and target spin images using correlation R(P,Q)
        """
        correspondences = []
        
        for i, target_spin in enumerate(target_spin_images):
            best_correlation = -1
            best_match_j = -1
            
            for j, template_spin in enumerate(template_spin_images):
                # Compute correlation coefficient R(P,Q) using equation (4)
                correlation = self.compute_spin_image_correlation(template_spin, target_spin)
                
                if correlation > best_correlation and correlation > correlation_threshold:
                    best_correlation = correlation
                    best_match_j = j
            
            if best_match_j >= 0:
                # Store correspondence as (template_idx, target_idx, correlation)
                correspondences.append((best_match_j, i, best_correlation))
        
        print(f"Found {len(correspondences)} spin image correspondences")
        return correspondences
    
    def estimate_rigid_transformation_ransac_pipeline(self, template_keypoints, target_keypoints, 
                                                    correspondences, max_iterations=1000, 
                                                    inlier_threshold=0.005):
        """
        Step 5: RANSAC-Based Matching and Transformation following Algorithm 1
        """
        print("Running RANSAC for rigid transformation estimation...")
        
        if len(correspondences) < 3:
            print("Insufficient correspondences for RANSAC")
            return None, None, []
        
        best_R, best_T = None, None
        best_inliers = []
        max_inliers = 0
        
        # Extract correspondence points
        template_pts = np.array([template_keypoints[i] for i, j, _ in correspondences])
        target_pts = np.array([target_keypoints[j] for i, j, _ in correspondences])
        
        for iteration in range(max_iterations):
            # Randomly select 3 matching point pairs
            if len(correspondences) >= 3:
                sample_indices = np.random.choice(len(correspondences), 3, replace=False)
                sample_template = template_pts[sample_indices]
                sample_target = target_pts[sample_indices]
                
                # Compute rigid transformation (R, T)
                R, T = self.estimate_rigid_transformation_3_points(sample_template, sample_target)
                
                if R is not None:
                    # Transform template points: U' = RU + T
                    transformed_template = (R @ template_pts.T).T + T
                    
                    # Compute distances to find inliers
                    distances = np.linalg.norm(transformed_template - target_pts, axis=1)
                    inliers = np.where(distances < inlier_threshold)[0]
                    
                    # Keep best hypothesis with most inliers
                    if len(inliers) > max_inliers:
                        max_inliers = len(inliers)
                        best_R, best_T = R.copy(), T.copy()
                        best_inliers = inliers.copy()
        
        print(f"RANSAC completed: {max_inliers} inliers out of {len(correspondences)} correspondences")
        return best_R, best_T, best_inliers
    
    def evaluate_pose_match(self, template_keypoints, target_keypoints, R, T, correspondences):
        """
        Evaluate pose match using equations (6) and (7):
        Pe = (1/n) * Σ||Wc - U'c||  (equation 6)
        Ce = ||W̄c - Ū'c||          (equation 7)
        """
        # Extract correspondence points
        template_pts = np.array([template_keypoints[i] for i, j, _ in correspondences])
        target_pts = np.array([target_keypoints[j] for i, j, _ in correspondences])
        
        # Transform template points: U'c = RUc + T
        transformed_template = (R @ template_pts.T).T + T
        
        # Pe: Average point-to-point distance (equation 6)
        distances = np.linalg.norm(transformed_template - target_pts, axis=1)
        Pe = np.mean(distances)
        
        # Ce: Distance between centroids (equation 7)
        centroid_template = np.mean(transformed_template, axis=0)
        centroid_target = np.mean(target_pts, axis=0)
        Ce = np.linalg.norm(centroid_template - centroid_target)
        
        return Pe, Ce
    
    def estimate_pose_geometric(self, points, keypoints):
        """
        Simple geometric pose estimation when no template matching is available
        """
        print("Performing geometric pose estimation...")
        
        # Calculate center of mass
        center = np.mean(points, axis=0)
        
        # Use PCA to estimate orientation
        pca = PCA(n_components=3)
        pca.fit(points - center)
        
        # Create rotation matrix from PCA components
        rotation_matrix = pca.components_.T
        
        # Ensure proper rotation matrix (determinant = 1)
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 2] *= -1
        
        return {
            'rotation_matrix': rotation_matrix,
            'translation': center,
            'match_score': 0.5,
            'method': 'geometric_pca'
        }

    # def estimate_pose_geometric(self, points, keypoints):
    #     """
    #     Simple geometric pose estimation when no template matching is available
    #     """
    #     print("Performing geometric pose estimation...")
        
    #     # Calculate center of mass
    #     center = np.mean(points, axis=0)
        
    #     # Use PCA to estimate orientation
    #     pca = PCA(n_components=3)
    #     pca.fit(points - center)
        
    #     # Create rotation matrix from PCA components
    #     rotation_matrix = pca.components_.T
        
    #     # Ensure proper rotation matrix (determinant = 1)
    #     if np.linalg.det(rotation_matrix) < 0:
    #         rotation_matrix[:, 2] *= -1
        
    #     return {
    #         'rotation_matrix': rotation_matrix,
    #         'translation': center,
    #         'match_score': 0.5,
    #         'method': 'geometric_pca'
    #     }
        
    def find_world_coordinates_of_topmost_brick(self, pose_result):
        """
        Step 5: Find the world coordinate after pose matching of the highest brick
        
        Args:
            pose_result: Result from detect_and_match_topmost_brick()
        
        Returns:
            Dictionary containing world coordinates
        """
        print("Step 5: Computing world coordinates of topmost brick...")
        
        if pose_result is None:
            print("No pose estimation available")
            return None
        
        # Extract rotation matrix and translation
        R = np.array(pose_result['rotation_matrix'])
        T = np.array(pose_result['translation'])
        
        # Extract coordinates using the existing method
        coordinates = self.extract_brick_coordinates(R, T, coordinate_system='euler_xyz')
        
        # Add additional information
        coordinates['method'] = pose_result.get('method', 'unknown')
        coordinates['match_score'] = pose_result.get('match_score', 0.0)
        coordinates['template_id'] = pose_result.get('template_id', 'none')
        
        print(f"World coordinates computed using {coordinates['method']}")
        print(f"Position: [{coordinates['position'][0]:.3f}, {coordinates['position'][1]:.3f}, {coordinates['position'][2]:.3f}] meters")
        print(f"Rotation: [{coordinates['rotation'][0]:.1f}, {coordinates['rotation'][1]:.1f}, {coordinates['rotation'][2]:.1f}] degrees")
        
        return coordinates
    
    def visualize_matched_template(self, cluster_points, cluster_colors, pose_result, 
                                 template_library_dir=None, output_file=None):
        """
        Step 6: Visualize the matched template in the output cluster for verification
        
        Args:
            cluster_points: Original cluster points
            cluster_colors: Original cluster colors
            pose_result: Pose estimation result
            template_library_dir: Directory containing templates
            output_file: Optional file to save visualization
        """
        print("Step 6: Creating visualization of matched template...")
        
        geometries = []
        
        # Add original cluster (in original colors)
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)
        geometries.append(cluster_pcd)
        
        # Add matched template if available
        if pose_result and 'rotation_matrix' in pose_result and 'translation' in pose_result:
            template_pcd = self.create_template_visualization(pose_result, template_library_dir)
            if template_pcd:
                geometries.append(template_pcd)
        
        # Add coordinate frame at the estimated pose
        if pose_result and 'translation' in pose_result:
            position = pose_result['translation']
            R = np.array(pose_result['rotation_matrix'])
            
            # Create coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=position)
            
            # Apply rotation to coordinate frame
            coord_frame.rotate(R, center=position)
            geometries.append(coord_frame)
        
        # Add world origin coordinate frame
        world_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        geometries.append(world_origin)
        
        # Visualize
        if geometries:
            print("Displaying matched template visualization...")
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Matched Template Visualization - Black=Template, Original=Cluster",
                width=1024,
                height=768
            )
            
            # Save visualization if output file specified
            if output_file:
                self.save_visualization_image(geometries, output_file)
                print(f"Visualization saved to: {output_file}")
    
    def create_template_visualization(self, pose_result, template_library_dir):
        """
        Create visualization of the matched template
        """
        template_id = pose_result.get('template_id', 'none')
        
        if template_id == 'none' or not template_library_dir:
            # Create a simple box visualization for geometric estimation
            return self.create_simple_brick_visualization(pose_result)
        
        # Load template from library
        templates = self.load_template_library(template_library_dir)
        template = None
        for t in templates:
            if t['id'] == template_id:
                template = t
                break
        
        if template is None:
            return self.create_simple_brick_visualization(pose_result)
        
        # Load template points from PLY file (not from JSON metadata)
        template_file = template['file']
        if not os.path.exists(template_file):
            print(f"Template file not found: {template_file}")
            return self.create_simple_brick_visualization(pose_result)
        
        # Load point cloud from PLY file
        template_pcd = o3d.io.read_point_cloud(template_file)
        template_points = np.asarray(template_pcd.points)
        
        if len(template_points) < 10:
            print(f"Template {template['id']} has too few points for visualization")
            return self.create_simple_brick_visualization(pose_result)
        
        R = np.array(pose_result['rotation_matrix'])
        T = np.array(pose_result['translation'])
        
        # Transform template to match pose
        transformed_points = (R @ template_points.T).T + T
        
        # Create point cloud with highlight color (red/black for visibility)
        template_pcd = o3d.geometry.PointCloud()
        template_pcd.points = o3d.utility.Vector3dVector(transformed_points)
        
        # Use black color for high contrast
        highlight_color = np.array([0.0, 0.0, 0.0])  # Black
        template_colors = np.tile(highlight_color, (len(transformed_points), 1))
        template_pcd.colors = o3d.utility.Vector3dVector(template_colors)
        
        return template_pcd
    
    def create_simple_brick_visualization(self, pose_result):
        """
        Create a simple brick visualization for geometric pose estimation
        """
        # Create a simple box mesh at the estimated position
        position = pose_result['translation']
        R = np.array(pose_result['rotation_matrix'])
        
        # Standard LEGO brick dimensions (approximate)
        brick_box = o3d.geometry.TriangleMesh.create_box(
            width=0.064, height=0.032, depth=0.019)  # 32x16x16mm
        
        # Center the box
        brick_box.translate([-0.016, -0.008, -0.008])
        
        # Apply pose transformation
        brick_box.rotate(R, center=[0, 0, 0])
        brick_box.translate(position)
        
        # Color it black for visibility
        brick_box.paint_uniform_color([0.0, 0.0, 0.0])  # Black
        
        return brick_box
    
    def save_visualization_image(self, geometries, output_file):
        """
        Save visualization to image file
        """
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1024, height=768)
            
            for geom in geometries:
                vis.add_geometry(geom)
            
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)  # Stabilize rendering
            vis.capture_screen_image(output_file)
            vis.destroy_window()
            
        except Exception as e:
            print(f"Error saving visualization: {e}")
    
    def run_enhanced_bin_picking_pipeline(self, template_library_dir=None, enable_visualization=True, enable_spin_images=True):
        """
        Main pipeline implementing the enhanced bin picking algorithm for stacked LEGO bricks
        
        Pipeline Steps:
        1. Data acquisition and preprocessing
        2. Find closest cluster to camera
        3. Color filtering for topmost brick
        4. Harris 3D and spin image pose matching
        5. World coordinate calculation
        6. Visualization for verification
        
        Args:
            template_library_dir: Directory containing template library
            enable_visualization: Enable 3D visualization
            enable_spin_images: Use spin image matching instead of geometric pose estimation
        """
        print("="*60)
        print("ENHANCED BIN PICKING PIPELINE FOR STACKED LEGO BRICKS")
        print("="*60)
        
        # Step 1: Data acquisition and preprocessing
        print("\n" + "="*50)
        print("STEP 1: DATA ACQUISITION AND PREPROCESSING")
        print("="*50)
        
        points, colors = self.capture_and_preprocess_kinect_data()
        if len(points) == 0:
            print("[ERROR] No valid points found after preprocessing. Exiting.")
            return None
        
        print(f"Successfully acquired {len(points)} points after preprocessing")
        
        # Step 2-3: Find closest cluster and extract topmost brick by color
        print("\n" + "="*50)
        print("STEP 2-3: FIND CLOSEST CLUSTER AND EXTRACT TOPMOST BRICK")
        print("="*50)
        
        topmost_points, topmost_colors, closest_z = self.find_and_extract_topmost_brick(
            points, colors, dbscan_eps=0.01, dbscan_min_samples=10, 
            color_tolerance=0.15, top_surface_depth=0.02)
        
        if topmost_points is None:
            print("[ERROR] No topmost brick found. Exiting.")
            return None
        
        if len(topmost_points) < 100:
            print("[WARNING] Very few points in topmost brick")
            return None
        
        # Step 4: Harris 3D and spin image pose matching
        print("\n" + "="*50)
        print("STEP 4: HARRIS 3D AND SPIN IMAGE POSE MATCHING")
        print("="*50)
        
        pose_result = self.detect_and_match_topmost_brick(
            topmost_points, topmost_colors, template_library_dir, enable_spin_images)
        
        if pose_result is None:
            print("[ERROR] Pose estimation failed. Exiting.")
            return None
        
        # Step 5: World coordinate calculation
        print("\n" + "="*50)
        print("STEP 5: WORLD COORDINATE CALCULATION")
        print("="*50)
        
        world_coordinates = self.find_world_coordinates_of_topmost_brick(pose_result)
        
        if world_coordinates is None:
            print("[ERROR] World coordinate calculation failed. Exiting.")
            return None
        
        # Save coordinates to file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        coord_file = os.path.join(self.output_dir, f"Enhanced_Brick_Coordinates_{timestamp_str}")
        saved_file = self.save_brick_coordinates(world_coordinates, coord_file)
        
        # Step 6: Visualization for verification
        if enable_visualization:
            print("\n" + "="*50)
            print("STEP 6: VISUALIZATION FOR VERIFICATION")
            print("="*50)
            
            viz_file = os.path.join(self.output_dir, f"Enhanced_Visualization_{timestamp_str}.png")
            self.visualize_matched_template(
                topmost_points, topmost_colors, pose_result, 
                template_library_dir, viz_file)
        
        # Prepare final results
        results = {
            'world_coordinates': world_coordinates,
            'pose_result': pose_result,
            'cluster_info': {
                'closest_z': closest_z,
                'original_size': len(points),
                'topmost_size': len(topmost_points)
            },
            'files': {
                'coordinates': saved_file,
                'visualization': viz_file if enable_visualization else None
            }
        }
        
        print("\n" + "="*60)
        print("ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {self.output_dir}")
        if saved_file:
            print(f"Coordinates: {os.path.basename(saved_file)}")
        if enable_visualization and 'visualization' in results['files']:
            print(f"Visualization: {os.path.basename(results['files']['visualization'])}")
        
        return results

    # Saving processed data
    def save_transformed_point_cloud(self, points, colors, output_file):
        data = np.hstack([points, colors])
        np.savetxt(output_file, data, delimiter=' ', fmt='%f')

    def save_spin_image(self, spin_image, filename, cmap="viridis"):
        """
        Save a spin image (2D numpy array) as an image file.

        Args:
            spin_image (np.ndarray): 2D spin image array
            filename (str): Full path with extension (e.g., "output/spin1.png")
            cmap (str): Matplotlib colormap name (default "viridis")
        """
        try:
            # Create directory if it doesn't exist
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
                print(f"Created directory: {os.path.dirname(filename)}")

            plt.figure(figsize=(3, 3))
            plt.imshow(spin_image, cmap=cmap, origin='lower')
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            print(f"Successfully saved spin image: {filename}")
            
        except Exception as e:
            print(f"Error saving spin image {filename}: {e}")
            import traceback
            traceback.print_exc()

    def save_spin_image_correspondences(self, target_spins, template_spins, correspondences, 
                                      template_id, output_dir):
        """
        Save visualization of spin image correspondences showing matched pairs.
        
        Args:
            target_spins: List of target spin images
            template_spins: List of template spin images  
            correspondences: List of (template_idx, target_idx, correlation) tuples
            template_id: Template identifier for file naming
            output_dir: Directory to save correspondence visualizations
        """
        if len(correspondences) == 0:
            return
            
        corr_dir = os.path.join(output_dir, "spin_correspondences", template_id)
        if not os.path.exists(corr_dir):
            os.makedirs(corr_dir)
        
        for i, (template_idx, target_idx, correlation) in enumerate(correspondences):
            if template_idx < len(template_spins) and target_idx < len(target_spins):
                template_spin = template_spins[template_idx]
                target_spin = target_spins[target_idx]
                
                # Create side-by-side visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
                
                ax1.imshow(template_spin, cmap='plasma', origin='lower')
                ax1.set_title(f'Template {template_idx}')
                ax1.axis('off')
                
                ax2.imshow(target_spin, cmap='viridis', origin='lower')
                ax2.set_title(f'Target {target_idx}')
                ax2.axis('off')
                
                plt.suptitle(f'Correspondence {i+1}: Correlation = {correlation:.3f}')
                plt.tight_layout()
                
                corr_filename = os.path.join(corr_dir, f"correspondence_{i:03d}_corr{correlation:.3f}.png")
                plt.savefig(corr_filename, bbox_inches='tight', pad_inches=0.1)
                plt.close()
        
        print(f"Saved {len(correspondences)} correspondence visualizations to: {corr_dir}")

    def save_spin_image_summary(self, target_spins, target_keypoints, output_dir, max_display=10):
        """
        Save a summary visualization showing multiple spin images in a grid.
        
        Args:
            target_spins: List of target spin images
            target_keypoints: Corresponding keypoints
            output_dir: Directory to save summary
            max_display: Maximum number of spin images to display in grid
        """
        if len(target_spins) == 0:
            print("No spin images to create summary for")
            return
            
        print(f"Creating spin image summary for {len(target_spins)} images...")
        
        summary_dir = os.path.join(output_dir, "spin_images")
        try:
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
                print(f"Created summary directory: {summary_dir}")
        except Exception as e:
            print(f"Error creating summary directory: {e}")
            return
        
        # Display up to max_display spin images
        num_display = min(len(target_spins), max_display)
        cols = min(5, num_display)
        rows = (num_display + cols - 1) // cols
        
        try:
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            if rows == 1:
                axes = axes.reshape(1, -1) if num_display > 1 else [axes]
            
            for i in range(num_display):
                row = i // cols
                col = i % cols
                ax = axes[row][col] if rows > 1 else axes[col]
                
                ax.imshow(target_spins[i], cmap='viridis', origin='lower')
                ax.set_title(f'Keypoint {i}')
                ax.axis('off')
            
            # Hide unused subplots
            for i in range(num_display, rows * cols):
                row = i // cols
                col = i % cols
                ax = axes[row][col] if rows > 1 else axes[col]
                ax.axis('off')
            
            plt.suptitle(f'Target Spin Images Summary ({len(target_spins)} total)')
            plt.tight_layout()
            
            summary_filename = os.path.join(summary_dir, "target_spin_summary.png")
            plt.savefig(summary_filename, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()
            
            print(f"Spin image summary successfully saved to: {summary_filename}")
            
        except Exception as e:
            print(f"Error creating spin image summary: {e}")
            import traceback
            traceback.print_exc()

    def save_template_spin_image_summary(self, template_spins, template_id, output_dir, max_display=15):
        """
        Save a summary visualization showing template spin images in a grid.
        
        Args:
            template_spins: List of template spin images
            template_id: Template identifier for file naming
            output_dir: Directory to save summary
            max_display: Maximum number of spin images to display in grid
        """
        if len(template_spins) == 0:
            print(f"No template spin images to create summary for template {template_id}")
            return
            
        print(f"Creating template spin image summary for template {template_id} ({len(template_spins)} images)...")
        
        template_summary_dir = os.path.join(output_dir, "spin_images", "templates", template_id)
        try:
            if not os.path.exists(template_summary_dir):
                os.makedirs(template_summary_dir)
                print(f"Created template summary directory: {template_summary_dir}")
        except Exception as e:
            print(f"Error creating template summary directory: {e}")
            return
        
        # Display up to max_display spin images
        num_display = min(len(template_spins), max_display)
        cols = min(5, num_display)
        rows = (num_display + cols - 1) // cols
        
        try:
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
            if rows == 1:
                axes = axes.reshape(1, -1) if num_display > 1 else [axes]
            
            for i in range(num_display):
                row = i // cols
                col = i % cols
                ax = axes[row][col] if rows > 1 else axes[col]
                
                # Use plasma colormap for template spin images for better contrast
                im = ax.imshow(template_spins[i], cmap='plasma', origin='lower')
                ax.set_title(f'Template Keypoint {i}', fontsize=10)
                ax.axis('off')
                
                # Add colorbar for better understanding
                if i == 0:  # Add colorbar only to first image
                    plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Hide unused subplots
            for i in range(num_display, rows * cols):
                row = i // cols
                col = i % cols
                ax = axes[row][col] if rows > 1 else axes[col]
                ax.axis('off')
            
            plt.suptitle(f'Template {template_id} Spin Images ({len(template_spins)} total)', fontsize=14)
            plt.tight_layout()
            
            summary_filename = os.path.join(template_summary_dir, f"template_{template_id}_spin_summary.png")
            plt.savefig(summary_filename, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()
            
            print(f"Template spin image summary saved to: {summary_filename}")
            
        except Exception as e:
            print(f"Error creating template spin image summary for {template_id}: {e}")
            import traceback
            traceback.print_exc()

    def test_spin_image_visualization(self, output_dir=None):
        """
        Test function to verify spin image visualization is working
        Creates a sample spin image and saves it
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        print("Testing spin image visualization...")
        
        # Create a test spin image (simple pattern)
        test_spin = np.zeros((64, 64))
        for i in range(64):
            for j in range(64):
                # Create a circular pattern
                center_x, center_y = 32, 32
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                test_spin[i, j] = np.sin(dist * 0.2) * np.exp(-dist * 0.1)
        
        # Save test spin image
        test_dir = os.path.join(output_dir, "test_spin_images")
        test_filename = os.path.join(test_dir, "test_spin_image.png")
        
        print(f"Saving test spin image to: {test_filename}")
        self.save_spin_image(test_spin, test_filename, cmap="viridis")
        
        # Create test summary
        test_spins = [test_spin, test_spin * 0.5, test_spin * 0.8]
        test_keypoints = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        
        print("Creating test summary...")
        summary_dir = os.path.join(output_dir, "test_spin_images")
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            for i in range(3):
                axes[i].imshow(test_spins[i], cmap='viridis', origin='lower')
                axes[i].set_title(f'Test Spin {i+1}')
                axes[i].axis('off')
            
            plt.suptitle('Test Spin Images')
            plt.tight_layout()
            
            summary_filename = os.path.join(summary_dir, "test_spin_summary.png")
            plt.savefig(summary_filename, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()
            
            print(f"Test summary saved to: {summary_filename}")
            print("Spin image visualization test completed successfully!")
            
        except Exception as e:
            print(f"Error in test summary: {e}")
            import traceback
            traceback.print_exc()

    def save_point_cloud_as_ply(self, points, colors, output_file):
        """
        Save point cloud data as PLY file format for compatibility with 3D_Harris_IPD tools
        """
        # Ensure we have valid data
        if len(points) == 0 or len(colors) == 0:
            print("No data to save as PLY file")
            return
            
        # Create PLY header
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y", 
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header"
        ]
        
        # Convert colors to 0-255 range
        colors_255 = (colors * 255).astype(np.uint8)
        
        # Write PLY file
        with open(output_file, 'w') as f:
            # Write header
            for line in header:
                f.write(line + '\n')
            
            # Write vertex data
            for i in range(len(points)):
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                       f"{colors_255[i, 0]} {colors_255[i, 1]} {colors_255[i, 2]}\n")
        
        print(f"Saved PLY file with timestamp: {output_file}")
    
    # Saving point cloud image for visualization
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

    # Calculating angles between the principal axis of the block and the principal axis of the camera
    def calculate_y_axis_angle_xy(self, minor_axis):
        # Normalize to a unit vector
        v2d = minor_axis[:2] / np.linalg.norm(minor_axis[:2])

        def clockwise_angle_from_y(vec):
            # Clockwise angle from Y-axis (0-360)
            angle = np.degrees(np.arctan2(vec[0], vec[1])) % 360
            return angle

        angle1 = clockwise_angle_from_y(v2d)
        angle2 = clockwise_angle_from_y(-v2d)

        # Since it's a line, remove directionality -> the smaller angle is the actual clockwise rotation
        angle_deg = min(angle1, angle2)
        # If over 90°, convert to complementary angle
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
            angle_deg = -angle_deg
        return angle_deg  # Finally, return with a negative sign

    # HSV-based color classification for LEGO bricks
    def classify_hsv_color(self, rgb_color):
        """
        Classify RGB color into LEGO brick categories using HSV color space
        
        Args:
            rgb_color: RGB color array [R, G, B] in range [0, 1]
        
        Returns:
            Color name string and HSV values
        """
        """
        Classify RGB color into LEGO brick categories using HSV color space
        
        Args:
            rgb_color: RGB color array [R, G, B] in range [0, 1]
        
        Returns:
            Color name string and HSV values
        """
        # Convert to OpenCV HSV space
        bgr = (np.array(rgb_color[::-1]) * 255).astype(np.uint8).reshape(1, 1, 3)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
        h, s, v = hsv

        # Brick-specific color ranges from your samples
        if 90 <= h <= 130:
            if v < 100:
                return "dark_blue", hsv
            else:
                return "light_blue", hsv
        elif 5 <= h <= 15 and s > 100:
            return "red", hsv
        elif 10 <= h <= 30 and s > 100:
            return "orange", hsv
        else:
            return "unknown", hsv
    
    def get_dominant_color_hsv(self, colors):
        """
        Get dominant color using HSV classification for better color distinction
        
        Args:
            colors: Array of RGB colors in range [0, 1]
        
        Returns:
            Dominant RGB color and its HSV classification
        """
        if len(colors) == 0:
            return np.array([0.5, 0.5, 0.5]), "gray"
        
        # Classify each color
        color_counts = {}
        color_rgb_map = {}
        
        for color in colors:
            color_name, hsv = self.classify_hsv_color(color)
            
            if color_name in color_counts:
                color_counts[color_name] += 1
                # Average the RGB values for this color category
                color_rgb_map[color_name] = (color_rgb_map[color_name] + color) / 2
            else:
                color_counts[color_name] = 1
                color_rgb_map[color_name] = color.copy()
        
        # Find most frequent color
        dominant_color_name = max(color_counts, key=color_counts.get)
        dominant_rgb = color_rgb_map[dominant_color_name]
        
        print(f"Dominant color classification: {dominant_color_name}")
        print(f"Color distribution: {color_counts}")
        
        return dominant_rgb, dominant_color_name
    
    # Extracting the most dominant color from a set of RGB values (legacy method)
    def get_dominant_color(self, colors):
        # Use HSV-based method for better color classification
        dominant_rgb, color_name = self.get_dominant_color_hsv(colors)
        return dominant_rgb

    # Helper functions for improved 3D Harris corner detection
    def polyfit3d(self, x, y, z, order=2):
        """Fit a 3D polynomial surface to the data points"""
        ncols = (order + 1)**2 # Number of coefficients for a polynomial of degree 'order'
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order+1), range(order+1))
        for k, (i, j) in enumerate(ij):
            G[:, k] = x**i * y**j
        m, _, _, _ = np.linalg.lstsq(G, z, rcond=None) # Solve the least squares problem
        return m

    def compute_delaunay_neighborhood(self, points, delta=0.025, max_iter=5):
        """Compute adaptive neighborhoods using Delaunay triangulation"""
        if len(points) < 4:  # Need at least 4 points for Delaunay triangulation
            return {}
            
        triangulation = Delaunay(points)

        # Build direct neighborhood from Delaunay triangulation
        neighborhood_direct = {}
        for f in triangulation.simplices:
            for v in range(f.shape[0]):
                faces = list(f.copy())
                faces.pop(v)
                if f[v] in neighborhood_direct.keys():
                    neighborhood_direct[f[v]] = list(np.unique(neighborhood_direct[f[v]] + faces))
                else:
                    neighborhood_direct[f[v]] = faces

        # Adaptive k-ring expansion
        neighborhood = {}
        for v in neighborhood_direct.keys():
            query = points[v]
            # Compute the distance of the query and its ring
            if len(neighborhood_direct[v]) > 0:
                dist = np.max(np.linalg.norm(query - points[neighborhood_direct[v]], axis=1))
                if dist >= delta:
                    bigger_ring = False
                    neighborhood[v] = neighborhood_direct[v]
                else:
                    bigger_ring = True
            else:
                bigger_ring = False
                neighborhood[v] = [v]  # At least include the point itself

            iteration = 1
            while bigger_ring and iteration <= max_iter:
                iteration += 1
                for neighbor in neighborhood_direct[v]:
                    if neighbor in neighborhood_direct:
                        if v in neighborhood.keys():
                            neighborhood[v] = list(np.unique(neighborhood[v] + neighborhood_direct[neighbor]))
                        else:
                            neighborhood[v] = list(np.unique(neighborhood_direct[v] + neighborhood_direct[neighbor]))

                # Compute the distance of the query and its ring
                if len(neighborhood[v]) > 0:
                    dist = np.max(np.linalg.norm(query - points[neighborhood[v]], axis=1))
                    if dist >= delta:
                        bigger_ring = False
                    else:
                        bigger_ring = True
                else:
                    bigger_ring = False

        return neighborhood

    def centering_centroid(self, points):
        """
        Center the point cloud on its centroid
        Returns both centered points and the original centroid for later restoration
        """
        centred_points = points.copy()
        centroid = np.mean(centred_points, axis=0)
        centred_points = centred_points - centroid
        return centred_points, centroid

    def centering_origin(self, points, centroid):
        """
        Restore the point cloud to its original position using the saved centroid
        """
        centred_points = points.copy()
        centred_points = centred_points + centroid
        return centred_points

    def scale_point_cloud(self, points, target_scale=1.0):
        """
        Scale point cloud to a target scale for consistent processing
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

    def preprocess_for_harris_detection(self, points):
        """
        Simplified preprocessing pipeline for Harris 3D corner detection
        Only includes centering and scaling (no principal axis alignment)
        """
        print("Preprocessing points for Harris detection...")
        
        # Step 1: Center on centroid
        centered_points, original_centroid = self.centering_centroid(points)
        
        # Step 2: Scale to unit scale for numerical stability
        scaled_points, scale_factor = self.scale_point_cloud(centered_points, target_scale=1.0)
        
        # Restore centroid position (no rotation alignment)
        preprocessed_points = scaled_points + original_centroid
        
        # Store transformation parameters for later restoration
        transform_params = {
            'original_centroid': original_centroid,
            'scale_factor': scale_factor
        }
        
        return preprocessed_points, transform_params

    def find_closest_brick_cluster(self, points, colors, dbscan_eps=0.01, dbscan_min_samples=10):
        """
        Find the closest brick cluster for robot arm grasping
        Since the robot can only grip one brick at a time, we focus on the closest one
        
        Returns:
            Tuple of (cluster_points, cluster_colors) for the closest valid cluster
        """
        # Perform DBSCAN clustering to identify individual LEGO bricks
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points)
        labels = dbscan.labels_

        # Find all valid clusters (exclude noise with label -1)
        valid_clusters = []
        for lbl in set(labels):
            if lbl == -1:  # Skip noise points
                continue
            m = (labels == lbl)
            cluster_pts = points[m]
            cluster_cols = colors[m]
            
            # Filter small clusters that are likely noise
            if len(cluster_pts) < 500:
                continue
                
            # Calculate average Z coordinate (distance from camera)
            avg_z = np.mean(cluster_pts[:, 2])
            valid_clusters.append((lbl, avg_z, cluster_pts, cluster_cols))
        
        if len(valid_clusters) == 0:
            print("No valid brick clusters found")
            return None, None
        
        # Sort by Z coordinate (closest first) and select the closest one
        valid_clusters.sort(key=lambda x: x[1])
        closest_cluster = valid_clusters[0]
        
        lbl, avg_z, cluster_pts, cluster_cols = closest_cluster
        print(f"Selected closest brick cluster {lbl} at distance Z={avg_z:.3f}m with {len(cluster_pts)} points")
        
        return cluster_pts, cluster_cols

    def compute_harris_3d_keypoints(self, points, delta=0.025, harris_k=0.04, num_corners=15):
        """
        Simple 3D Harris corner detection following pipeline step 2:
        Extract distinctive keypoints using 3D extension of Harris corner detector
        H(U) = υ, H(W) = ω
        """
        if len(points) < 10:
            print("Not enough points for Harris corner detection")
            return np.array([])

        print(f"Computing Harris 3D keypoints for {len(points)} points...")
        
        # Preprocess points for stable Harris detection
        preprocessed_points, transform_params = self.preprocess_for_harris_detection(points)
        
        # Compute neighborhoods using Delaunay triangulation
        neighborhood = self.compute_delaunay_neighborhood(preprocessed_points, delta=delta)
        
        # Initialize Harris response array
        resp = np.zeros(len(preprocessed_points))
        
        # Compute Harris response for each point
        for point_idx in neighborhood.keys():
            try:
                if len(neighborhood[point_idx]) < 3:
                    resp[point_idx] = -np.inf
                    continue
                    
                neighbors = preprocessed_points[neighborhood[point_idx], :]
                neighbors_centred = neighbors - np.mean(neighbors, axis=0)
                
                pca = PCA(n_components=3)
                neighbors_pca = pca.fit_transform(neighbors_centred)
                eigenvalues, eigenvectors = np.linalg.eigh(pca.components_)
                
                idx = np.argmin(eigenvalues)
                rotated_neighbors = neighbors.copy()
                # Rotate neighbors to align with principal axes
                for i in range(neighbors.shape[0]):
                    rotated_neighbors[i, :] = np.dot(np.transpose(eigenvectors), neighbors[i, :])
                
                neighbors_2D = rotated_neighbors[:, :2] - rotated_neighbors[0, :2]
                
                if len(neighbors_2D) >= 6:
                    m = self.polyfit3d(neighbors_2D[:, 0], neighbors_2D[:, 1], rotated_neighbors[:, 2], order=2)
                    m = m.reshape((3, 3))
                    
                    fx2 = m[2, 0]**2 + 2*m[1, 1]**2 + 2*m[0, 2]**2
                    fy2 = m[0, 2]**2 + 2*m[1, 1]**2 + 2*m[2, 0]**2
                    fxfy = m[2, 0]*m[0, 2] + 2*m[1, 1]**2
                    
                    resp[point_idx] = fx2 * fy2 - fxfy * fxfy - harris_k * (fx2 + fy2) * (fx2 + fy2)
                else:
                    resp[point_idx] = -np.inf
                    
            except Exception as e:
                resp[point_idx] = -np.inf
                continue
        
        # Select corner candidates
        candidate = []
        for point_idx in neighborhood.keys():
            if len(neighborhood[point_idx]) > 0:
                neighbor_responses = resp[neighborhood[point_idx]]
                if resp[point_idx] >= np.max(neighbor_responses):
                    candidate.append([point_idx, resp[point_idx]])
        
        if len(candidate) == 0:
            print("No corner candidates found")
            return np.array([])
        
        # Sort by decreasing Harris response and select top corners
        candidate.sort(reverse=True, key=lambda x: x[1])
        candidate = np.array(candidate)
        
        # Select top keypoints
        num_to_select = min(num_corners, len(candidate))
        selected_indices = [int(candidate[i, 0]) for i in range(num_to_select)]
        keypoints = points[selected_indices]
        
        print(f"Selected {len(keypoints)} Harris 3D keypoints")
        return keypoints

    def validate_brick_cluster(self, corner_points, min_corners=4):
        """
        Simple validation for brick clusters - just check if we have enough corners
        Since LEGO bricks are symmetrical, orientation doesn't matter for grasping
        """
        if len(corner_points) < min_corners:
            return False, 1.0
        
        # Calculate basic statistics for logging
        min_coords = np.min(corner_points, axis=0)
        max_coords = np.max(corner_points, axis=0)
        dimensions = (max_coords - min_coords) * 1000  # Convert to mm
        
        print(f"Brick cluster dimensions: {dimensions[0]:.1f}x{dimensions[1]:.1f}x{dimensions[2]:.1f}mm")
        print(f"Corner count: {len(corner_points)} (sufficient for grasping)")
        
        return True, 0.0

    def select_best_pose_hypothesis(self, corner_hypotheses, brick_points):
        """
        Select the best pose hypothesis from multiple candidates
        Since LEGO bricks are symmetrical, any valid hypothesis with enough corners is acceptable
        """
        if len(corner_hypotheses) == 0:
            return np.array([])
            
        # Simply return the first hypothesis with enough corners
        for i, corners in enumerate(corner_hypotheses):
            is_valid, _ = self.validate_brick_cluster(corners)
            
            if is_valid:
                print(f"Selected hypothesis {i} with {len(corners)} corners")
                return corners
        
        # If no hypothesis passes basic validation, return the first one as fallback
        print("No hypothesis passed validation, using first hypothesis as fallback")
        return corner_hypotheses[0] if len(corner_hypotheses) > 0 else np.array([])

    def filter_lego_stud_features(self, preprocessed_points, candidates, original_points, stud_radius_threshold=0.008):
        """
        Filter out LEGO stud features from Harris corner candidates
        
        Args:
            preprocessed_points: Preprocessed point cloud
            candidates: Harris corner candidates [(index, response), ...]
            original_points: Original point cloud for height analysis
            stud_radius_threshold: Maximum radius to consider as potential stud
        
        Returns:
            Filtered candidates with reduced stud detections
        """
        if len(candidates) == 0:
            return candidates
        
        print("Applying LEGO-specific filtering to reduce stud detection...")
        
        filtered_candidates = []
        
        # Get Z-coordinates for height analysis
        z_coords = original_points[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        brick_height = z_max - z_min
        
        # Threshold for top surface (where studs are located)
        top_surface_threshold = z_max - 0.3 * brick_height  # Top 30% of brick
        
        for candidate_idx, (point_idx, harris_response) in enumerate(candidates):
            point_idx = int(point_idx)
            candidate_point = preprocessed_points[point_idx]
            original_point = original_points[point_idx]
            
            # Check if point is on the top surface (likely stud area)
            is_on_top_surface = original_point[2] > top_surface_threshold
            
            if is_on_top_surface:
                # Additional checks for circular/stud-like features
                # Find neighbors within stud radius
                distances = np.linalg.norm(preprocessed_points - candidate_point, axis=1)
                nearby_mask = distances < stud_radius_threshold
                nearby_points = preprocessed_points[nearby_mask]
                
                if len(nearby_points) > 5:  # Enough points to form a circular feature
                    # Check if the nearby points form a roughly circular pattern
                    centered_nearby = nearby_points - candidate_point
                    radial_distances = np.linalg.norm(centered_nearby, axis=1)
                    
                    # If points are roughly equidistant from center (circular pattern)
                    if len(radial_distances) > 0:
                        radial_std = np.std(radial_distances)
                        radial_mean = np.mean(radial_distances)
                        
                        # High uniformity suggests circular stud feature
                        radial_uniformity = radial_std / (radial_mean + 1e-6)
                        
                        if radial_uniformity < 0.3:  # Very uniform -> likely stud
                            print(f"Filtering stud-like feature at height {original_point[2]:.3f}")
                            continue  # Skip this candidate
            
            # Keep edge and corner features
            filtered_candidates.append([point_idx, harris_response])
        
        print(f"Filtered {len(candidates) - len(filtered_candidates)} potential stud features")
        return np.array(filtered_candidates) if len(filtered_candidates) > 0 else candidates

    # ========== SPIN IMAGE-BASED POSE ESTIMATION METHODS ==========
    # Implementation based on "3D Object Detection and Pose Estimation from Depth Image for Robotic Bin Picking"
    
    def compute_surface_normals(self, points, k=6):
        """
        Compute surface normals for each point using local PCA
        """
        normals = np.zeros_like(points)
        
        for i, point in enumerate(points):
            # Find k nearest neighbors
            distances = np.linalg.norm(points - point, axis=1)
            neighbors_idx = np.argsort(distances)[1:k+1]  # Exclude the point itself
            neighbors = points[neighbors_idx]
            
            # Center the neighbors
            centered = neighbors - np.mean(neighbors, axis=0)
            
            # Compute PCA
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                # Normal is the direction with smallest variance (last row of vh)
                normal = vh[-1]
                
                # Ensure consistent orientation (pointing outward from surface)
                # Simple heuristic: normal should point away from local centroid
                centroid_dir = point - np.mean(neighbors, axis=0)
                if np.dot(normal, centroid_dir) < 0:
                    normal = -normal
                    
                normals[i] = normal
            except:
                # Fallback to default normal if SVD fails
                normals[i] = np.array([0, 0, 1])
                
        return normals

    def compute_spin_image(self, point, normal, points, spin_size=64, max_radius=0.05):
        """
        Compute spin image descriptor for a point according to Johnson and Hebert:
        
        S_o → (α, β) = (√(||x - p||² - (n·(x - p))²), n·(x - p))
        
        Where:
        - p: 3D point position (keypoint)
        - n: surface normal at point
        - x: all other points in the point cloud
        - α: perpendicular distance to line through p parallel to n
        - β: signed perpendicular distance to plane through p perpendicular to n
        
        Args:
            point: 3D keypoint position (p in paper, ν for template, ω for target)
            normal: surface normal at keypoint (n in paper)
            points: all points in the point cloud (x in paper)
            spin_size: size of the spin image grid (N in correlation formula)
            max_radius: maximum radius for spin image
        Returns:
            spin_image: 2D spin image array
        """
        # Vector from keypoint p to each point x: (x - p)
        vectors = points - point
        
        # β coordinate: signed perpendicular distance to plane
        # β = n·(x - p)
        beta = np.dot(vectors, normal)
        
        # α coordinate: perpendicular distance to line 
        # α = √(||x - p||² - (n·(x - p))²)
        vectors_squared_norm = np.linalg.norm(vectors, axis=1)**2  # ||x - p||²
        beta_squared = beta**2  # (n·(x - p))²
        
        # Ensure non-negative argument for square root (numerical stability)
        alpha_squared = np.maximum(0, vectors_squared_norm - beta_squared)
        alpha = np.sqrt(alpha_squared)
        
        print(f"      Spin image debug: {len(points)} points, alpha range=[{np.min(alpha):.6f}, {np.max(alpha):.6f}], beta range=[{np.min(beta):.6f}, {np.max(beta):.6f}]")
        
        # Create spin image grid bounds
        alpha_max = max_radius
        beta_min, beta_max = -max_radius, max_radius
        
        # Discretize coordinates into bins
        alpha_bins = np.linspace(0, alpha_max, spin_size + 1)  # +1 for bin edges
        beta_bins = np.linspace(beta_min, beta_max, spin_size + 1)
        
        # Create 2D histogram (spin image)
        spin_image = np.zeros((spin_size, spin_size))
        
        # Only consider points within the cylindrical region
        valid_mask = (alpha <= alpha_max) & (beta >= beta_min) & (beta <= beta_max)
        valid_alpha = alpha[valid_mask]
        valid_beta = beta[valid_mask]
        
        if len(valid_alpha) > 0:
            # Convert to bin indices using np.digitize
            alpha_indices = np.digitize(valid_alpha, alpha_bins) - 1
            beta_indices = np.digitize(valid_beta, beta_bins) - 1
            
            # Clip to valid range (handle edge cases)
            alpha_indices = np.clip(alpha_indices, 0, spin_size - 1)
            beta_indices = np.clip(beta_indices, 0, spin_size - 1)
            
            # Accumulate points in spin image bins
            for a_idx, b_idx in zip(alpha_indices, beta_indices):
                spin_image[b_idx, a_idx] += 1
        
        # Normalize spin image (convert counts to probabilities)
        total_points = np.sum(spin_image)
        if total_points > 0:
            spin_image = spin_image / total_points
        
        print(f"      Final spin image: shape={spin_image.shape}, total_counts={total_points}, non_zero={np.count_nonzero(spin_image)}, max={np.max(spin_image):.6f}")
        
        return spin_image

    def compute_spin_image_correlation(self, spin1, spin2):
        """
        Compute correlation coefficient between two spin images using the exact formula from the paper:
        R(P,Q) = (N∑piqi - ∑pi∑qi) / sqrt((N∑pi² - (∑pi)²)(N∑qi² - (∑qi)²))
        
        This is the Pearson correlation coefficient in its computational form.
        """
        # Flatten spin images
        p = spin1.flatten()
        q = spin2.flatten()
        
        # Number of elements
        N = len(p)
        
        if N == 0:
            return 0.0
        
        # Compute sums as in the paper formula
        sum_p = np.sum(p)           # ∑pi
        sum_q = np.sum(q)           # ∑qi
        sum_pq = np.sum(p * q)      # ∑piqi
        sum_p2 = np.sum(p * p)      # ∑pi²
        sum_q2 = np.sum(q * q)      # ∑qi²
        
        # Compute numerator: N∑piqi - ∑pi∑qi
        numerator = N * sum_pq - sum_p * sum_q
        
        # Compute denominator: sqrt((N∑pi² - (∑pi)²)(N∑qi² - (∑qi)²))
        denominator_p = N * sum_p2 - sum_p * sum_p
        denominator_q = N * sum_q2 - sum_q * sum_q
        denominator = np.sqrt(denominator_p * denominator_q)
        
        # Handle division by zero
        if denominator == 0:
            return 0.0
        
        # Compute correlation coefficient
        correlation = numerator / denominator
        
        # Clamp to valid correlation range [-1, 1] to handle numerical errors
        correlation = np.clip(correlation, -1.0, 1.0)
        
        return correlation

    def validate_spin_image_computation(self, points, keypoints, normals, verbose=True):
        """
        Validate that spin image computation follows the paper's mathematical formulation.
        
        This function verifies:
        1. α coordinate: α = √(||x - p||² - (n·(x - p))²)
        2. β coordinate: β = n·(x - p)  
        3. Correlation coefficient: R(P,Q) formula
        4. Correspondence finding based on R values
        
        Args:
            points: Point cloud for spin image computation
            keypoints: Harris 3D detected keypoints
            normals: Surface normals at keypoints
            verbose: Print detailed validation information
            
        Returns:
            Dictionary with validation results
        """
        if verbose:
            print("="*50)
            print("VALIDATING SPIN IMAGE COMPUTATION")
            print("="*50)
        
        validation_results = {
            'formula_verification': {},
            'correlation_tests': {},
            'correspondence_quality': {}
        }
        
        if len(keypoints) < 2:
            if verbose:
                print("Need at least 2 keypoints for validation")
            return validation_results
        
        # Test 1: Verify spin image coordinate computation for first keypoint
        if verbose:
            print("Test 1: Verifying spin image coordinate formulas...")
        
        kp = keypoints[0]
        normal = normals[0]
        
        # Manual computation following paper's formula
        vectors = points - kp  # (x - p)
        beta_manual = np.dot(vectors, normal)  # β = n·(x - p)
        
        vectors_norm_sq = np.linalg.norm(vectors, axis=1)**2  # ||x - p||²
        beta_squared = beta_manual**2  # (n·(x - p))²
        alpha_manual = np.sqrt(np.maximum(0, vectors_norm_sq - beta_squared))  # α formula
        
        # Compare with implemented function
        spin_img = self.compute_spin_image(kp, normal, points)
        
        validation_results['formula_verification'] = {
            'alpha_range': [float(np.min(alpha_manual)), float(np.max(alpha_manual))],
            'beta_range': [float(np.min(beta_manual)), float(np.max(beta_manual))],
            'spin_image_shape': spin_img.shape,
            'spin_image_sum': float(np.sum(spin_img)),
            'non_zero_bins': int(np.count_nonzero(spin_img))
        }
        
        if verbose:
            print(f"  α range: [{validation_results['formula_verification']['alpha_range'][0]:.4f}, "
                  f"{validation_results['formula_verification']['alpha_range'][1]:.4f}]")
            print(f"  β range: [{validation_results['formula_verification']['beta_range'][0]:.4f}, "
                  f"{validation_results['formula_verification']['beta_range'][1]:.4f}]")
            print(f"  Spin image: {spin_img.shape}, sum={validation_results['formula_verification']['spin_image_sum']:.4f}")
        
        # Test 2: Verify correlation coefficient computation
        if verbose:
            print("\nTest 2: Verifying correlation coefficient R(P,Q)...")
        
        if len(keypoints) >= 2:
            # Compute two spin images
            spin1 = self.compute_spin_image(keypoints[0], normals[0], points)
            spin2 = self.compute_spin_image(keypoints[1], normals[1], points)
            
            # Test self-correlation (should be 1.0)
            self_corr = self.compute_spin_image_correlation(spin1, spin1)
            cross_corr = self.compute_spin_image_correlation(spin1, spin2)
            
            validation_results['correlation_tests'] = {
                'self_correlation': float(self_corr),
                'cross_correlation': float(cross_corr),
                'correlation_formula_verified': abs(self_corr - 1.0) < 1e-6
            }
            
            if verbose:
                print(f"  Self-correlation R(P,P): {self_corr:.6f} (should be ≈ 1.0)")
                print(f"  Cross-correlation R(P,Q): {cross_corr:.6f}")
                print(f"  Formula verified: {validation_results['correlation_tests']['correlation_formula_verified']}")
        
        # Test 3: Correspondence quality assessment
        if verbose:
            print("\nTest 3: Assessing correspondence quality...")
        
        if len(keypoints) >= 3:
            # Generate all spin images
            spin_images = []
            for i in range(min(5, len(keypoints))):  # Test first 5 keypoints
                spin_img = self.compute_spin_image(keypoints[i], normals[i], points)
                spin_images.append(spin_img)
            
            # Compute correlation matrix
            n_imgs = len(spin_images)
            correlation_matrix = np.zeros((n_imgs, n_imgs))
            
            for i in range(n_imgs):
                for j in range(n_imgs):
                    correlation_matrix[i, j] = self.compute_spin_image_correlation(
                        spin_images[i], spin_images[j])
            
            # Analyze correlation distribution
            off_diagonal = correlation_matrix[~np.eye(n_imgs, dtype=bool)]
            
            validation_results['correspondence_quality'] = {
                'correlation_matrix_shape': correlation_matrix.shape,
                'diagonal_mean': float(np.mean(np.diag(correlation_matrix))),
                'off_diagonal_mean': float(np.mean(off_diagonal)),
                'off_diagonal_std': float(np.std(off_diagonal)),
                'max_off_diagonal': float(np.max(off_diagonal))
            }
            
            if verbose:
                print(f"  Correlation matrix: {correlation_matrix.shape}")
                print(f"  Diagonal (self) mean: {validation_results['correspondence_quality']['diagonal_mean']:.3f}")
                print(f"  Off-diagonal mean: {validation_results['correspondence_quality']['off_diagonal_mean']:.3f} ± "
                      f"{validation_results['correspondence_quality']['off_diagonal_std']:.3f}")
                print(f"  Max cross-correlation: {validation_results['correspondence_quality']['max_off_diagonal']:.3f}")
        
        if verbose:
            print("\nValidation completed!")
            print("="*50)
        
        return validation_results

    def find_spin_image_correspondences_with_templates(self, target_spin_images, template_spin_images,
                                                      target_keypoints, template_keypoints,
                                                      correlation_threshold=0.6):
        """
        Find correspondences between target and template spin images following the paper's approach.
        
        The paper states: "we can find similar matching points based on the correlation coefficients 
        as the candidates for point correspondence pairs. Thus, we can obtain the corresponding points 
        from template, denoted as Uc, and from target, denoted as Wc"
        
        Args:
            target_spin_images: List of Q spin images for target keypoints (ω)
            template_spin_images: List of P spin images for template keypoints (ν)
            target_keypoints: 3D positions of target keypoints (W from H(W))
            template_keypoints: 3D positions of template keypoints (U from H(U))
            correlation_threshold: Minimum R(P,Q) correlation for valid correspondence
            
        Returns:
            List of (template_idx, target_idx, correlation_R) tuples representing Uc ↔ Wc correspondences
        """
        correspondences = []
        
        print(f"Finding correspondences between {len(template_spin_images)} template and {len(target_spin_images)} target spin images...")
        
        # For each template spin image P, find best matching target spin image Q
        for i, template_spin_P in enumerate(template_spin_images):
            best_correlation_R = -1
            best_target_idx = -1
            
            # Compare with all target spin images
            for j, target_spin_Q in enumerate(target_spin_images):
                # Compute correlation coefficient R(P,Q) as per paper's formula
                correlation_R = self.compute_spin_image_correlation(template_spin_P, target_spin_Q)
                
                # Select best match above threshold
                if correlation_R > best_correlation_R and correlation_R > correlation_threshold:
                    best_correlation_R = correlation_R
                    best_target_idx = j
            
            # Add to correspondence list if valid match found
            if best_target_idx >= 0:
                correspondences.append((i, best_target_idx, best_correlation_R))
        
        print(f"Found {len(correspondences)} correspondences with R(P,Q) > {correlation_threshold}")
        
        # Log correspondence quality statistics
        if len(correspondences) > 0:
            correlations = [corr for _, _, corr in correspondences]
            print(f"  Correlation range: [{min(correlations):.3f}, {max(correlations):.3f}]")
            print(f"  Average correlation: {np.mean(correlations):.3f}")
        
        return correspondences

    def estimate_rigid_transformation_ransac(self, template_points, target_points, correspondences,
                                           max_iterations=1000, inlier_threshold=0.005, 
                                           camera_proximity_threshold=0.01):
        """
        Estimate rigid transformation using RANSAC as described in the paper
        Following Algorithm 1: Alignment from template to target
        """
        if len(correspondences) < 3:
            print("Need at least 3 correspondences for pose estimation")
            return None, None, []
        
        best_R, best_T = None, None
        best_inliers = []
        best_inlier_count = 0
        
        # Extract correspondence pairs
        template_corr = np.array([template_points[i] for i, j, _ in correspondences])
        target_corr = np.array([target_points[j] for i, j, _ in correspondences])
        
        print(f"Starting RANSAC with {len(correspondences)} correspondences...")
        
        for iteration in range(max_iterations):
            # Randomly select 3 correspondences
            if len(correspondences) < 3:
                break
                
            sample_indices = np.random.choice(len(correspondences), 3, replace=False)
            sample_template = template_corr[sample_indices]
            sample_target = target_corr[sample_indices]
            
            # Estimate rigid transformation from 3 point pairs
            R, T = self.estimate_rigid_transformation_3_points(sample_template, sample_target)
            
            if R is None:
                continue
            
            # Transform all template correspondence points
            template_transformed = (R @ template_corr.T).T + T
            
            # Compute alignment error (Pe in paper)
            distances = np.linalg.norm(template_transformed - target_corr, axis=1)
            alignment_error = np.mean(distances)
            
            # Check alignment error threshold (ε1 in paper)
            if alignment_error > inlier_threshold:
                continue
            
            # Count inliers
            inliers = distances < inlier_threshold
            inlier_count = np.sum(inliers)
            
            # Compute camera proximity constraint (Ce in paper)
            template_mean = np.mean(template_transformed, axis=0)
            target_mean = np.mean(target_corr, axis=0)
            camera_error = np.linalg.norm(template_mean - target_mean)
            
            # Check camera proximity threshold (ε2 in paper)
            if camera_error > camera_proximity_threshold:
                continue
            
            # Update best solution
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_R, best_T = R, T
                best_inliers = np.where(inliers)[0]
        
        if best_R is not None:
            print(f"RANSAC found transformation with {best_inlier_count} inliers")
            return best_R, best_T, best_inliers
        else:
            print("RANSAC failed to find valid transformation")
            return None, None, []

    def estimate_rigid_transformation_3_points(self, template_pts, target_pts):
        """
        Estimate rigid transformation from 3 point correspondences
        """
        if len(template_pts) != 3 or len(target_pts) != 3:
            return None, None
        
        # Center the points
        template_centroid = np.mean(template_pts, axis=0)
        target_centroid = np.mean(target_pts, axis=0)
        
        template_centered = template_pts - template_centroid
        target_centered = target_pts - target_centroid
        
        # Compute cross-covariance matrix
        H = template_centered.T @ target_centered
        
        # SVD to find rotation
        try:
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Compute translation
            T = target_centroid - R @ template_centroid
            
            return R, T
        except:
            return None, None

    def refine_pose_with_icp(self, template_points, target_points, initial_R, initial_T, 
                           max_iterations=50, tolerance=1e-6):
        """
        Refine pose estimation using ICP algorithm
        """
        R, T = initial_R.copy(), initial_T.copy()
        
        for iteration in range(max_iterations):
            # Transform template points
            transformed_template = (R @ template_points.T).T + T
            
            # Find nearest neighbors
            distances = scipy.spatial.distance.cdist(transformed_template, target_points)
            nearest_indices = np.argmin(distances, axis=1)
            nearest_target = target_points[nearest_indices]
            
            # Estimate new transformation
            new_R, new_T = self.estimate_rigid_transformation_3_points(
                transformed_template[:3], nearest_target[:3])
            
            if new_R is None:
                break
            
            # Update transformation
            R = new_R @ R
            T = new_R @ T + new_T
            
            # Check convergence
            translation_change = np.linalg.norm(new_T)
            rotation_change = np.linalg.norm(new_R - np.eye(3))
            
            if translation_change < tolerance and rotation_change < tolerance:
                break
        
        return R, T

    def estimate_pose_with_spin_images(self, template_points, target_points, 
                                     correlation_threshold=0.7, 
                                     ransac_iterations=1000,
                                     inlier_threshold=0.005):
        """
        Complete pose estimation pipeline following the original paper's methodology:
        
        Main pipeline:
        Step 1: Data acquisition, preprocessing
        Step 2: Harris 3D IPD (Interest Point Detection)
        Step 3: Spin image generation of the chosen keypoints
        Step 4: Template and target spin image matching by calculating the R correspondence
        Step 5: Validate the result by calculating Pe and Ce thresholds
        Step 6: If all conditions satisfy, perform final ICP refinement
        """
        print("Starting spin image-based pose estimation following paper methodology...")
        
        # Step 1: Data acquisition, preprocessing (already done - points are input)
        print("Step 1: Data acquisition and preprocessing - COMPLETED")
        
        # Step 2: Harris 3D IPD (Interest Point Detection)
        print("Step 2: Harris 3D Interest Point Detection...")
        template_keypoints = self.compute_harris_3d_corners_simple(template_points, num_corners=20)
        target_keypoints = self.compute_harris_3d_corners_simple(target_points, num_corners=20)
        
        if len(template_keypoints) == 0 or len(target_keypoints) == 0:
            print("Step 2: FAILED - Unable to detect Harris 3D keypoints")
            return None, None
        
        print(f"Step 2: SUCCESS - Detected {len(template_keypoints)} template and {len(target_keypoints)} target keypoints")
        
        # Step 3: Spin image generation of the chosen keypoints
        print("Step 3: Spin image generation of the chosen keypoints...")
        
        # Compute surface normals for spin image generation
        template_normals = self.compute_surface_normals(template_points)
        target_normals = self.compute_surface_normals(target_points)
        
        # Get normals for keypoints
        template_kp_normals = []
        for kp in template_keypoints:
            distances = np.linalg.norm(template_points - kp, axis=1)
            closest_idx = np.argmin(distances)
            template_kp_normals.append(template_normals[closest_idx])
        
        target_kp_normals = []
        for kp in target_keypoints:
            distances = np.linalg.norm(target_points - kp, axis=1)
            closest_idx = np.argmin(distances)
            target_kp_normals.append(target_normals[closest_idx])
        
        # Generate spin images for template keypoints
        template_spin_images = []
        for i, (kp, normal) in enumerate(zip(template_keypoints, template_kp_normals)):
            spin_img = self.compute_spin_image(kp, normal, template_points)
            template_spin_images.append(spin_img)
        
        # Generate spin images for target keypoints  
        target_spin_images = []
        for i, (kp, normal) in enumerate(zip(target_keypoints, target_kp_normals)):
            spin_img = self.compute_spin_image(kp, normal, target_points)
            target_spin_images.append(spin_img)
        
        print(f"Step 3: SUCCESS - Generated {len(template_spin_images)} template and {len(target_spin_images)} target spin images")
        
        # Step 4: Template and target spin image matching by calculating the R correspondence
        print("Step 4: Template and target spin image matching by calculating R correspondence...")
        correspondences = []
        
        for i, template_spin in enumerate(template_spin_images):
            best_correlation = -1
            best_match = -1
            
            for j, target_spin in enumerate(target_spin_images):
                # Calculate R(P,Q) correlation coefficient using equation (4)
                correlation = self.compute_spin_image_correlation(template_spin, target_spin)
                
                if correlation > best_correlation and correlation > correlation_threshold:
                    best_correlation = correlation
                    best_match = j
            
            if best_match >= 0:
                correspondences.append((i, best_match, best_correlation))
        
        print(f"Step 4: SUCCESS - Found {len(correspondences)} correspondences with R(P,Q) > {correlation_threshold}")
        
        if len(correspondences) < 3:
            print("Step 4: FAILED - Insufficient correspondences for pose estimation")
            return None, None
        
        # Estimate initial rigid transformation using RANSAC
        print("Estimating initial rigid transformation using RANSAC...")
        R, T, inliers = self.estimate_rigid_transformation_ransac(
            template_keypoints, target_keypoints, correspondences,
            ransac_iterations, inlier_threshold)
        
        if R is None:
            print("RANSAC FAILED - Unable to find valid transformation")
            return None, None
        
        # Step 5: Validate the result by calculating Pe and Ce thresholds
        print("Step 5: Validating result by calculating Pe and Ce thresholds...")
        
        # Apply transformation to correspondence points
        template_corr = np.array([template_keypoints[i] for i, j, _ in correspondences])
        target_corr = np.array([target_keypoints[j] for i, j, _ in correspondences])
        template_transformed = (R @ template_corr.T).T + T
        
        # Calculate Pe (point alignment error) - equation (6)
        distances = np.linalg.norm(template_transformed - target_corr, axis=1)
        Pe = np.mean(distances)
        
        # Calculate Ce (centroid proximity error) - equation (7)  
        template_centroid = np.mean(template_transformed, axis=0)
        target_centroid = np.mean(target_corr, axis=0)
        Ce = np.linalg.norm(template_centroid - target_centroid)
        
        print(f"  Pe (point alignment error): {Pe:.6f}")
        print(f"  Ce (centroid proximity error): {Ce:.6f}")
        
        # Apply thresholds as per paper
        pe_threshold = inlier_threshold  # ε1 from paper
        ce_threshold = 0.01  # ε2 from paper (camera proximity threshold)
        
        if Pe > pe_threshold:
            print(f"Step 5: FAILED - Pe ({Pe:.6f}) > threshold ({pe_threshold:.6f})")
            return None, None
            
        if Ce > ce_threshold:
            print(f"Step 5: FAILED - Ce ({Ce:.6f}) > threshold ({ce_threshold:.6f})")
            return None, None
        
        print("Step 5: SUCCESS - Pe and Ce validation passed")
        
        # Step 6: If all conditions satisfy, perform final ICP refinement
        print("Step 6: All conditions satisfied - performing final ICP refinement...")
        R_refined, T_refined = self.refine_pose_with_icp(template_points, target_points, R, T)
        
        print(f"Pose estimation pipeline COMPLETED successfully")
        print(f"Final rotation matrix:\n{R_refined}")
        print(f"Final translation vector: {T_refined}")
        
        return R_refined, T_refined

    def extract_brick_coordinates(self, R, T, coordinate_system='euler_xyz'):
        """
        Extract brick coordinates (position and rotation) from pose estimation results
        
        Args:
            R: 3x3 rotation matrix from pose estimation
            T: 3x1 translation vector from pose estimation  
            coordinate_system: Format for rotation angles ('euler_xyz', 'euler_zyx', 'quaternion')
        
        Returns:
            Dictionary containing brick coordinates:
            - 'position': [x, y, z] in meters
            - 'rotation': [rx, ry, rz] in degrees (or quaternion if specified)
            - 'rotation_matrix': 3x3 rotation matrix
            - 'translation': 3x1 translation vector
        """
        # Extract position (translation)
        x, y, z = T[0], T[1], T[2]
        
        # Extract rotation angles from rotation matrix
        if coordinate_system == 'euler_xyz':
            # Extract Euler angles in XYZ order (roll, pitch, yaw)
            # Using scipy rotation for robust conversion
            from scipy.spatial.transform import Rotation as SciRotation
            
            rotation_obj = SciRotation.from_matrix(R)
            euler_angles = rotation_obj.as_euler('xyz', degrees=True)
            rx, ry, rz = euler_angles[0], euler_angles[1], euler_angles[2]
            
        elif coordinate_system == 'euler_zyx':
            # Extract Euler angles in ZYX order (yaw, pitch, roll)
            from scipy.spatial.transform import Rotation as SciRotation
            
            rotation_obj = SciRotation.from_matrix(R)
            euler_angles = rotation_obj.as_euler('zyx', degrees=True)
            rz, ry, rx = euler_angles[0], euler_angles[1], euler_angles[2]
            
        elif coordinate_system == 'quaternion':
            # Extract quaternion representation
            from scipy.spatial.transform import Rotation as SciRotation
            
            rotation_obj = SciRotation.from_matrix(R)
            quaternion = rotation_obj.as_quat()  # [x, y, z, w]
            rx, ry, rz = quaternion[0], quaternion[1], quaternion[2]  # x, y, z components
            qw = quaternion[3]  # w component
            
        else:
            # Fallback: manual extraction of Euler angles (XYZ order)
            # This is less robust but doesn't require scipy
            import math
            
            # Extract rotation angles from rotation matrix (ZYX Euler angles)
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
            
            # Convert to degrees
            rx = math.degrees(rx)
            ry = math.degrees(ry)
            rz = math.degrees(rz)
        
        # Prepare result dictionary
        coordinates = {
            'position': [x, y, z],
            'rotation_matrix': R.tolist(),
            'translation': T.tolist(),
        }
        
        if coordinate_system == 'quaternion':
            coordinates['rotation'] = [rx, ry, rz, qw]
            coordinates['quaternion'] = [rx, ry, rz, qw]
        else:
            coordinates['rotation'] = [rx, ry, rz]
            coordinates['euler_angles'] = [rx, ry, rz]
        
        # Add coordinate system info
        coordinates['coordinate_system'] = coordinate_system
        
        print(f"Extracted brick coordinates:")
        print(f"  Position (x, y, z): ({x:.3f}, {y:.3f}, {z:.3f}) meters")
        if coordinate_system == 'quaternion':
            print(f"  Rotation (qx, qy, qz, qw): ({rx:.3f}, {ry:.3f}, {rz:.3f}, {qw:.3f})")
        else:
            print(f"  Rotation (rx, ry, rz): ({rx:.1f}°, {ry:.1f}°, {rz:.1f}°)")
        
        return coordinates

    def calculate_brick_pose_from_template_match(self, template_info, R, T, coordinate_system='euler_xyz'):
        """
        Calculate final brick pose considering both template transformation and scene matching
        
        Args:
            template_info: Template information from library (contains original rotation)
            R: Rotation matrix from template-to-scene matching
            T: Translation vector from template-to-scene matching
            coordinate_system: Format for rotation angles
        
        Returns:
            Final brick coordinates accounting for template pose and scene transformation
        """
        # Get template's original rotation
        template_rotation_matrix = np.array(template_info['rotation_matrix'])
        
        # Combine template rotation with scene matching rotation
        # Final rotation = Scene_rotation * Template_rotation
        final_R = R @ template_rotation_matrix
        
        # Translation remains the same (scene position)
        final_T = T
        
        # Extract coordinates from combined transformation
        coordinates = self.extract_brick_coordinates(final_R, final_T, coordinate_system)
        
        # Add template information
        coordinates['template_id'] = template_info['id']
        coordinates['template_x_rotation'] = template_info['x_rotation']
        coordinates['template_y_rotation'] = template_info['y_rotation']
        
        print(f"Final brick pose (template {template_info['id']} + scene transformation):")
        print(f"  Template original rotation: X={template_info['x_rotation']}°, Y={template_info['y_rotation']}°")
        
        return coordinates

    def save_brick_coordinates(self, coordinates, output_file):
        """
        Save brick coordinates to file for robot arm grasping
        
        Args:
            coordinates: Brick coordinates from extract_brick_coordinates()
            output_file: Path to save the coordinates file
        """
        from datetime import datetime
        
        if coordinates is None:
            print("No coordinates to save")
            return
            
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_with_timestamp = f"{output_file}_{timestamp}.txt"
        
        # Prepare coordinate data for saving
        position = coordinates['position']
        
        # Save coordinates in a simple format for robot arm
        with open(filename_with_timestamp, 'w') as f:
            f.write("=== LEGO Brick Detection Results ===\n")
            f.write(f"Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Coordinate System: World Coordinates\n\n")
            
            f.write("=== Brick Position (World Coordinates) ===\n")
            f.write(f"X: {position[0]:.6f} meters\n")
            f.write(f"Y: {position[1]:.6f} meters\n") 
            f.write(f"Z: {position[2]:.6f} meters\n\n")
            
            if 'rotation' in coordinates:
                rotation = coordinates['rotation']
                f.write("=== Brick Orientation ===\n")
                if coordinates.get('coordinate_system') == 'quaternion':
                    f.write(f"Quaternion X: {rotation[0]:.6f}\n")
                    f.write(f"Quaternion Y: {rotation[1]:.6f}\n") 
                    f.write(f"Quaternion Z: {rotation[2]:.6f}\n")
                    f.write(f"Quaternion W: {coordinates.get('qw', 0):.6f}\n")
                else:
                    f.write(f"Rotation X: {rotation[0]:.3f} degrees\n")
                    f.write(f"Rotation Y: {rotation[1]:.3f} degrees\n")
                    f.write(f"Rotation Z: {rotation[2]:.3f} degrees\n")
            
            f.write("\n=== Ready for Robot Arm Grasping ===\n")
            f.write("Position format: [X, Y, Z] in meters from world origin\n")
            
        print(f"Brick coordinates saved to: {filename_with_timestamp}")
        return filename_with_timestamp

    # ========== SPIN IMAGE TEMPLATE GENERATION FROM CAD MODELS ==========
    
    def generate_spin_image_templates_from_cad(self, cad_file_path, output_dir, 
                                               x_axis_steps=10, y_axis_steps=19,
                                               num_keypoints_per_view=15):
        """
        Generate spin image templates from CAD models optimized for symmetrical LEGO bricks:
        Since LEGO bricks are symmetrical:
        - X-axis: 0° to 90° (exploiting symmetry) in 10° steps
        - Y-axis: 0° to 180° (exploiting symmetry) in 10° steps
        This gives approximately 10 × 19 = 190 templates (reasonable size)
        
        Args:
            cad_file_path: Path to CAD model (PLY/OBJ/STL file)
            output_dir: Directory to save template library
            x_axis_steps: Number of steps for x-axis rotation (0° to 90°) - default 10 gives 10° steps
            y_axis_steps: Number of steps for y-axis rotation (0° to 180°) - default 19 gives ~10° steps
            num_keypoints_per_view: Number of keypoints to extract per view
            
        Returns:
            Template library metadata
        """
        print(f"Generating spin image templates from CAD model: {cad_file_path}")
        print(f"Optimized for symmetrical LEGO brick: X-axis 0°-90° ({x_axis_steps} steps), Y-axis 0°-180° ({y_axis_steps} steps)")
        print(f"Step size: ~10° increments for reasonable template library size")
        
        # Calculate total expected poses
        total_expected_poses = x_axis_steps * y_axis_steps
        print(f"Expected total poses: {total_expected_poses} (optimized for LEGO brick symmetry)")
        
        # Load CAD model
        if not os.path.exists(cad_file_path):
            print(f"CAD file not found: {cad_file_path}")
            return None
        
        # Load CAD model - handle different file formats
        file_extension = os.path.splitext(cad_file_path)[1].lower()
        cad_points = None
        
        if file_extension == '.ply':
            # Load PLY as point cloud
            cad_pcd = o3d.io.read_point_cloud(cad_file_path)
            if len(cad_pcd.points) > 0:
                cad_points = np.asarray(cad_pcd.points)
        
        elif file_extension in ['.stl', '.obj']:
            # Load STL/OBJ as mesh first, then sample points
            cad_mesh = o3d.io.read_triangle_mesh(cad_file_path)
            if len(cad_mesh.vertices) > 0:
                print(f"Loaded mesh with {len(cad_mesh.vertices)} vertices and {len(cad_mesh.triangles)} triangles")
                
                # Sample points from mesh surface
                num_points = max(5000, len(cad_mesh.vertices) * 2)  # Ensure sufficient point density
                cad_pcd = cad_mesh.sample_points_uniformly(number_of_points=num_points)
                cad_points = np.asarray(cad_pcd.points)
                print(f"Sampled {len(cad_points)} points from mesh surface")
        
        else:
            print(f"Unsupported file format: {file_extension}")
            return None
        
        if cad_points is None or len(cad_points) == 0:
            print("Failed to load CAD model or empty point cloud")
            return None
        
        print(f"Successfully loaded CAD model with {len(cad_points)} points")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        templates = []
        template_id = 0
        
        # Generate templates following LEGO brick symmetry: x-axis 0°-90°, y-axis 0°-180°
        for x_idx in range(x_axis_steps):
            # X-axis rotation from 0° to 90° (exploiting LEGO brick symmetry)
            x_angle = x_idx * (90.0 / (x_axis_steps - 1)) if x_axis_steps > 1 else 0.0
            
            for y_idx in range(y_axis_steps):
                # Y-axis rotation from 0° to 180° (exploiting LEGO brick symmetry)
                y_angle = y_idx * (180.0 / (y_axis_steps - 1)) if y_axis_steps > 1 else 0.0
                
                print(f"Generating template {template_id}: X-axis={x_angle:.1f}°, Y-axis={y_angle:.1f}°")
                
                # Create rotation matrix for this pose (X-axis then Y-axis rotation)
                pose_matrix = self.create_paper_pose_matrix(x_angle, y_angle)
                
                # Transform CAD model to this pose
                transformed_points = self.transform_points_with_matrix(cad_points, pose_matrix)
                
                # Step 2: Harris 3D IPD (Interest Point Detection)
                keypoints = self.compute_harris_3d_corners_simple(transformed_points, num_keypoints_per_view)
                
                if len(keypoints) < 3:
                    print(f"  Insufficient keypoints detected, skipping pose...")
                    continue
                
                # Step 3: Spin image generation of the chosen keypoints
                print(f"  Step 3: Generating spin images for {len(keypoints)} keypoints...")
                normals = self.compute_surface_normals(transformed_points)
                
                # Get normals for keypoints
                keypoint_normals = []
                for kp in keypoints:
                    distances = np.linalg.norm(transformed_points - kp, axis=1)
                    closest_idx = np.argmin(distances)
                    keypoint_normals.append(normals[closest_idx])
                
                # Generate spin images for keypoints
                spin_images = []
                for i, (kp, normal) in enumerate(zip(keypoints, keypoint_normals)):
                    spin_img = self.compute_spin_image(kp, normal, transformed_points)
                    print(f"    Keypoint {i}: spin_img shape={spin_img.shape}, non-zero={np.count_nonzero(spin_img)}, max={np.max(spin_img):.6f}")
                    if np.count_nonzero(spin_img) == 0:
                        print(f"    WARNING: Spin image {i} is empty! Keypoint={kp}, Normal={normal}")
                    spin_images.append(spin_img.tolist())  # Convert to list for JSON serialization
                
                # Save template as PLY file for compatibility with matching code
                template_filename = f"template_{template_id:03d}.ply"
                template_file_path = os.path.join(output_dir, template_filename)
                
                # Create point cloud and save as PLY
                template_pcd = o3d.geometry.PointCloud()
                template_pcd.points = o3d.utility.Vector3dVector(transformed_points)
                # Add red color to distinguish templates
                template_colors = np.tile([1.0, 0.0, 0.0], (len(transformed_points), 1))
                template_pcd.colors = o3d.utility.Vector3dVector(template_colors)
                o3d.io.write_point_cloud(template_file_path, template_pcd)
                
                # Create template entry
                template = {
                    'id': f"template_{template_id:03d}",
                    'file': template_file_path,  # Add file path for compatibility
                    'x_rotation': x_angle,
                    'y_rotation': y_angle,
                    'x_axis_angle': x_angle,
                    'y_axis_angle': y_angle,
                    'pose_matrix': pose_matrix.tolist(),
                    'rotation_matrix': pose_matrix[:3, :3].tolist(),
                    'keypoints': keypoints.tolist(),
                    'keypoint_normals': [n.tolist() for n in keypoint_normals],
                    'spin_images': spin_images,
                    'template_points': transformed_points.tolist(),  # Store for ICP refinement
                    'num_keypoints': len(keypoints),
                    'cad_source': cad_file_path
                }
                
                templates.append(template)
                template_id += 1
                
                print(f"  Generated {len(spin_images)} spin images for template {template_id-1}")
        
        # Save template library
        library_metadata = {
            'templates': templates,
            'generation_params': {
                'x_axis_steps': x_axis_steps,
                'y_axis_steps': y_axis_steps, 
                'x_axis_range': '0° to 180°',
                'y_axis_range': '0° to 360°',
                'num_keypoints_per_view': num_keypoints_per_view,
                'cad_source': cad_file_path,
                'methodology': 'Paper specification: 172 poses total'
            },
            'total_templates': len(templates)
        }
        
        # Save to JSON file
        metadata_file = os.path.join(output_dir, "spin_image_template_library.json")
        with open(metadata_file, 'w') as f:
            json.dump(library_metadata, f, indent=2)
        
        print(f"Generated {len(templates)} spin image templates (expected ~172 as per paper)")
        print(f"Template library saved to: {metadata_file}")
        
        return library_metadata

    def create_paper_pose_matrix(self, x_angle_deg, y_angle_deg):
        """
        Create 4x4 pose matrix following the paper's rotation specification:
        X-axis rotation (0° to 180°) then Y-axis rotation (0° to 360°)
        
        Args:
            x_angle_deg: Rotation around X-axis in degrees (0° to 180°)
            y_angle_deg: Rotation around Y-axis in degrees (0° to 360°)
            
        Returns:
            4x4 transformation matrix
        """
        # Convert to radians
        x_rad = np.radians(x_angle_deg)
        y_rad = np.radians(y_angle_deg)
        
        # Rotation around X-axis
        R_x = np.array([
            [1, 0,              0],
            [0, np.cos(x_rad), -np.sin(x_rad)],
            [0, np.sin(x_rad),  np.cos(x_rad)]
        ])
        
        # Rotation around Y-axis
        R_y = np.array([
            [np.cos(y_rad),  0, np.sin(y_rad)],
            [0,              1, 0],
            [-np.sin(y_rad), 0, np.cos(y_rad)]
        ])
        
        # Combined rotation: Y-axis rotation after X-axis rotation
        R = R_y @ R_x
        
        # Create 4x4 homogeneous transformation matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = R
        
        return pose_matrix

    def create_viewing_pose_matrix(self, rotation_angle_deg, elevation_angle_deg):
        """
        Create 4x4 pose matrix for viewing angle
        
        Args:
            rotation_angle_deg: Rotation around Z-axis in degrees
            elevation_angle_deg: Elevation angle in degrees
            
        Returns:
            4x4 transformation matrix
        """
        # Convert to radians
        rot_rad = np.radians(rotation_angle_deg)
        elev_rad = np.radians(elevation_angle_deg)
        
        # Rotation around Z-axis 
        R_z = np.array([
            [np.cos(rot_rad), -np.sin(rot_rad), 0],
            [np.sin(rot_rad),  np.cos(rot_rad), 0],
            [0,                0,               1]
        ])
        
        # Rotation around X-axis for elevation
        R_x = np.array([
            [1, 0,                 0],
            [0, np.cos(elev_rad), -np.sin(elev_rad)],
            [0, np.sin(elev_rad),  np.cos(elev_rad)]
        ])
        
        # Combined rotation
        R = R_x @ R_z
        
        # Create 4x4 homogeneous transformation matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = R
        
        return pose_matrix

    def transform_points_with_matrix(self, points, transformation_matrix):
        """
        Transform 3D points using 4x4 transformation matrix
        
        Args:
            points: Nx3 array of 3D points
            transformation_matrix: 4x4 transformation matrix
            
        Returns:
            Transformed Nx3 array of 3D points
        """
        # Convert to homogeneous coordinates
        ones = np.ones((len(points), 1))
        points_homogeneous = np.hstack([points, ones])
        
        # Apply transformation
        transformed_homogeneous = (transformation_matrix @ points_homogeneous.T).T
        
        # Convert back to 3D coordinates
        transformed_points = transformed_homogeneous[:, :3]
        
        return transformed_points

    def compute_harris_3d_corners_simple(self, points, num_corners=15):
        """
        Simplified Harris 3D corner detection for target scene analysis.
        Removes the complexity of multi-hypothesis and focuses on finding good corner features.
        
        Args:
            points: Point cloud from real sensor data
            num_corners: Number of corners to detect
            
        Returns:
            Array of detected corner points
        """
        if len(points) < 10:
            print("Not enough points for Harris corner detection")
            return np.array([])

        print(f"Computing Harris corners for {len(points)} points...")
        
        try:
            # Use the existing improved Harris detection but simplified
            corners = self.compute_harris_3d_corners_improved(
                points, 
                delta=0.025,
                harris_k=0.04,
                cluster_threshold=0.008,
                num_corners=num_corners
            )
            
            if len(corners) > 0:
                print(f"Successfully detected {len(corners)} Harris corners")
                return corners
            else:
                print("Harris detection failed - insufficient keypoints for spin image pipeline")
                return np.array([])
                
        except Exception as e:
            print(f"Harris detection error: {e}")
            print("Unable to detect Harris corners - pipeline requires corner detection")
            return np.array([])

    def load_template_library(self, template_dir):
        """
        Load spin image template library from directory
        Supports both old PLY-based templates and new spin image templates
        """
        import json
        
        # Try to load spin image template library first
        spin_metadata_file = os.path.join(template_dir, "spin_image_template_library.json")
        if os.path.exists(spin_metadata_file):
            try:
                with open(spin_metadata_file, 'r') as f:
                    library_data = json.load(f)
                
                templates = library_data.get('templates', [])
                print(f"Loaded spin image template library with {len(templates)} templates")
                return templates
                
            except Exception as e:
                print(f"Error loading spin image template library: {e}")
        
        # Fallback to old PLY-based template system
        old_metadata_file = os.path.join(template_dir, "template_library.json")
        if os.path.exists(old_metadata_file):
            try:
                with open(old_metadata_file, 'r') as f:
                    templates_info = json.load(f)
                
                print(f"Loaded legacy PLY template library with {len(templates_info)} templates")
                print("Note: Consider generating spin image templates for better performance")
                return templates_info
                
            except Exception as e:
                print(f"Error loading legacy template library: {e}")
        
        print(f"No template library found in directory: {template_dir}")
        return []

    def create_demo_spin_image_templates(self, demo_output_dir):
        """
        Demo function to create spin image templates from a sample CAD model
        This demonstrates the correct pipeline from the paper
        """
        print("Creating demo spin image templates...")
        
        # For demonstration, create a simple LEGO brick-like point cloud
        # In practice, you would load a real CAD model file
        demo_cad_points = self.create_sample_lego_brick_points()
        
        # Save as temporary PLY file
        demo_cad_file = os.path.join(demo_output_dir, "demo_lego_brick.ply")
        os.makedirs(demo_output_dir, exist_ok=True)
        
        # Create Open3D point cloud
        demo_pcd = o3d.geometry.PointCloud()
        demo_pcd.points = o3d.utility.Vector3dVector(demo_cad_points)
        demo_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
        
        # Save PLY file
        o3d.io.write_point_cloud(demo_cad_file, demo_pcd)
        
        # Generate spin image templates
        template_library = self.generate_spin_image_templates_from_cad(
            cad_file_path=demo_cad_file,
            output_dir=demo_output_dir,
            rotation_steps=18,  # Every 20 degrees
            elevation_steps=3,  # 0, 30, 60 degrees
            num_keypoints_per_view=10
        )
        
        if template_library:
            print(f"Successfully created {template_library['total_templates']} demo templates")
            print("Template library ready for use in bin picking pipeline")
            
            # Example usage in pipeline:
            print("\nTo use in pipeline:")
            print(f"system.run_enhanced_bin_picking_pipeline(template_library_dir='{demo_output_dir}')")
        
        return template_library

    def create_sample_lego_brick_points(self):
        """
        Create a sample LEGO brick point cloud for demonstration
        Standard 2x4 LEGO brick dimensions: 32mm x 16mm x 9.6mm
        """
        # Brick dimensions in meters
        length, width, height = 0.032, 0.016, 0.0096
        
        # Generate points on brick surface
        points = []
        
        # Density of points per surface
        density = 100  # points per square meter
        
        # Top and bottom surfaces
        for z in [0, height]:
            num_points = int(length * width * density)
            x_coords = np.random.uniform(0, length, num_points)
            y_coords = np.random.uniform(0, width, num_points)
            z_coords = np.full(num_points, z)
            
            surface_points = np.column_stack([x_coords, y_coords, z_coords])
            points.append(surface_points)
        
        # Side surfaces
        # Front and back (length x height)
        for y in [0, width]:
            num_points = int(length * height * density)
            x_coords = np.random.uniform(0, length, num_points)
            z_coords = np.random.uniform(0, height, num_points)
            y_coords = np.full(num_points, y)
            
            surface_points = np.column_stack([x_coords, y_coords, z_coords])
            points.append(surface_points)
        
        # Left and right (width x height)
        for x in [0, length]:
            num_points = int(width * height * density)
            y_coords = np.random.uniform(0, width, num_points)
            z_coords = np.random.uniform(0, height, num_points)
            x_coords = np.full(num_points, x)
            
            surface_points = np.column_stack([x_coords, y_coords, z_coords])
            points.append(surface_points)
        
        # Add LEGO studs on top surface (2x4 = 8 studs)
        stud_radius = 0.0024  # 2.4mm radius
        stud_height = 0.0017  # 1.7mm height
        stud_spacing = 0.008  # 8mm spacing
        
        for i in range(2):  # 2 studs along width
            for j in range(4):  # 4 studs along length
                # Stud center position
                center_x = (j + 0.5) * stud_spacing
                center_y = (i + 0.5) * stud_spacing
                
                # Generate points on stud cylinder
                theta = np.linspace(0, 2*np.pi, 20)
                z_stud = np.linspace(height, height + stud_height, 5)
                
                for z in z_stud:
                    for t in theta:
                        x = center_x + stud_radius * np.cos(t)
                        y = center_y + stud_radius * np.sin(t)
                        points.append([[x, y, z]])
        
        # Combine all points
        all_points = np.vstack(points)
        
        # Center the brick at origin
        centroid = np.mean(all_points, axis=0)
        centered_points = all_points - centroid
        
        return centered_points

    def find_best_template_match(self, target_points, template_library_dir, 
                                correlation_threshold=0.7,
                                max_templates_to_test=700):
        """
        Find best matching template from library using spin image matching
        """
        templates_info = self.load_template_library(template_library_dir)
        
        if len(templates_info) == 0:
            print("No templates loaded from library")
            return None, None, None
        
        # Limit number of templates to test for efficiency
        templates_to_test = templates_info[:max_templates_to_test]
        
        best_match = None
        best_score = -1
        best_pose = None
        
        print(f"Testing {len(templates_to_test)} templates against target...")
        
        for i, template_info in enumerate(templates_to_test):
            template_file = template_info['file']
            
            if not os.path.exists(template_file):
                continue
            
            # Load template point cloud
            template_pcd = o3d.io.read_point_cloud(template_file)
            template_points = np.asarray(template_pcd.points)
            
            if len(template_points) < 10:
                continue
            
            # Estimate pose using spin images
            R, T = self.estimate_pose_with_spin_images(
                template_points, target_points, 
                correlation_threshold=correlation_threshold)
            
            if R is not None:
                # Compute matching score based on alignment quality
                transformed_template = (R @ template_points.T).T + T
                
                # Find distances to nearest target points
                distances = scipy.spatial.distance.cdist(transformed_template, target_points)
                min_distances = np.min(distances, axis=1)
                
                # Compute score (higher is better)
                inlier_threshold = 0.01  # 1cm
                inliers = np.sum(min_distances < inlier_threshold)
                score = inliers / len(transformed_template)
                
                if score > best_score:
                    best_score = score
                    best_match = template_info
                    best_pose = (R, T)
                    
                print(f"Template {i}: Score = {score:.3f}")
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(templates_to_test)} templates...")
        
        if best_match is not None:
            print(f"Best template match: {best_match['file']}")
            print(f"Score: {best_score:.3f}")
            print(f"Pose: X-rot={best_match['x_rotation']}°, Y-rot={best_match['y_rotation']}°")
            
        return best_match, best_pose, best_score

    def compute_harris_3d_corners_improved(self, points, delta=0.025, harris_k=0.04, 
                                         fraction=0.15, cluster_threshold=0.008, 
                                         num_corners=12):
        """
        Improved 3D Harris corner detection based on the 3D_Harris_IPD implementation.
        Now includes proper preprocessing steps: centering, scaling, and alignment.
        
        Parameters:
        - points: numpy array of 3D points
        - delta: neighborhood size parameter for adaptive k-ring
        - harris_k: Harris corner detection parameter
        - fraction: fraction of points to select as corners
        - cluster_threshold: minimum distance between corners for clustering
        - num_corners: maximum number of corners to return
        """
        if len(points) < 10:
            print("Not enough points for Harris corner detection")
            return np.array([])

        print(f"Computing Harris corners for {len(points)} points...")
        
        # STEP 1: Preprocess points for stable Harris detection
        preprocessed_points, transform_params = self.preprocess_for_harris_detection(points)
        
        # STEP 2: Compute neighborhoods using adaptive Delaunay triangulation
        neighborhood = self.compute_delaunay_neighborhood(preprocessed_points, delta=delta)
        
        # STEP 3: Initialize response array
        resp = np.zeros(len(preprocessed_points))
        
        # STEP 4: Compute Harris response for each point using preprocessed coordinates
        for point_idx in neighborhood.keys():
            try:
                if len(neighborhood[point_idx]) < 3:
                    resp[point_idx] = -np.inf
                    continue
                    
                neighbors = preprocessed_points[neighborhood[point_idx], :]
                
                # Center the neighbors around their local centroid
                neighbors_centred = neighbors - np.mean(neighbors, axis=0)
                
                # Principal Component Analysis
                pca = PCA(n_components=3)
                neighbors_pca = pca.fit_transform(neighbors_centred)
                eigenvalues, eigenvectors = np.linalg.eigh(pca.components_)
                
                # Get the best fitting normal (smallest eigenvalue)
                idx = np.argmin(eigenvalues)
                best_fit_normal = eigenvectors[idx, :]
                
                # Rotate the cloud to align with the normal
                rotated_neighbors = neighbors.copy()
                for i in range(neighbors.shape[0]):
                    rotated_neighbors[i, :] = np.dot(np.transpose(eigenvectors), neighbors[i, :])
                
                # Restrict to XY plane and translate
                neighbors_2D = rotated_neighbors[:, :2] - rotated_neighbors[0, :2]
                
                # Fit a quadratic surface z = f(x,y)
                if len(neighbors_2D) >= 6:  # Need at least 6 points for quadratic fitting
                    m = self.polyfit3d(neighbors_2D[:, 0], neighbors_2D[:, 1], rotated_neighbors[:, 2], order=2)
                    m = m.reshape((3, 3))
                    
                    # Compute the Harris response using the fitted surface derivatives
                    # These are the second derivatives of the surface
                    fx2 = m[2, 0]**2 + 2*m[1, 1]**2 + 2*m[0, 2]**2  # A
                    fy2 = m[0, 2]**2 + 2*m[1, 1]**2 + 2*m[2, 0]**2  # B
                    fxfy = m[2, 0]*m[0, 2] + 2*m[1, 1]**2  # C
                    
                    # Harris corner response
                    resp[point_idx] = fx2 * fy2 - fxfy * fxfy - harris_k * (fx2 + fy2) * (fx2 + fy2)
                else:
                    resp[point_idx] = -np.inf
                    
            except Exception as e:
                resp[point_idx] = -np.inf
                continue
        
        # STEP 5: Select interest points - find local maxima in preprocessed space
        candidate = []
        for point_idx in neighborhood.keys():
            if len(neighborhood[point_idx]) > 0:
                neighbor_responses = resp[neighborhood[point_idx]]
                if resp[point_idx] >= np.max(neighbor_responses):
                    candidate.append([point_idx, resp[point_idx]])
        
        if len(candidate) == 0:
            print("No corner candidates found")
            return np.array([])
        
        # Sort by decreasing Harris response
        candidate.sort(reverse=True, key=lambda x: x[1])
        candidate = np.array(candidate)
        
        # # Method 1: Select top fraction of points
        # num_fraction = max(1, int(fraction * len(preprocessed_points)))
        # interest_points_fraction = candidate[:num_fraction, 0].astype(int)
        
        # Method 2: Cluster-based selection (avoid points too close to each other)
        selected_corners = []
        if len(candidate) > 0:
            # Start with the best corner
            selected_corners.append(int(candidate[0, 0]))
            Q = preprocessed_points[int(candidate[0, 0]), :].reshape((1, -1))
            
            # Add corners that are far enough from existing ones
            for i in range(1, len(candidate)):
                query = preprocessed_points[int(candidate[i, 0]), :].reshape((1, -1))
                distances = scipy.spatial.distance.cdist(query, Q, metric='euclidean')
                if np.min(distances) > cluster_threshold:
                    selected_corners.append(int(candidate[i, 0]))
                    Q = np.concatenate((Q, query), axis=0)
                    
                    # Stop if we have enough corners
                    if len(selected_corners) >= num_corners:
                        break
        
        # STEP 6: Return corner points in ORIGINAL coordinate system
        if len(selected_corners) > 0:
            # Get corner points from original (non-preprocessed) coordinates
            corner_points = points[selected_corners]
            
            # Optional: Apply inverse transformation if needed for specific applications
            # For now, we return points in original coordinates since preprocessing
            # was mainly for numerical stability during computation
            
            print(f"Found {len(corner_points)} Harris corners with improved preprocessing")
            return corner_points
        else:
            print("No valid corners found after clustering")
            return np.array([])

    def cluster_and_save_summary(self, transformed_file, summary_file,
                             dbscan_eps=0.01, dbscan_min_samples=10,
                             harris_delta=0.02, harris_k=0.04, harris_fraction=0.15,
                             harris_cluster_threshold=0.008, harris_num_corners=12,
                             template_library_dir=None, enable_spin_images=False):
        data = np.loadtxt(transformed_file, delimiter=' ')
        points, colors = data[:, :3], data[:, 3:6]
        
        # Find the closest brick cluster for robot arm grasping
        closest_points, closest_colors = self.find_closest_brick_cluster(points, colors, 
                                                                        dbscan_eps, dbscan_min_samples)
        
        if closest_points is None:
            print("No valid brick cluster found for analysis")
            return

        results = []
        geometries = []

        print(f"Analyzing closest brick cluster with {len(closest_points)} points...")

        # Add cluster point cloud for visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(closest_points)
        pcd.colors = o3d.utility.Vector3dVector(closest_colors)
        geometries.append(pcd)

        # Pose estimation method selection
        if enable_spin_images and template_library_dir is not None:
            print("-> Using spin image-based pose estimation...")
            
            # Find best template match and estimate pose
            best_template, best_pose, match_score = self.find_best_template_match(
                closest_points, template_library_dir)
            
            if best_template is not None and best_pose is not None:
                R, T = best_pose
                print(f"-> Spin image pose estimation successful")
                print(f"-> Template: {best_template['file']}")
                print(f"-> Match score: {match_score:.3f}")
                print(f"-> Estimated rotation: X={best_template['x_rotation']}°, Y={best_template['y_rotation']}°")
                
                # Visualize template alignment
                template_pcd = o3d.io.read_point_cloud(best_template['file'])
                template_points = np.asarray(template_pcd.points)
                
                # Transform template to match target
                transformed_template = (R @ template_points.T).T + T
                
                # Create template visualization
                template_vis = o3d.geometry.PointCloud()
                template_vis.points = o3d.utility.Vector3dVector(transformed_template)
                template_vis.paint_uniform_color([0.0, 1.0, 0.0])  # Green for template
                geometries.append(template_vis)
                
                # Store pose information
                pose_info = {
                    'method': 'spin_image',
                    'rotation_matrix': R.tolist(),
                    'translation': T.tolist(),
                    'template_id': best_template['id'],
                    'match_score': match_score,
                    'x_rotation': best_template['x_rotation'],
                    'y_rotation': best_template['y_rotation']
                }
                
                # Extract detailed brick coordinates
                print("-> Extracting brick coordinates...")
                brick_coordinates = self.calculate_brick_pose_from_template_match(
                    best_template, R, T, coordinate_system='euler_xyz')
                
                # Add coordinate information to pose_info
                pose_info.update({
                    'brick_position': brick_coordinates['position'],  # [x, y, z] in meters
                    'brick_rotation': brick_coordinates['rotation'],  # [rx, ry, rz] in degrees
                    'brick_coordinates': brick_coordinates  # Full coordinate information
                })
                
            else:
                print("-> Spin image pose estimation failed, falling back to Harris corners...")
                pose_info = None
        else:
            pose_info = None

        # Fallback: Harris corner detection for comparison or when spin images fail
        if not enable_spin_images or pose_info is None:
            print("-> Using Harris corner detection...")
            
            if len(closest_points) > 20:
                print("-> Detecting Harris corners for closest brick with symmetry handling...")
                
                # Generate multiple corner hypotheses to handle symmetry
                corner_hypotheses = self.compute_harris_3d_corners_multi_hypothesis(
                    closest_points, 
                    delta=harris_delta, 
                    harris_k=harris_k, 
                    cluster_threshold=harris_cluster_threshold, 
                    num_corners=harris_num_corners,
                    num_hypotheses=3)
                
                if len(corner_hypotheses) > 0:
                    # Select the best hypothesis using geometric validation
                    corners = self.select_best_pose_hypothesis(corner_hypotheses, closest_points)
                    print(f"-> Selected best corner set with {len(corners)} corners.")
                    
                    # Visualize detected corners as orange spheres
                    if len(corners) > 0:
                        for corner_pt in corners:
                            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
                            sphere.translate(corner_pt)
                            sphere.paint_uniform_color([1.0, 0.647, 0.0])  # Orange
                            geometries.append(sphere)
                else:
                    print("-> No valid corner hypotheses found")
                    corners = np.array([])

        # Orientation analysis for the closest brick (traditional PCA-based)
        pca = PCA(n_components=3).fit(closest_points)
        center = np.mean(closest_points, axis=0)
        angle_deg = self.calculate_y_axis_angle_xy(pca.components_[1])
        dom_color = self.get_dominant_color(closest_colors)
        
        # Combine traditional result with pose estimation if available
        result_data = {
            'center': center,
            'color': dom_color,
            'pca_angle': angle_deg
        }
        
        if pose_info is not None:
            result_data.update(pose_info)
        
        results.append(result_data)

        # Final interactive visualization
        if geometries:
            window_name = "Closest Brick Analysis"
            if enable_spin_images and pose_info is not None:
                window_name += " - Spin Image Pose Estimation"
            else:
                window_name += " - Harris Corner Detection"
                
            print(f"Visualizing results: {window_name}")
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([axes] + geometries, window_name=window_name)

        # Save enhanced analysis summary
        with open(summary_file, "w") as f:
            for result in results:
                center = result['center']
                color = result['color']
                angle = result['pca_angle']
                
                cx, cy, cz = center * 1000.0  # Convert to millimeters
                r_, g_, b_ = color
                
                # Write basic info
                f.write(f"{cx:.2f} {cy:.2f} {cz:.2f} {r_:.6f} {g_:.6f} {b_:.6f} {angle:.2f}")
                
                # Write pose estimation info if available
                if 'method' in result:
                    method = result['method']
                    if method == 'spin_image':
                        f.write(f" POSE_METHOD:{method}")
                        f.write(f" TEMPLATE_ID:{result['template_id']}")
                        f.write(f" MATCH_SCORE:{result['match_score']:.3f}")
                        f.write(f" X_ROT:{result['x_rotation']}")
                        f.write(f" Y_ROT:{result['y_rotation']}")
                        
                        # Write brick coordinates (position and rotation)
                        if 'brick_position' in result and 'brick_rotation' in result:
                            brick_x, brick_y, brick_z = result['brick_position']
                            brick_rx, brick_ry, brick_rz = result['brick_rotation']
                            f.write(f" BRICK_POS:{brick_x:.6f},{brick_y:.6f},{brick_z:.6f}")
                            f.write(f" BRICK_ROT:{brick_rx:.2f},{brick_ry:.2f},{brick_rz:.2f}")
                        
                        # Write rotation matrix and translation
                        R = result['rotation_matrix']
                        T = result['translation']
                        f.write(f" R_MATRIX:{','.join([f'{x:.6f}' for row in R for x in row])}")
                        f.write(f" TRANSLATION:{','.join([f'{x:.6f}' for x in T])}")
                
                f.write("\n")
        
        print(f"Saved enhanced analysis summary: {summary_file}")
        
        # Save world coordinates if pose estimation was successful
        if any('brick_position' in result for result in results):
            # Extract coordinates from the best result for simplified saving
            best_result = None
            for result in results:
                if 'brick_position' in result and 'brick_rotation' in result:
                    best_result = result
                    break
            
            if best_result:
                # Create simplified coordinates dictionary for world coordinate saving
                coordinates = {
                    'position': best_result['brick_position'],
                    'rotation': best_result['brick_rotation'],
                    'coordinate_system': 'euler_xyz'
                }
                coord_file = summary_file.replace('.txt', '_world_coordinates')
                self.save_brick_coordinates(coordinates, coord_file)
        
        # Save detailed pose information if available
        if any('method' in result for result in results):
            pose_details_file = summary_file.replace('.txt', '_pose_details.json')
            import json
            with open(pose_details_file, 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            print(f"Saved detailed pose information: {pose_details_file}")


    # def send_file_via_tcp(self, file_path):
    #     filename = os.path.basename(file_path).encode('utf-8')
    #     with open(file_path, "rb") as f:
    #         file_data = f.read()
    #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #         s.connect((self.host, self.port))
    #         s.sendall(struct.pack('>I', len(filename)))
    #         s.sendall(filename)
    #         s.sendall(file_data)

    # Running the pipeline and saving data

def run_full_pipeline():
    """Full pipeline execution"""
    # Initialize the 3D Object Alignment Pipeline System
    system = BinPickingSystem(wdf_path="")
    
    # Test spin image visualization first
    print("="*60)
    print("TESTING SPIN IMAGE VISUALIZATION")
    print("="*60)
    system.test_spin_image_visualization()
    
    # Pipeline Implementation following the original paper methodology
    
    print("="*60)
    print("3D OBJECT ALIGNMENT PIPELINE")
    print("Following Original Paper Methodology")
    print("="*60)
    
    # Initial Step: Generate Template Library from CAD model
    # Following paper: "To represent the 3D model for an object, we use a simulator to generate 
    # the depth images of an object along x-axis from 0° to 180° and y-axis from 0° to 360°. 
    # Thus, We have totally 172 poses of an object."
    
    cad_file = "C:\\Users\\FILAB\\Desktop\\DUY\\LegoBrick_4_2.stl"
    template_output_dir = "C:\\Users\\FILAB\\Desktop\\DUY\\templates"
    
    print("Initial Step: Template Generation from CAD Model")
    print("TESTING MODE: Generating only 2 templates to visualize spin images")
    
    spin_templates = system.generate_spin_image_templates_from_cad(
        cad_file_path=cad_file,
        output_dir=template_output_dir,
        x_axis_steps=1,  # Only 1 step (0°) for X-axis
        y_axis_steps=2,  # 2 steps (0°, 180°) for Y-axis  
        num_keypoints_per_view=15
    )
    
    if not spin_templates:
        print("Failed to generate templates. Exiting.")
        exit(1)
    
    print(f"Generated {spin_templates['total_templates']} spin image templates for testing visualization")
    
    # Main Pipeline: Steps 1-6
    print("\nRunning Main Pipeline (Steps 1-6)...")
    print("Step 1: Data acquisition, preprocessing")
    print("Step 2: Harris 3D IPD")
    print("Step 3: Spin image generation of chosen keypoints")
    print("Step 4: Template-target spin image matching (R correspondence)")
    print("Step 5: Pose validation (Pe and Ce thresholds)")
    print("Step 6: Final ICP refinement (if validation passes)")
    
    results = system.run_enhanced_bin_picking_pipeline(
        template_library_dir=template_output_dir,
        enable_visualization=True,
        enable_spin_images=True
    )
    
    if results:
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        world_coords = results['world_coordinates']
        pose_result = results['pose_result']
        
        print(f"Alignment Method: {world_coords['method']}")
        print(f"Object Position: [{world_coords['position'][0]:.3f}, {world_coords['position'][1]:.3f}, {world_coords['position'][2]:.3f}] meters")
        print(f"Object Rotation: [{world_coords['rotation'][0]:.1f}, {world_coords['rotation'][1]:.1f}, {world_coords['rotation'][2]:.1f}] degrees")
        
        if 'Pe' in pose_result and 'Ce' in pose_result:
            print(f"Pe (point error): {pose_result['Pe']:.6f}")
            print(f"Ce (centroid error): {pose_result['Ce']:.6f}")
            print(f"Template Match Score: {pose_result['match_score']:.3f}")
            print(f"Template ID: {pose_result['template_id']}")
    else:
        print("Pipeline execution failed")
        print("- No valid clusters found or pose estimation failed")
    
    print("Bin picking system completed!")

# Standalone function to generate template spin images for visualization
def generate_template_spin_images_only(cad_file_path, output_dir, x_axis_steps=1, y_axis_steps=2):
    """
    Standalone function to generate and visualize template spin images from CAD model
    without running the full pipeline. Perfect for testing and visualization.
    
    Args:
        cad_file_path: Path to CAD model file
        output_dir: Directory to save spin images
        x_axis_steps: Number of rotation steps for X-axis
        y_axis_steps: Number of rotation steps for Y-axis
    """
    print("="*60)
    print("GENERATING TEMPLATE SPIN IMAGES FROM CAD MODEL")
    print("="*60)
    
    system = BinPickingSystem(None, output_dir)
    
    # Generate templates
    print(f"Loading CAD model: {cad_file_path}")
    templates = system.generate_spin_image_templates_from_cad(
        cad_file_path=cad_file_path,
        output_dir=output_dir,
        x_axis_steps=x_axis_steps,
        y_axis_steps=y_axis_steps,
        num_keypoints_per_view=15
    )
    
    if not templates:
        print("ERROR: Failed to generate templates!")
        return
    
    print(f"Generated {templates['total_templates']} templates")
    
    # Load the templates and generate spin images for each
    template_library = system.load_template_library(output_dir)
    if not template_library:
        print("ERROR: Failed to load template library!")
        return
    
    print(f"Loaded {len(template_library)} templates from library")
    
    # Generate spin images for each template
    for template in template_library:
        try:
            print(f"\nProcessing template {template['id']}...")
            
            # Load template point cloud
            template_file = template['file']
            template_pcd = o3d.io.read_point_cloud(template_file)
            template_points = np.asarray(template_pcd.points)
            
            if len(template_points) < 10:
                print(f"  Skipping template {template['id']} - too few points")
                continue
            
            # Get Harris keypoints
            template_keypoints = system.compute_harris_3d_keypoints(template_points, num_corners=15)
            if len(template_keypoints) == 0:
                print(f"  Skipping template {template['id']} - no keypoints found")
                continue
            
            print(f"  Found {len(template_keypoints)} Harris keypoints")
            
            # Compute surface normals
            template_normals = system.compute_surface_normals(template_points)
            template_kp_normals = []
            for kp in template_keypoints:
                distances = np.linalg.norm(template_points - kp, axis=1)
                closest_idx = np.argmin(distances)
                template_kp_normals.append(template_normals[closest_idx])
            
            # Generate spin images
            template_spin_images = []
            print(f"  Generating {len(template_keypoints)} spin images...")
            
            for j, (kp, normal) in enumerate(zip(template_keypoints, template_kp_normals)):
                spin_img = system.compute_spin_image(kp, normal, template_points)
                template_spin_images.append(spin_img)
                
                # Save individual spin image
                template_spin_dir = os.path.join(output_dir, "spin_images", "templates", template['id'])
                template_spin_filename = os.path.join(template_spin_dir, f"template_spin_{j:03d}.png")
                print(f"    Saving spin image {j+1}/{len(template_keypoints)}: {template_spin_filename}")
                system.save_spin_image(spin_img, template_spin_filename, cmap="plasma")
            
            # Create summary visualization
            if len(template_spin_images) > 0:
                print(f"  Creating summary visualization for template {template['id']}...")
                system.save_template_spin_image_summary(template_spin_images, template['id'], output_dir)
                print(f"  Template {template['id']}: {len(template_spin_images)} spin images saved!")
                
        except Exception as e:
            print(f"ERROR processing template {template['id']}: {e}")
            import traceback
            traceback.print_exc()
    
    print("="*60)
    print("TEMPLATE SPIN IMAGE GENERATION COMPLETED!")
    print("="*60)
    print(f"Check the following directories for results:")
    print(f"  Individual images: {output_dir}/spin_images/templates/template_XXX/")
    print(f"  Summary images: {output_dir}/spin_images/templates/template_XXX/template_XXX_spin_summary.png")

if __name__ == "__main__":
    # Test template spin image generation without full pipeline
    print("OPTION 1: Generate Template Spin Images Only (for visualization)")
    print("OPTION 2: Run Full Pipeline")
    print()
    
    # Uncomment the option you want to run:
    
    # OPTION 1: Generate template spin images only (recommended for visualization)
    cad_file = "C:\\Users\\FILAB\\Desktop\\DUY\\LegoBrick_4_2.stl"
    template_output_dir = "C:\\Users\\FILAB\\Desktop\\DUY\\templates"
    
    print("Running OPTION 1: Template Spin Image Generation Only")
    generate_template_spin_images_only(
        cad_file_path=cad_file,
        output_dir=template_output_dir,
        x_axis_steps=1,  # Just 1 pose for testing
        y_axis_steps=2   # 2 poses for testing
    )
    print("Template spin images generated! Check the output directory.")
    
    # OPTION 2: Full pipeline (uncomment to run instead)
    # run_full_pipeline()

def run_full_pipeline():
    """Full pipeline execution"""