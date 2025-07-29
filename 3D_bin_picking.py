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

# Main Program
class BinPickingSystem:

    # Initializing (This part is for the robot arm so it is not necessary)
    def __init__(self, wdf_path, output_dir=None):
        self.output_dir = "C:\\Users\\FILAB\\Desktop\\DUY\\Results" # Data saving location
        # self.host = "192.168.1.23" # Target IP for sending data to the robot arm
        # self.port = 9999 # Target (Robot arm) port

    # ========== STEP 1: DATA ACQUISITION AND PREPROCESSING ==========
    
    # Border filtering algorithm using AND logic
    def keep_inside_boundary_points(self, points, colors, x_min, x_max, y_min, y_max, margin=0.02):
        mask = (
            (points[:, 0] >= x_min + margin) & (points[:, 0] <= x_max - margin) &
            (points[:, 1] >= y_min + margin) & (points[:, 1] <= y_max - margin)
        ) # Removing the border by an amount of margin
        # The scanning range will be (x_min + margin, x_max - margin) x (y_min + margin, y_max - margin)
        return points[mask], colors[mask] # Return filtered point cloud and color values

    # ========== STEPS 2-3: CLUSTERING AND TOPMOST BRICK EXTRACTION ==========
    
    def find_and_extract_topmost_brick(self, points, colors, dbscan_eps=0.01, dbscan_min_samples=10, 
                                      color_tolerance=0.15, top_surface_depth=0.02):
        """
        Combined Step 2-3: Find the closest cluster to camera and extract topmost brick by dominant color
        
        This function:
        1. Finds the cluster with the closest point to camera
        2. Extracts the color from the closest points in that cluster
        3. Filters the entire point cloud to keep only points with the dominant color
        
        Args:
            points: 3D point cloud
            colors: RGB colors for each point
            dbscan_eps: DBSCAN radius parameter
            dbscan_min_samples: DBSCAN minimum samples parameter
            color_tolerance: Tolerance for color similarity (0.0 = exact match, 1.0 = any color)
            top_surface_depth: Depth range to consider as "top surface" for color extraction (meters)
        
        Returns:
            Tuple of (topmost_brick_points, topmost_brick_colors, cluster_label)
        """
        print("Step 2-3: Finding closest cluster and extracting topmost brick by color...")
        
        # PHASE 1: Find the cluster with the closest point to camera
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points)
        labels = dbscan.labels_
        
        # Find all valid clusters (exclude noise with label -1)
        valid_clusters = []
        for lbl in set(labels):
            if lbl == -1:  # Skip noise points
                continue
            
            cluster_mask = (labels == lbl)
            cluster_points = points[cluster_mask]
            cluster_colors = colors[cluster_mask]
            
            # Filter small clusters that are likely noise
            if len(cluster_points) < 100:
                continue
            
            # Find the closest point in this cluster (minimum Z value)
            min_z = np.min(cluster_points[:, 2])
            cluster_center = np.mean(cluster_points, axis=0)
            
            valid_clusters.append({
                'label': lbl,
                'points': cluster_points,
                'colors': cluster_colors,
                'min_z': min_z,
                'center': cluster_center,
                'size': len(cluster_points)
            })
        
        if len(valid_clusters) == 0:
            print("No valid clusters found")
            return None, None, -1
        
        # Sort by closest point (smallest Z value) and select the closest cluster
        valid_clusters.sort(key=lambda x: x['min_z'])
        closest_cluster = valid_clusters[0]
        
        print(f"Found {len(valid_clusters)} valid clusters")
        print(f"Selected closest cluster {closest_cluster['label']} with closest point at Z={closest_cluster['min_z']:.3f}m")
        print(f"Cluster size: {closest_cluster['size']} points")
        
        # PHASE 2: Extract dominant color from the closest points in the selected cluster
        cluster_points = closest_cluster['points']
        cluster_colors = closest_cluster['colors']
        
        # Find the topmost points (closest to camera) in the selected cluster
        z_coords = cluster_points[:, 2]
        z_min = np.min(z_coords)
        z_threshold = z_min + top_surface_depth  # Consider top surface for color extraction
        
        top_surface_mask = z_coords <= z_threshold
        top_surface_colors = cluster_colors[top_surface_mask]
        
        if len(top_surface_colors) == 0:
            print("No top surface points found in closest cluster, using cluster center color")
            # Fallback: use colors near the cluster center
            center = closest_cluster['center']
            distances_to_center = np.linalg.norm(cluster_points - center, axis=1)
            closest_to_center_idx = np.argmin(distances_to_center)
            dominant_color = cluster_colors[closest_to_center_idx]
            dominant_color_name = "unknown"
        else:
            # Get dominant color from top surface of the closest cluster using HSV classification
            dominant_color, dominant_color_name = self.get_dominant_color(top_surface_colors)
        
        print(f"Extracted dominant color from closest cluster: {dominant_color_name}")
        print(f"RGB values: [{dominant_color[0]:.3f}, {dominant_color[1]:.3f}, {dominant_color[2]:.3f}]")
        
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
        
        return topmost_brick_points, topmost_brick_colors, closest_cluster['label']
    
    # ========== ENHANCED ALIGNMENT ALGORITHM ==========
    
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
    
    def detect_and_match_topmost_brick(self, cluster_points, cluster_colors, template_library_dir=None):
        """
        Step 4: Apply Harris 3D and spin image for pose matching the top brick
        
        Args:
            cluster_points: Points from the closest cluster after color filtering
            cluster_colors: Colors from the closest cluster after color filtering
            template_library_dir: Directory containing template library (optional)
        
        Returns:
            Dictionary containing pose estimation results
        """
        print("Step 4: Detecting and matching topmost brick using Harris 3D and spin images...")
        
        # Apply Harris 3D corner detection
        corner_hypotheses = self.compute_harris_3d_corners_multi_hypothesis(
            cluster_points, 
            delta=0.02,
            harris_k=0.04,
            num_corners=15,
            num_hypotheses=1
        )
        
        if len(corner_hypotheses) == 0:
            print("No Harris corners detected")
            return None
        
        target_keypoints = corner_hypotheses[0]
        print(f"Detected {len(target_keypoints)} Harris corner keypoints")
        
        # If template library is provided, use spin image matching
        if template_library_dir and os.path.exists(template_library_dir):
            return self.match_with_template_library(cluster_points, cluster_colors, 
                                                   target_keypoints, template_library_dir)
        else:
            print("No template library provided")
            return None
    
    def match_with_template_library(self, target_points, target_colors, target_keypoints, template_library_dir):
        """
        Match target brick with template library using spin images
        """
        print("Matching with template library using spin images...")
        
        # Load template library
        templates = self.load_template_library(template_library_dir)
        if not templates:
            print("No templates found in library")
            return None
        
        best_match = None
        best_score = -1
        
        # Compute surface normals for target
        target_normals = self.compute_surface_normals(target_points)
        
        # Get normals for keypoints
        target_kp_normals = []
        for kp in target_keypoints:
            distances = np.linalg.norm(target_points - kp, axis=1)
            closest_idx = np.argmin(distances)
            target_kp_normals.append(target_normals[closest_idx])
        
        # Match against each template
        for template in templates:
            try:
                template_points = np.array(template['points'])
                template_keypoints = np.array(template['keypoints']) if 'keypoints' in template else []
                
                if len(template_keypoints) == 0:
                    continue
                
                # Perform spin image matching
                R, T = self.estimate_pose_with_spin_images(
                    template_points, target_points,
                    correlation_threshold=0.6
                )
                
                if R is not None:
                    # Calculate match score
                    match_score = self.calculate_match_score(template_points, target_points, R, T)
                    
                    if match_score > best_score:
                        best_score = match_score
                        best_match = {
                            'template_id': template['id'],
                            'rotation_matrix': R,
                            'translation': T,
                            'match_score': match_score,
                            'method': 'spin_image_matching'
                        }
                        
            except Exception as e:
                print(f"Error matching template {template.get('id', 'unknown')}: {e}")
                continue
        
        if best_match:
            print(f"Best template match: {best_match['template_id']} (score: {best_match['match_score']:.3f})")
            return best_match
        else:
            print("No suitable template match found")
            return None
    
    def calculate_match_score(self, template_points, target_points, R, T, distance_threshold=0.01):
        """
        Calculate match score between template and target after transformation
        """
        # Transform template points
        transformed_template = (R @ template_points.T).T + T
        
        # Find nearest neighbors
        distances = scipy.spatial.distance.cdist(transformed_template, target_points)
        min_distances = np.min(distances, axis=1)
        
        # Calculate score based on percentage of points within threshold
        good_matches = np.sum(min_distances < distance_threshold)
        score = good_matches / len(transformed_template)
        
        return score
        
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
    
    # ========== STEP 6: VISUALIZATION AND OUTPUT ==========
    
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
        
        # Create template point cloud
        template_points = np.array(template['points'])
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
            width=0.032, height=0.016, depth=0.016)  # 32x16x16mm
        
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
    
    def run_enhanced_bin_picking_pipeline(self, template_library_dir=None, enable_visualization=True):
        """
        Main pipeline implementing the enhanced bin picking algorithm for stacked LEGO bricks
        
        Pipeline Steps:
        1. Data acquisition and preprocessing
        2. Find closest cluster to camera
        3. Color filtering for topmost brick
        4. Harris 3D and spin image pose matching
        5. World coordinate calculation
        6. Visualization for verification
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
        
        topmost_points, topmost_colors, cluster_label = self.find_and_extract_topmost_brick(
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
            topmost_points, topmost_colors, template_library_dir)
        
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
                'label': cluster_label,
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

    # Saving processed data
    def save_transformed_point_cloud(self, points, colors, output_file):
        data = np.hstack([points, colors])
        np.savetxt(output_file, data, delimiter=' ', fmt='%f')

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

    # ========== COLOR CLASSIFICATION AND FILTERING ==========
    
    # HSV-based color classification and dominant color detection for LEGO bricks
    def classify_hsv_color(self, rgb_color):
        """
        Classify RGB color into LEGO brick categories using HSV color space
        
        Args:
            rgb_color: RGB color array [R, G, B] in range [0, 1]
        
        Returns:
            Color name string and HSV values
        """
        # Convert RGB to HSV
        rgb_255 = (rgb_color * 255).astype(np.uint8)
        hsv = cv2.cvtColor(np.uint8([[rgb_255]]), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv[0], hsv[1], hsv[2]
        
        # LEGO color classification based on HSV ranges
        # H: 0-179, S: 0-255, V: 0-255 in OpenCV
        
        # Low saturation = white/gray/black
        if s < 50:
            if v > 200:
                return "white", hsv
            elif v < 80:
                return "black", hsv
            else:
                return "gray", hsv
        
        # High saturation colors
        if h < 10 or h > 170:  # Red range (wraps around)
            return "red", hsv
        elif 10 <= h < 25:  # Orange range
            return "orange", hsv
        elif 25 <= h < 35:  # Yellow range
            return "yellow", hsv
        elif 35 <= h < 85:  # Green range
            return "green", hsv
        elif 85 <= h < 130:  # Light blue range
            return "light_blue", hsv
        elif 130 <= h <= 170:  # Dark blue range
            return "dark_blue", hsv
        else:
            return "unknown", hsv
    
    def get_dominant_color(self, colors):
        """
        Get dominant color using HSV classification for better color distinction
        
        Args:
            colors: Array of RGB colors in range [0, 1]
        
        Returns:
            Dominant RGB color and its HSV classification name
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
            
        try:
            triangulation = Delaunay(points)
        except:
            # If Delaunay fails, return empty neighborhood
            print("Delaunay triangulation failed")
            return {}

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

    # ========== STEP 4: HARRIS CORNER DETECTION FOR KEYPOINTS ==========

    def compute_harris_3d_corners_multi_hypothesis(self, points, delta=0.025, harris_k=0.04, 
                                                 cluster_threshold=0.008, num_corners=8, num_hypotheses=3):
        """
        Enhanced Harris corner detection that returns multiple valid corner hypotheses
        to handle symmetrical LEGO bricks better
        """
        if len(points) < 10:
            print("Not enough points for Harris corner detection")
            return []

        print(f"Computing Harris corners with multiple hypotheses for {len(points)} points...")
        
        # STEP 1: Preprocess points for stable Harris detection
        preprocessed_points, transform_params = self.preprocess_for_harris_detection(points)
        
        # STEP 2: Compute neighborhoods using adaptive Delaunay triangulation
        neighborhood = self.compute_delaunay_neighborhood(preprocessed_points, delta=delta)
        
        # STEP 3: Initialize response array
        resp = np.zeros(len(preprocessed_points))
        
        # STEP 4: Compute Harris response for each point 
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
        
        # STEP 5: LEGO-specific feature filtering and corner candidate selection
        candidate = []
        for point_idx in neighborhood.keys():
            if len(neighborhood[point_idx]) > 0:
                neighbor_responses = resp[neighborhood[point_idx]]
                if resp[point_idx] >= np.max(neighbor_responses):
                    candidate.append([point_idx, resp[point_idx]])
        
        if len(candidate) == 0:
            print("No corner candidates found")
            return []
        
        # Sort by decreasing Harris response
        candidate.sort(reverse=True, key=lambda x: x[1])
        candidate = np.array(candidate)
        
        # STEP 5.1: Apply LEGO-specific filtering to reduce stud detection
        filtered_candidate = self.filter_lego_stud_features(preprocessed_points, candidate, points)
        
        if len(filtered_candidate) == 0:
            print("No corner candidates remaining after LEGO filtering, using original candidates")
            filtered_candidate = candidate
        
        # STEP 6: Generate multiple hypotheses with different starting points
        all_hypotheses = []
        
        for hypothesis_idx in range(min(num_hypotheses, len(filtered_candidate))):
            # Start with different high-scoring corners for each hypothesis
            selected_corners = []
            if len(filtered_candidate) > hypothesis_idx:
                # Start with the hypothesis_idx-th best corner
                start_idx = hypothesis_idx
                selected_corners.append(int(filtered_candidate[start_idx, 0]))
                Q = preprocessed_points[int(filtered_candidate[start_idx, 0]), :].reshape((1, -1))
                
                # Add corners that are far enough from existing ones
                for i in range(len(filtered_candidate)):
                    if i == start_idx:
                        continue
                    query = preprocessed_points[int(filtered_candidate[i, 0]), :].reshape((1, -1))
                    distances = scipy.spatial.distance.cdist(query, Q, metric='euclidean')
                    if np.min(distances) > cluster_threshold:
                        selected_corners.append(int(filtered_candidate[i, 0]))
                        Q = np.concatenate((Q, query), axis=0)
                        
                        # Stop if we have enough corners
                        if len(selected_corners) >= num_corners:
                            break
                
                if len(selected_corners) > 0:
                    corner_points = points[selected_corners]
                    all_hypotheses.append(corner_points)
        
        print(f"Generated {len(all_hypotheses)} LEGO-filtered corner hypotheses")
        return all_hypotheses

    def select_best_pose_hypothesis(self, corner_hypotheses, brick_points):
        """
        Select the best pose hypothesis from multiple candidates
        Since LEGO bricks are symmetrical, any valid hypothesis with enough corners is acceptable
        """
        if len(corner_hypotheses) == 0:
            return np.array([])
            
        # Simply return the first hypothesis with enough corners (minimum 4)
        for i, corners in enumerate(corner_hypotheses):
            if len(corners) >= 4:  # Direct check instead of using validation function
                print(f"Selected hypothesis {i} with {len(corners)} corners")
                return corners
        
        # If no hypothesis has enough corners, return the one with the most corners
        if len(corner_hypotheses) > 0:
            best_idx = max(range(len(corner_hypotheses)), key=lambda i: len(corner_hypotheses[i]))
            print(f"Selected best available hypothesis {best_idx} with {len(corner_hypotheses[best_idx])} corners")
            return corner_hypotheses[best_idx]
            
        return np.array([])

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

    # ========== STEP 5: POSE ESTIMATION METHODS ==========
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
        Compute spin image descriptor for a point according to Johnson & Hebert
        Args:
            point: 3D point position (p in paper)
            normal: surface normal at point (n in paper)
            points: all points in the point cloud
            spin_size: size of the spin image grid
            max_radius: maximum radius for spin image
        Returns:
            spin_image: 2D spin image array
        """
        # Define cylindrical coordinate system as in the paper
        # α = perpendicular distance to line through p parallel to n
        # β = signed perpendicular distance to plane through p perpendicular to n
        
        # Vector from point p to each point x
        vectors = points - point
        
        # β coordinate: projection onto normal direction
        beta = np.dot(vectors, normal)
        
        # α coordinate: distance to line (perpendicular distance)
        # α = ||(x-p) - ((x-p)·n)n||
        projections = np.outer(beta, normal)  # ((x-p)·n)n for all points
        perpendicular = vectors - projections
        alpha = np.linalg.norm(perpendicular, axis=1)
        
        # Create spin image grid
        alpha_max = max_radius
        beta_min, beta_max = -max_radius, max_radius
        
        # Discretize coordinates
        alpha_bins = np.linspace(0, alpha_max, spin_size)
        beta_bins = np.linspace(beta_min, beta_max, spin_size)
        
        # Create 2D histogram (spin image)
        spin_image = np.zeros((spin_size, spin_size))
        
        # Only consider points within the cylindrical region
        valid_mask = (alpha <= alpha_max) & (beta >= beta_min) & (beta <= beta_max)
        valid_alpha = alpha[valid_mask]
        valid_beta = beta[valid_mask]
        
        if len(valid_alpha) > 0:
            # Convert to bin indices
            alpha_indices = np.digitize(valid_alpha, alpha_bins) - 1
            beta_indices = np.digitize(valid_beta, beta_bins) - 1
            
            # Clip to valid range
            alpha_indices = np.clip(alpha_indices, 0, spin_size - 1)
            beta_indices = np.clip(beta_indices, 0, spin_size - 1)
            
            # Accumulate in spin image
            for a_idx, b_idx in zip(alpha_indices, beta_indices):
                spin_image[b_idx, a_idx] += 1
        
        # Normalize spin image
        if np.sum(spin_image) > 0:
            spin_image = spin_image / np.sum(spin_image)
            
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

    def find_spin_image_correspondences(self, template_keypoints, template_normals, template_points,
                                      target_keypoints, target_normals, target_points, 
                                      correlation_threshold=0.7):
        """
        Find point correspondences based on spin image similarity
        """
        correspondences = []
        
        print(f"Computing spin images for {len(template_keypoints)} template and {len(target_keypoints)} target keypoints...")
        
        # Compute spin images for template keypoints
        template_spin_images = []
        for i, (kp, normal) in enumerate(zip(template_keypoints, template_normals)):
            spin_img = self.compute_spin_image(kp, normal, template_points)
            template_spin_images.append(spin_img)
        
        # Compute spin images for target keypoints
        target_spin_images = []
        for i, (kp, normal) in enumerate(zip(target_keypoints, target_normals)):
            spin_img = self.compute_spin_image(kp, normal, target_points)
            target_spin_images.append(spin_img)
        
        # Find correspondences based on correlation
        for i, template_spin in enumerate(template_spin_images):
            best_correlation = -1
            best_match = -1
            
            for j, target_spin in enumerate(target_spin_images):
                correlation = self.compute_spin_image_correlation(template_spin, target_spin)
                
                if correlation > best_correlation and correlation > correlation_threshold:
                    best_correlation = correlation
                    best_match = j
            
            if best_match >= 0:
                correspondences.append((i, best_match, best_correlation))
                
        print(f"Found {len(correspondences)} spin image correspondences with correlation > {correlation_threshold}")
        return correspondences

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
        Complete pose estimation pipeline using spin images following the correct algorithm:
        1. Interest Point Detection using Harris Corner Detection
        2. Spin Image generation using 3D mesh of the object  
        3. Calculate R value to determine correlation between model and template
        4. Select point correspondence pairs
        5. Transform and Rotate template to correspond with model
        6. Calculate Pe and Ce to validate whether they are below threshold
        """
        print("Starting spin image-based pose estimation...")
        
        # Step 1: Interest Point Detection using Harris Corner Detection
        print("Step 1: Detecting interest points using Harris Corner Detection...")
        template_hypotheses = self.compute_harris_3d_corners_multi_hypothesis(
            template_points, num_corners=20, num_hypotheses=1)
        target_hypotheses = self.compute_harris_3d_corners_multi_hypothesis(
            target_points, num_corners=20, num_hypotheses=1)
        
        if len(template_hypotheses) == 0 or len(target_hypotheses) == 0:
            print("Failed to detect Harris corner interest points")
            return None, None
        
        # Use the first (best) hypothesis for both template and target
        template_keypoints = template_hypotheses[0]
        target_keypoints = target_hypotheses[0]
        
        print(f"Detected {len(template_keypoints)} template and {len(target_keypoints)} target interest points")
        
        # Step 2: Spin Image generation using 3D mesh of the object
        print("Step 2: Computing surface normals for spin image generation...")
        template_normals = self.compute_surface_normals(template_points)
        target_normals = self.compute_surface_normals(target_points)
        
        # Get normals for interest points (keypoints)
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
        
        print("Step 2: Generating spin images for interest points...")
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
        
        # Step 3: Calculate R value to determine correlation between model and template
        print("Step 3: Computing correlations between spin images...")
        correspondences = []
        
        for i, template_spin in enumerate(template_spin_images):
            best_correlation = -1
            best_match = -1
            
            for j, target_spin in enumerate(target_spin_images):
                # Calculate R(P,Q) correlation coefficient
                correlation = self.compute_spin_image_correlation(template_spin, target_spin)
                
                if correlation > best_correlation and correlation > correlation_threshold:
                    best_correlation = correlation
                    best_match = j
            
            if best_match >= 0:
                correspondences.append((i, best_match, best_correlation))
        
        print(f"Step 3: Found {len(correspondences)} correspondences with correlation > {correlation_threshold}")
        
        if len(correspondences) < 3:
            print("Insufficient correspondences for pose estimation")
            return None, None
        
        # Step 4: Select point correspondence pairs and estimate transformation
        print("Step 4: Estimating rigid transformation using enhanced alignment algorithm...")
        
        # Extract corresponding points from correspondences
        template_corr_points = []
        target_corr_points = []
        for temp_idx, targ_idx, correlation in correspondences:
            template_corr_points.append(template_keypoints[temp_idx])
            target_corr_points.append(target_keypoints[targ_idx])
        
        template_corr_points = np.array(template_corr_points)
        target_corr_points = np.array(target_corr_points)
        
        # Apply enhanced alignment algorithm with RANSAC integration
        R, T = self.apply_enhanced_alignment_algorithm(
            template_corr_points, target_corr_points,
            ransac_iterations=ransac_iterations,
            distance_threshold=inlier_threshold,
            min_inliers=max(3, len(correspondences) // 3)  # Require at least 1/3 of correspondences as inliers
        )
        
        if R is None:
            print("Step 4: Enhanced alignment algorithm failed to find valid pose")
            return None, None
        
        # Step 5: Transform and Rotate template to correspond with model
        # Step 6: Calculate Pe and Ce validation (already done in enhanced algorithm)
        print("Step 5-6: Pose estimation with enhanced alignment completed")
        R_refined, T_refined = R, T  # Enhanced algorithm already includes refinement
        
        print(f"Pose estimation completed successfully")
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

    # ========== TEMPLATE LIBRARY GENERATION METHODS ==========
    
    def generate_template_library_from_cad(self, cad_file_path, output_dir, 
                                         rotation_step_degrees=5,
                                         camera_distance=0.5,
                                         use_brick_symmetry=True):
        """
        Generate template library from CAD model for different poses
        Optimized for LEGO brick symmetry to reduce template count
        
        Args:
            cad_file_path: Path to CAD model (PLY, STL, OBJ formats)
            output_dir: Directory to save template point clouds
            rotation_step_degrees: Angular step for pose sampling
            camera_distance: Distance from camera to object
            use_brick_symmetry: If True, utilize brick symmetry to reduce templates
        """
        try:
            # Load CAD model
            if cad_file_path.endswith('.ply'):
                mesh = o3d.io.read_triangle_mesh(cad_file_path)
            elif cad_file_path.endswith('.stl'):
                mesh = o3d.io.read_triangle_mesh(cad_file_path)
            elif cad_file_path.endswith('.obj'):
                mesh = o3d.io.read_triangle_mesh(cad_file_path)
            else:
                raise ValueError("Unsupported CAD file format. Use PLY, STL, or OBJ.")
            
            if len(mesh.vertices) == 0:
                raise ValueError("Failed to load CAD model or model is empty")
            
            print(f"Loaded CAD model with {len(mesh.vertices)} vertices")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate templates for different poses
            template_count = 0
            templates_info = []
            
            # Optimize rotation ranges based on LEGO brick symmetry
            if use_brick_symmetry:
                # LEGO brick is symmetrical, so we only need:
                # Y-axis: 0-180° (instead of 0-360°) due to 180° rotational symmetry
                # X-axis: 0-90° (instead of 0-180°) due to top-bottom symmetry when grasping
                x_range = range(0, 91, rotation_step_degrees)  # 0-90°
                y_range = range(0, 180, rotation_step_degrees)  # 0-180°
                print("Using optimized rotation ranges for LEGO brick symmetry:")
                print(f"X-axis: 0-90° (step: {rotation_step_degrees}°)")
                print(f"Y-axis: 0-180° (step: {rotation_step_degrees}°)")
            else:
                # Full rotation ranges (original method)
                x_range = range(0, 181, rotation_step_degrees)  # 0-180°
                y_range = range(0, 360, rotation_step_degrees)  # 0-360°
                print("Using full rotation ranges:")
                print(f"X-axis: 0-180° (step: {rotation_step_degrees}°)")
                print(f"Y-axis: 0-360° (step: {rotation_step_degrees}°)")
            
            # Sample rotations around Y-axis and X-axis
            for x_rot in x_range:
                for y_rot in y_range:
                    # Create rotation matrix
                    R_x = mesh.get_rotation_matrix_from_xyz([np.radians(x_rot), 0, 0])
                    R_y = mesh.get_rotation_matrix_from_xyz([0, np.radians(y_rot), 0])
                    R_total = R_y @ R_x
                    
                    # Create a copy of the mesh by copying vertices and triangles
                    import copy
                    mesh_rotated = o3d.geometry.TriangleMesh()
                    mesh_rotated.vertices = copy.deepcopy(mesh.vertices)
                    mesh_rotated.triangles = copy.deepcopy(mesh.triangles)
                    if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) > 0:
                        mesh_rotated.vertex_normals = copy.deepcopy(mesh.vertex_normals)
                    
                    # Apply rotation to mesh
                    mesh_rotated.rotate(R_total, center=(0, 0, 0))
                    
                    # Generate point cloud from rotated mesh
                    # Sample points on surface
                    point_cloud = mesh_rotated.sample_points_uniformly(number_of_points=2000)
                    
                    # Simulate depth capture from camera viewpoint
                    # Camera looking down negative Z-axis
                    points = np.asarray(point_cloud.points)
                    
                    # Translate object to camera distance
                    points[:, 2] += camera_distance
                    
                    # Filter points visible from camera (front-facing)
                    # Simple visibility check - keep points with Z > 0
                    visible_mask = points[:, 2] > 0
                    visible_points = points[visible_mask]
                    
                    if len(visible_points) < 100:  # Skip if too few visible points
                        continue
                    
                    # Add some noise to simulate real depth sensor
                    noise_level = 0.001  # 1mm noise
                    noise = np.random.normal(0, noise_level, visible_points.shape)
                    noisy_points = visible_points + noise
                    
                    # Save template
                    template_file = os.path.join(output_dir, f"template_{template_count:03d}.ply")
                    
                    # Create point cloud object and save
                    template_pcd = o3d.geometry.PointCloud()
                    template_pcd.points = o3d.utility.Vector3dVector(noisy_points)
                    o3d.io.write_point_cloud(template_file, template_pcd)
                    
                    # Save template info with normalized rotation values
                    template_info = {
                        'id': template_count,
                        'file': template_file,
                        'x_rotation': x_rot,
                        'y_rotation': y_rot,
                        'rotation_matrix': R_total.tolist(),
                        'num_points': len(noisy_points),
                        'symmetry_optimized': use_brick_symmetry
                    }
                    templates_info.append(template_info)
                    
                    template_count += 1
            
            # Save template library metadata
            import json
            metadata_file = os.path.join(output_dir, "template_library.json")
            with open(metadata_file, 'w') as f:
                json.dump(templates_info, f, indent=2)
            
            print(f"Generated {template_count} templates saved to {output_dir}")
            if use_brick_symmetry:
                original_count = len(range(0, 181, rotation_step_degrees)) * len(range(0, 360, rotation_step_degrees))
                reduction_percentage = (1 - template_count / original_count) * 100
                print(f"Symmetry optimization reduced templates by {reduction_percentage:.1f}%")
                print(f"(Would have been {original_count} templates without optimization)")
            print(f"Template metadata saved to {metadata_file}")
            
            return templates_info
            
        except Exception as e:
            print(f"Error generating template library: {e}")
            return []

    def load_template_library(self, template_dir):
        """
        Load template library from directory
        """
        import json
        
        metadata_file = os.path.join(template_dir, "template_library.json")
        
        if not os.path.exists(metadata_file):
            print(f"Template metadata file not found: {metadata_file}")
            return []
        
        try:
            with open(metadata_file, 'r') as f:
                templates_info = json.load(f)
            
            print(f"Loaded template library with {len(templates_info)} templates")
            return templates_info
            
        except Exception as e:
            print(f"Error loading template library: {e}")
            return []

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
        
        if len(neighborhood) == 0:
            print("Failed to compute neighborhoods, falling back to k-NN")
            neighborhood = self.compute_knn_neighborhood(preprocessed_points, k=6)
        
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
    def run_pipeline(self, template_library_dir=None, enable_spin_images=False):
        print("0. Starting pipeline...")
        print(f"   - Spin image pose estimation: {'Enabled' if enable_spin_images else 'Disabled'}")
        if template_library_dir:
            print(f"   - Template library: {template_library_dir}")
            
        points, colors = self.capture_and_preprocess_kinect_data()
        # Check if the pipeline is running or not
        if len(points) == 0:
            print("[PIPELINE] No valid points found after preprocessing. Exiting.")
            return
        
        print("1. Data captured and preprocessed.")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        transformed_file = os.path.join(self.output_dir, f"Transformed_ROI_point_cloud_{timestamp_str}.txt")
        summary_file = os.path.join(self.output_dir, f"Cluster_Summary_{timestamp_str}.txt")
        image_file = os.path.join(self.output_dir, f"PointCloud_Img_{timestamp_str}.png")
        
        print("2. Saving transformed point cloud...")
        self.save_transformed_point_cloud(points, colors, transformed_file)
        
        # Also save as PLY format for compatibility with 3D_Harris_IPD tools
        ply_file = os.path.join(self.output_dir, f"Transformed_ROI_point_cloud_{timestamp_str}.ply")
        self.save_point_cloud_as_ply(points, colors, ply_file)
        
        print("3. Saving point cloud image...")
        self.save_cloud_image(points, colors, image_file)
        
        print("4. Clustering, detecting corners, and pose estimation...")
        # Use parameters optimized for LEGO bricks detection
        self.cluster_and_save_summary(transformed_file, summary_file,
                                    dbscan_eps=0.01, dbscan_min_samples=10,
                                    harris_delta=0.02,  # Smaller neighborhood for LEGO brick details
                                    harris_k=0.04,      # Standard Harris parameter
                                    harris_fraction=0.15, # Select more potential corners
                                    harris_cluster_threshold=0.008, # Closer corners allowed for LEGO
                                    harris_num_corners=12,  # More corners per brick
                                    template_library_dir=template_library_dir,
                                    enable_spin_images=enable_spin_images)
        
        print("5. Sending summary file to server...")
        # Server transfer function is commented out
        # try:
        #     self.send_file_via_tcp(summary_file)
        #     print("[PIPELINE] Summary file sent successfully.")
        # except Exception as e:
        #     print(f"[ERROR] Failed to send file to server: {e}")

        print("[PIPELINE] All processes completed.")

    def generate_lego_templates(self, cad_file_path, output_dir):
        """
        Convenience method to generate LEGO brick template library
        """
        print("Generating LEGO brick template library...")
        templates = self.generate_template_library_from_cad(
            cad_file_path, output_dir, 
            rotation_step_degrees=10,  # 20-degree steps as mentioned in paper
            camera_distance=0.5)
        
        if len(templates) > 0:
            print(f"Successfully generated {len(templates)} LEGO brick templates")
            return True
        else:
            print("Failed to generate template library")
            return False
        
if __name__ == "__main__":
    # Initialize the bin picking system
    system = BinPickingSystem(wdf_path="")
    
    # Example 1: Generate LEGO brick template library from CAD model (optional)
    # cad_file = "C:\\Users\\FILAB\\Desktop\\DUY\\LegoBrick_4_2.stl"  # or .stl, .obj
    # template_output_dir = "C:\\Users\\FILAB\\Desktop\\DUY\\templates"
    # success = system.generate_lego_templates(cad_file, template_output_dir)
    # if success:
    #     print("Template library generated successfully!")
    
    # Example 2: Run ENHANCED pipeline for stacked LEGO bricks (RECOMMENDED)
    print("Running ENHANCED pipeline for stacked LEGO bricks...")
    template_library_dir = None  # Set to template directory path if available
    # template_library_dir = "C:\\Users\\FILAB\\Desktop\\DUY\\templates\\lego_brick"  # Uncomment if you have templates

    enhanced_results = system.run_enhanced_bin_picking_pipeline(
        template_library_dir=template_library_dir,
        enable_visualization=True
    )
    
    if enhanced_results:
        print("\nENHANCED PIPELINE SUMMARY:")
        print(f"- Method: {enhanced_results['world_coordinates']['method']}")
        print(f"- Position: {enhanced_results['world_coordinates']['position']}")
        print(f"- Rotation: {enhanced_results['world_coordinates']['rotation']}")
        print(f"- Match Score: {enhanced_results['world_coordinates']['match_score']:.3f}")
        print(f"- Cluster Size: {enhanced_results['cluster_info']['filtered_size']} points")
    
    # Example 3: Run traditional pipeline with Harris corner detection (for comparison)
    # print("Running traditional pipeline with Harris corner detection...")
    # system.run_pipeline()
    
    # Example 4: Run pipeline with spin image pose estimation (research paper method)
    # Uncomment to use spin image-based pose estimation with template matching
    # print("Running pipeline with spin image pose estimation...")
    # template_library_dir = "C:\\Users\\FILAB\\Desktop\\DUY\\templates\\lego_brick"  # Directory with generated templates
    # system.run_pipeline(template_library_dir=template_library_dir, enable_spin_images=True)
    
    # Example 5: Load and use existing template library
    # templates = system.load_template_library("C:\\Users\\FILAB\\Desktop\\DUY\\templates\\lego_brick")
    # if templates:
    #     print(f"Loaded {len(templates)} templates from library")
    
    print("Bin picking system completed!")