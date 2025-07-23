# Essential Libraries
import os # Interacting with OS
import socket # Network communication between systems
import time # Time related functions
import cv2 # Used for image processing
import numpy as np # Matrix operation
import open3d as o3d # 3D data processing
import struct # Convert Python values to C struct to communicate with sensors (In this case: Kinect V2)
import scipy.spatial
from datetime import datetime # Time operation
from collections import Counter
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

    # Border filtering algorithm using AND logic
    def keep_inside_boundary_points(self, points, colors, x_min, x_max, y_min, y_max, margin=0.02):
        mask = (
            (points[:, 0] >= x_min + margin) & (points[:, 0] <= x_max - margin) &
            (points[:, 1] >= y_min + margin) & (points[:, 1] <= y_max - margin)
        ) # Removing the border by an amount of margin
        # The scanning range will be (x_min + margin, x_max - margin) x (y_min + margin, y_max - margin)
        return points[mask], colors[mask] # Return filtered point cloud and color values

    # Noise filtering in depth scanning using Density-based spatial clustering
    def apply_dbscan(self, points, colors, eps=0.05, min_samples=10):
        # eps: Radius around a point to search for neighbors
        # min_samples: Number of points used in a neighborhood to qualify as a core point
        # fit(points): Assigns cluster labels to each point
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = dbscan.labels_ # Points in dense regions get acluster label 
        mask = (labels != -1) # Noise points get a lable -1
        return points[mask], colors[mask]

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

    # Extracting the most dominant color from a set of RGB values
    def get_dominant_color(self, colors):
        c_int = (colors * 255).astype(int)
        counter = Counter(map(tuple, c_int))
        dom = max(counter, key=counter.get)
        return np.array(dom) / 255.0 # Normalizing it to float type

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
            # Fall back to simple k-NN if Delaunay fails
            return self.compute_knn_neighborhood(points, k=6)

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

    def compute_knn_neighborhood(self, points, k=6):
        """Fallback k-NN neighborhood computation"""
        neighborhoods = {}
        for i, query in enumerate(points):
            dist = np.linalg.norm(query - points, axis=1)
            sample_idx = []
            for _ in range(min(k, len(points))):
                idx = np.argmin(dist)
                sample_idx.append(idx)
                dist[idx] = np.inf
            neighborhoods[i] = sample_idx
        return neighborhoods

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

    def restore_harris_points(self, harris_points, original_points, transform_params):
        """
        Restore Harris corner points to original coordinate system
        """
        if len(harris_points) == 0:
            return harris_points
            
        restored_points = harris_points.copy()
        
        # Note: For this implementation, we'll keep points in the transformed space
        # since the clustering and analysis work better in the aligned coordinate system
        # If needed, we can add full inverse transformation here
        
        return restored_points

    def find_closest_brick_cluster(self, points, colors, dbscan_eps=0.01, dbscan_min_samples=10):
        """
        Find the closest brick cluster to the camera (smallest Z coordinate)
        Returns only the points and colors of the closest brick
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
            return np.array([]), np.array([])
        
        # Sort by Z coordinate (closest first - smallest Z)
        valid_clusters.sort(key=lambda x: x[1])
        
        # Return the closest cluster
        closest_label, closest_z, closest_points, closest_colors = valid_clusters[0]
        print(f"Focusing on closest brick cluster {closest_label} at distance Z={closest_z:.3f}m")
        print(f"Closest brick has {len(closest_points)} points")
        
        return closest_points, closest_colors

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
        
        if len(neighborhood) == 0:
            print("Failed to compute neighborhoods, falling back to k-NN")
            neighborhood = self.compute_knn_neighborhood(preprocessed_points, k=6)
        
        # STEP 3: Initialize response array
        resp = np.zeros(len(preprocessed_points))
        
        # STEP 4: Compute Harris response for each point (same as before)
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
        
        # STEP 5: Find multiple valid corner hypotheses
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
        
        # STEP 6: Generate multiple hypotheses with different starting points
        all_hypotheses = []
        
        for hypothesis_idx in range(min(num_hypotheses, len(candidate))):
            # Start with different high-scoring corners for each hypothesis
            selected_corners = []
            if len(candidate) > hypothesis_idx:
                # Start with the hypothesis_idx-th best corner
                start_idx = hypothesis_idx
                selected_corners.append(int(candidate[start_idx, 0]))
                Q = preprocessed_points[int(candidate[start_idx, 0]), :].reshape((1, -1))
                
                # Add corners that are far enough from existing ones
                for i in range(len(candidate)):
                    if i == start_idx:
                        continue
                    query = preprocessed_points[int(candidate[i, 0]), :].reshape((1, -1))
                    distances = scipy.spatial.distance.cdist(query, Q, metric='euclidean')
                    if np.min(distances) > cluster_threshold:
                        selected_corners.append(int(candidate[i, 0]))
                        Q = np.concatenate((Q, query), axis=0)
                        
                        # Stop if we have enough corners
                        if len(selected_corners) >= num_corners:
                            break
                
                if len(selected_corners) > 0:
                    corner_points = points[selected_corners]
                    all_hypotheses.append(corner_points)
        
        print(f"Generated {len(all_hypotheses)} corner hypotheses for symmetrical object")
        return all_hypotheses

    def validate_lego_geometry(self, corner_points, expected_length=64.0, expected_width=32.0, 
                              expected_height=19.2, tolerance=0.1):
        """
        Validate if detected corners form a valid LEGO Duplo brick geometry
        Standard 2x4 LEGO Duplo brick: 64mm x 32mm x 19.2mm
        """
        if len(corner_points) < 4:
            return False, 0.0
            
        # Calculate bounding box dimensions
        min_coords = np.min(corner_points, axis=0)
        max_coords = np.max(corner_points, axis=0)
        dimensions = (max_coords - min_coords) * 1000  # Convert to mm
        
        length, width, height = sorted(dimensions, reverse=True)
        
        # Check if dimensions match expected LEGO brick ratios
        length_error = abs(length - expected_length) / expected_length
        width_error = abs(width - expected_width) / expected_width
        height_error = abs(height - expected_height) / expected_height
        
        total_error = length_error + width_error + height_error
        
        # Valid if all dimensions are within tolerance
        is_valid = (length_error < tolerance and 
                   width_error < tolerance and 
                   height_error < tolerance)
        
        if is_valid:
            print(f"Valid LEGO Duplo geometry: {length:.1f}x{width:.1f}x{height:.1f}mm (error: {total_error:.3f})")
        
        return is_valid, total_error

    def select_best_pose_hypothesis(self, corner_hypotheses, brick_points):
        """
        Select the best pose hypothesis from multiple candidates using geometric validation
        """
        if len(corner_hypotheses) == 0:
            return np.array([])
            
        best_corners = None
        best_score = float('inf')
        
        for i, corners in enumerate(corner_hypotheses):
            is_valid, error_score = self.validate_lego_geometry(corners)
            
            if is_valid and error_score < best_score:
                best_score = error_score
                best_corners = corners
                print(f"Hypothesis {i}: Valid with score {error_score:.3f}")
            else:
                print(f"Hypothesis {i}: Invalid or poor score {error_score:.3f}")
        
        if best_corners is not None:
            print(f"Selected best hypothesis with validation score: {best_score:.3f}")
            return best_corners
        else:
            # Fallback: return the first hypothesis if none pass validation
            print("No hypothesis passed validation, using first hypothesis as fallback")
            return corner_hypotheses[0] if len(corner_hypotheses) > 0 else np.array([])
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
                             harris_cluster_threshold=0.008, harris_num_corners=12):
        data = np.loadtxt(transformed_file, delimiter=' ')
        points, colors = data[:, :3], data[:, 3:6]
        
        # Find and focus only on the closest brick to the camera
        closest_points, closest_colors = self.find_closest_brick_cluster(points, colors, 
                                                                        dbscan_eps, dbscan_min_samples)
        
        if len(closest_points) == 0:
            print("No valid brick found for analysis")
            return

        results = []
        geometries = []

        print(f"Analyzing closest brick with {len(closest_points)} points...")

        # Add the closest LEGO brick as point cloud for visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(closest_points)
        pcd.colors = o3d.utility.Vector3dVector(closest_colors)
        geometries.append(pcd)

        # Harris 3D keypoint detection for the closest brick only
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

        # Orientation analysis for the closest brick
        pca = PCA(n_components=3).fit(closest_points)
        center = np.mean(closest_points, axis=0)
        angle_deg = self.calculate_y_axis_angle_xy(pca.components_[1])
        dom_color = self.get_dominant_color(closest_colors)
        results.append((center, dom_color, angle_deg))

        # Final interactive visualization of the closest brick
        if geometries:
            print("Visualizing closest brick with Harris corners...")
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([axes] + geometries, 
                                            window_name="Closest Brick with Harris Corners")

        # Save analysis summary (only for the closest brick)
        with open(summary_file, "w") as f:
            for center, color, angle in results:
                cx, cy, cz = center * 1000.0  # Convert to millimeters
                r_, g_, b_ = color
                f.write(f"{cx:.2f} {cy:.2f} {cz:.2f} {r_:.6f} {g_:.6f} {b_:.6f} {angle:.2f}\n")
        
        print(f"Saved analysis summary for closest brick only: {summary_file}")


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
    def run_pipeline(self):
        print("0. Starting pipeline...")
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
        
        print("4. Clustering, detecting corners, and saving summary...")
        # Use parameters optimized for LEGO bricks detection
        self.cluster_and_save_summary(transformed_file, summary_file,
                                    dbscan_eps=0.01, dbscan_min_samples=10,
                                    harris_delta=0.02,  # Smaller neighborhood for LEGO brick details
                                    harris_k=0.04,      # Standard Harris parameter
                                    harris_fraction=0.15, # Select more potential corners
                                    harris_cluster_threshold=0.008, # Closer corners allowed for LEGO
                                    harris_num_corners=12)  # More corners per brick
        
        print("5. Sending summary file to server...")
        # Server transfer function is commented out
        # try:
        #     self.send_file_via_tcp(summary_file)
        #     print("[PIPELINE] Summary file sent successfully.")
        # except Exception as e:
        #     print(f"[ERROR] Failed to send file to server: {e}")

        print("[PIPELINE] All processes completed.")
        
if __name__ == "__main__":
    # The wdf_path is not used in the current implementation, so passing an empty string is fine.
    system = BinPickingSystem(wdf_path="")
    system.run_pipeline()