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
from scipy.spatial import Delaunay, ConvexHull

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

    def detect_brick_edges_and_corners(self, points, edge_threshold=0.01):
        """
        Alternative method: Directly detect brick edges and corners using geometric analysis
        
        Args:
            points: Point cloud of LEGO brick
            edge_threshold: Distance threshold for edge detection
        
        Returns:
            Edge points and corner points
        """
        print("Detecting brick edges and corners using geometric analysis...")
        
        # Compute 2D convex hull to find brick boundary
        points_2d = points[:, :2]  # Project to XY plane
        hull = ConvexHull(points_2d)
        boundary_indices = hull.vertices
        
        # Get boundary points
        boundary_points = points[boundary_indices]
        
        # Detect corners as points where edge direction changes significantly
        corner_points = []
        edge_points = []
        
        for i in range(len(boundary_points)):
            p1 = boundary_points[i-1]
            p2 = boundary_points[i]
            p3 = boundary_points[(i+1) % len(boundary_points)]
            
            # Compute edge vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Normalize vectors
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
            
            # Compute angle between edges
            dot_product = np.dot(v1_norm, v2_norm)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            
            # Corner if significant direction change
            if angle < np.pi - 0.3:  # Less than ~150 degrees
                corner_points.append(p2)
            else:
                edge_points.append(p2)
        
        corner_points = np.array(corner_points) if corner_points else np.array([]).reshape(0, 3)
        edge_points = np.array(edge_points) if edge_points else np.array([]).reshape(0, 3)
        
        print(f"Found {len(corner_points)} corners and {len(edge_points)} edge points")
        return corner_points, edge_points

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
        print("Step 4: Estimating rigid transformation using RANSAC...")
        R, T, inliers = self.estimate_rigid_transformation_ransac(
            template_keypoints, target_keypoints, correspondences,
            ransac_iterations, inlier_threshold)
        
        if R is None:
            print("Step 4: RANSAC failed to find valid pose")
            return None, None
        
        # Step 5: Transform and Rotate template to correspond with model
        # Step 6: Calculate Pe and Ce validation (already done in RANSAC)
        print("Step 5-6: Refining pose with ICP...")
        R_refined, T_refined = self.refine_pose_with_icp(template_points, target_points, R, T)
        
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

    def convert_to_robot_coordinates(self, coordinates, camera_to_robot_transform=None):
        """
        Convert brick coordinates from camera coordinate system to robot coordinate system
        
        Args:
            coordinates: Brick coordinates from extract_brick_coordinates()
            camera_to_robot_transform: 4x4 transformation matrix from camera to robot base
                                     If None, uses identity (assumes camera and robot frames are aligned)
        
        Returns:
            Brick coordinates in robot coordinate system
        """
        if camera_to_robot_transform is None:
            # Default: assume camera and robot frames are aligned
            print("Warning: Using identity transform for camera-to-robot conversion")
            print("Please calibrate camera-to-robot transformation for accurate coordinates")
            return coordinates
        
        # Extract position and rotation from coordinates
        position = np.array(coordinates['position'])
        R_camera = np.array(coordinates['rotation_matrix'])
        
        # Create 4x4 transformation matrix in camera frame
        T_camera = np.eye(4)
        T_camera[:3, :3] = R_camera
        T_camera[:3, 3] = position
        
        # Transform to robot coordinate system
        T_robot = camera_to_robot_transform @ T_camera
        
        # Extract robot coordinates
        robot_position = T_robot[:3, 3]
        robot_rotation = T_robot[:3, :3]
        
        # Extract coordinates in robot frame
        robot_coordinates = self.extract_brick_coordinates(
            robot_rotation, robot_position, coordinates['coordinate_system'])
        
        # Copy additional information
        for key in ['template_id', 'template_x_rotation', 'template_y_rotation']:
            if key in coordinates:
                robot_coordinates[key] = coordinates[key]
        
        robot_coordinates['coordinate_frame'] = 'robot'
        
        print("Converted coordinates to robot frame:")
        print(f"  Robot position (x, y, z): ({robot_coordinates['position'][0]:.3f}, "
              f"{robot_coordinates['position'][1]:.3f}, {robot_coordinates['position'][2]:.3f}) meters")
        
        return robot_coordinates

    def save_brick_coordinates_for_robot(self, results, output_file, coordinate_system='euler_xyz'):
        """
        Save brick coordinates in a robot-friendly format
        
        Args:
            results: Results from cluster_and_save_summary()
            output_file: Path to save robot coordinates file
            coordinate_system: Coordinate system for rotations
        """
        robot_coords_file = output_file.replace('.txt', '_robot_coordinates.txt')
        
        with open(robot_coords_file, 'w') as f:
            f.write("# LEGO Brick Coordinates for Robot Arm\n")
            f.write("# Format: brick_id x(m) y(m) z(m) rx(deg) ry(deg) rz(deg) confidence\n")
            f.write("# Coordinate system: camera frame\n")
            f.write("# Note: Apply camera-to-robot transformation for robot coordinates\n")
            f.write("\n")
            
            brick_count = 0
            for result in results:
                if 'brick_position' in result and 'brick_rotation' in result:
                    brick_count += 1
                    
                    # Extract coordinates
                    x, y, z = result['brick_position']
                    rx, ry, rz = result['brick_rotation']
                    
                    # Calculate confidence based on match score
                    confidence = result.get('match_score', 0.0)
                    
                    # Write robot-friendly format
                    f.write(f"BRICK_{brick_count:03d} {x:.6f} {y:.6f} {z:.6f} {rx:.2f} {ry:.2f} {rz:.2f} {confidence:.3f}\n")
                    
                    # Additional information for debugging
                    f.write(f"# Template_ID: {result.get('template_id', 'N/A')}\n")
                    f.write(f"# Method: {result.get('method', 'N/A')}\n")
                    f.write("\n")
        
        print(f"Saved robot coordinates: {robot_coords_file}")
        print(f"Found {brick_count} bricks with pose information")
        
        return robot_coords_file

    # ========== TEMPLATE LIBRARY GENERATION METHODS ==========
    
    def generate_template_library_from_cad(self, cad_file_path, output_dir, 
                                         rotation_step_degrees=20,
                                         camera_distance=0.5):
        """
        Generate template library from CAD model for different poses
        Args:
            cad_file_path: Path to CAD model (PLY, STL, OBJ formats)
            output_dir: Directory to save template point clouds
            rotation_step_degrees: Angular step for pose sampling
            camera_distance: Distance from camera to object
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
            
            # Sample rotations around Y-axis (0-360°) and X-axis (0-180°)
            for x_rot in range(0, 181, rotation_step_degrees):
                for y_rot in range(0, 360, rotation_step_degrees):
                    # Create rotation matrix
                    R_x = mesh.get_rotation_matrix_from_xyz([np.radians(x_rot), 0, 0])
                    R_y = mesh.get_rotation_matrix_from_xyz([0, np.radians(y_rot), 0])
                    R_total = R_y @ R_x
                    
                    # Apply rotation to mesh
                    mesh_rotated = mesh.copy()
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
                    
                    # Save template info
                    template_info = {
                        'id': template_count,
                        'file': template_file,
                        'x_rotation': x_rot,
                        'y_rotation': y_rot,
                        'rotation_matrix': R_total.tolist(),
                        'num_points': len(noisy_points)
                    }
                    templates_info.append(template_info)
                    
                    template_count += 1
            
            # Save template library metadata
            import json
            metadata_file = os.path.join(output_dir, "template_library.json")
            with open(metadata_file, 'w') as f:
                json.dump(templates_info, f, indent=2)
            
            print(f"Generated {template_count} templates saved to {output_dir}")
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
                                max_templates_to_test=50):
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
        
        # Save robot-friendly coordinates if pose estimation was successful
        if any('brick_position' in result for result in results):
            self.save_brick_coordinates_for_robot(results, summary_file)
        
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
    
    # Example 1: Generate LEGO brick template library from CAD model
    # Uncomment and modify the path to your LEGO CAD file
    # cad_file = "path/to/your/lego_brick.ply"  # or .stl, .obj
    # template_output_dir = "templates/lego_brick"
    # success = system.generate_lego_templates(cad_file, template_output_dir)
    # if success:
    #     print("Template library generated successfully!")
    
    # Example 2: Run pipeline with Harris corner detection (traditional method)
    print("Running pipeline with Harris corner detection...")
    system.run_pipeline()
    
    # Example 3: Run pipeline with spin image pose estimation (research paper method)
    # Uncomment to use spin image-based pose estimation with template matching
    # print("Running pipeline with spin image pose estimation...")
    # template_library_dir = "templates/lego_brick"  # Directory with generated templates
    # system.run_pipeline(template_library_dir=template_library_dir, enable_spin_images=True)
    
    # Example 4: Load and use existing template library
    # templates = system.load_template_library("templates/lego_brick")
    # if templates:
    #     print(f"Loaded {len(templates)} templates from library")
    
    print("Bin picking system completed!")