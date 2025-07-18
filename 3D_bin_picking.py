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
        # Use provided output directory or create a default one
        if output_dir is None:
            self.output_dir = os.path.join(os.getcwd(), "Results")
        else:
            self.output_dir = output_dir
        
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
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
        
        print(f"Saved PLY file: {output_file}")

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
        ncols = (order + 1)**2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order+1), range(order+1))
        for k, (i, j) in enumerate(ij):
            G[:, k] = x**i * y**j
        m, _, _, _ = np.linalg.lstsq(G, z, rcond=None)
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

    def compute_harris_3d_corners(self, points, delta=0.025, harris_k=0.04, fraction=0.1, cluster_threshold=0.01, num_corners=8):
        """
        Improved 3D Harris corner detection based on the 3D_Harris_IPD implementation.
        
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
        
        # Compute neighborhoods using adaptive Delaunay triangulation
        neighborhood = self.compute_delaunay_neighborhood(points, delta=delta)
        
        if len(neighborhood) == 0:
            print("Failed to compute neighborhoods, falling back to k-NN")
            neighborhood = self.compute_knn_neighborhood(points, k=6)
        
        # Initialize response array
        resp = np.zeros(len(points))
        
        # Compute Harris response for each point
        for point_idx in neighborhood.keys():
            try:
                if len(neighborhood[point_idx]) < 3:
                    resp[point_idx] = -np.inf
                    continue
                    
                neighbors = points[neighborhood[point_idx], :]
                
                # Center the neighbors
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
        
        # Select interest points - find local maxima
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
        
        # Method 1: Select top fraction of points
        num_fraction = max(1, int(fraction * len(points)))
        interest_points_fraction = candidate[:num_fraction, 0].astype(int)
        
        # Method 2: Cluster-based selection (avoid points too close to each other)
        selected_corners = []
        if len(candidate) > 0:
            # Start with the best corner
            selected_corners.append(int(candidate[0, 0]))
            Q = points[int(candidate[0, 0]), :].reshape((1, -1))
            
            # Add corners that are far enough from existing ones
            for i in range(1, len(candidate)):
                query = points[int(candidate[i, 0]), :].reshape((1, -1))
                distances = scipy.spatial.distance.cdist(query, Q, metric='euclidean')
                if np.min(distances) > cluster_threshold:
                    selected_corners.append(int(candidate[i, 0]))
                    Q = np.concatenate((Q, query), axis=0)
                    
                    # Stop if we have enough corners
                    if len(selected_corners) >= num_corners:
                        break
        
        # Return the selected corner points
        if len(selected_corners) > 0:
            corner_points = points[selected_corners]
            print(f"Found {len(corner_points)} Harris corners")
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
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points)
        labels = dbscan.labels_

        results = []
        geometries = []

        for lbl in set(labels):
            if lbl == -1:
                continue
            m = (labels == lbl)
            lego_pts = points[m]
            lego_cols = colors[m]
            
            # Filter small clusters
            if len(lego_pts) < 500:
                continue
            
            print(f"Cluster {lbl}: {len(lego_pts)} points found.")

            # Add LEGO cluster as point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lego_pts)
            pcd.colors = o3d.utility.Vector3dVector(lego_cols)
            geometries.append(pcd)

            # Harris 3D keypoint detection with improved parameters for LEGO bricks
            if len(lego_pts) > 20:
                print(f"-> Detecting Harris corners for cluster {lbl}...")
                corners = self.compute_harris_3d_corners(lego_pts, 
                                                       delta=harris_delta, 
                                                       harris_k=harris_k, 
                                                       fraction=harris_fraction, 
                                                       cluster_threshold=harris_cluster_threshold, 
                                                       num_corners=harris_num_corners)
                print(f"-> Found {len(corners)} corners.")

                if len(corners) > 0:
                    for corner_pt in corners:
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
                        sphere.translate(corner_pt)
                        sphere.paint_uniform_color([1.0, 0.647, 0.0])  # Orange
                        geometries.append(sphere)

            # Orientation analysis
            pca = PCA(n_components=3).fit(lego_pts)
            center = np.mean(lego_pts, axis=0)
            angle_deg = self.calculate_y_axis_angle_xy(pca.components_[1])
            dom_color = self.get_dominant_color(lego_cols)
            results.append((center, dom_color, angle_deg))

        # Final interactive visualization
        if geometries:
            print("Visualizing results...")
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([axes] + geometries, window_name="Filtered Clusters with Harris Corners")

        # Save summary
        with open(summary_file, "w") as f:
            for center, color, angle in results:
                cx, cy, cz = center * 1000.0
                r_, g_, b_ = color
                f.write(f"{cx:.2f} {cy:.2f} {cz:.2f} {r_:.6f} {g_:.6f} {b_:.6f} {angle:.2f}\n")


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