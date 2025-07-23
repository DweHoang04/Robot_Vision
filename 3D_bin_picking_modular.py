# 3D_bin_picking.py
# Main Application File for 3D Bin Picking System
# Integrates Kinect V2 data capture with modular processing libraries

# Essential Libraries
import os
import socket
import time
import cv2
import numpy as np
import open3d as o3d
import struct
from datetime import datetime

# Kinect Libraries
from pykinect2 import PyKinectRuntime, PyKinectV2
from pykinect2.PyKinectV2 import *

# Import our modular libraries
from transformation import PointCloudTransformation
from harris_detection import Harris3DDetector
from clustering import ClusteringAnalysis


class BinPickingSystem:
    """
    Main 3D Bin Picking System for LEGO brick detection using Kinect V2
    Integrates modular libraries for transformation, Harris detection, and clustering
    """

    def __init__(self, wdf_path, output_dir=None):
        """
        Initialize the bin picking system
        
        Args:
            wdf_path: path to WDF file (not used in current implementation)
            output_dir: directory where to save results
        """
        self.output_dir = "C:\\Users\\FILAB\\Desktop\\DUY\\Results" # Data saving location
        # self.host = "192.168.1.23" # Target IP for sending data to the robot arm
        # self.port = 9999 # Target (Robot arm) port

    def capture_and_preprocess_kinect_data(self, roi_x=195, roi_y=50, roi_w=245, roi_h=300,
                                           plane_dist_thresh=0.005, ransac_n=3, ransac_iter=1000,
                                           boundary_margin=0.005, dbscan_eps_pre=0.01, 
                                           dbscan_min_samples_pre=50):
        """
        Main depth and RGB data acquiring and processing using Kinect V2
        
        Args:
            roi_x, roi_y, roi_w, roi_h: Region of Interest parameters
            plane_dist_thresh: RANSAC plane segmentation threshold
            ransac_n: minimum number of points for RANSAC
            ransac_iter: number of RANSAC iterations
            boundary_margin: margin for boundary filtering
            dbscan_eps_pre: DBSCAN epsilon for preprocessing
            dbscan_min_samples_pre: DBSCAN minimum samples for preprocessing
            
        Returns:
            tuple: (transformed_points, colors) processed point cloud data
        """
        # Initialize Kinect with depth and color streams
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
        above_points, above_colors = PointCloudTransformation.keep_points_above_plane(
            np.asarray(non_plane_cloud.points),
            np.asarray(non_plane_cloud.colors),
            plane_model)
        
        # If there is no point above the plane then return null arrays
        if len(above_points) == 0:
            return np.array([]), np.array([])

        # Removing margin points
        x_min, y_min, _ = np.min(above_points, axis=0)
        x_max, y_max, _ = np.max(above_points, axis=0)
        margin_points, margin_colors = PointCloudTransformation.keep_inside_boundary_points(
            above_points, above_colors, x_min, x_max, y_min, y_max, margin=boundary_margin)
        
        if len(margin_points) == 0:
            return np.array([]), np.array([])

        # Apply DBSCAN for noise removal
        denoised_points, denoised_colors = ClusteringAnalysis.apply_dbscan(
            margin_points, margin_colors, eps=dbscan_eps_pre, min_samples=dbscan_min_samples_pre)
        
        if len(denoised_points) == 0:
            return np.array([]), np.array([])

        # Apply coordinate transformation to normalize points to world coordinate
        transformed = PointCloudTransformation.transform_point_cloud_to_world(denoised_points)
        return transformed, denoised_colors

    # def send_file_via_tcp(self, file_path):
    #     """
    #     Send file to robot arm via TCP connection (currently disabled)
    #     """
    #     filename = os.path.basename(file_path).encode('utf-8')
    #     with open(file_path, "rb") as f:
    #         file_data = f.read()
    #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #         s.connect((self.host, self.port))
    #         s.sendall(struct.pack('>I', len(filename)))
    #         s.sendall(filename)
    #         s.sendall(file_data)

    def run_pipeline(self):
        """
        Main pipeline execution - captures data, processes it, and saves results
        """
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
        ClusteringAnalysis.save_transformed_point_cloud(points, colors, transformed_file)
        
        # Also save as PLY format for compatibility with 3D_Harris_IPD tools
        ply_file = os.path.join(self.output_dir, f"Transformed_ROI_point_cloud_{timestamp_str}.ply")
        ClusteringAnalysis.save_point_cloud_as_ply(points, colors, ply_file)
        
        print("3. Saving point cloud image...")
        ClusteringAnalysis.save_cloud_image(points, colors, image_file)
        
        print("4. Clustering, detecting corners, and saving summary...")
        # Use parameters optimized for LEGO bricks detection
        ClusteringAnalysis.cluster_and_save_summary(
            transformed_file, summary_file,
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
