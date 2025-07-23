# clustering.py
# Clustering, Analysis, and Visualization Library
# Contains functions for DBSCAN clustering, orientation analysis, and 3D visualization

import numpy as np
import open3d as o3d
import time
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from harris_detection import Harris3DDetector


class ClusteringAnalysis:
    """
    A class containing clustering algorithms, orientation analysis, and visualization methods
    for LEGO brick detection and analysis in 3D point clouds.
    """

    @staticmethod
    def apply_dbscan(points, colors, eps=0.05, min_samples=10):
        """
        Noise filtering in depth scanning using Density-based spatial clustering
        
        Args:
            points: numpy array of 3D points
            colors: numpy array of RGB colors
            eps: radius around a point to search for neighbors
            min_samples: number of points used in a neighborhood to qualify as a core point
            
        Returns:
            tuple: (filtered_points, filtered_colors) with noise removed
        """
        # eps: Radius around a point to search for neighbors
        # min_samples: Number of points used in a neighborhood to qualify as a core point
        # fit(points): Assigns cluster labels to each point
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = dbscan.labels_ # Points in dense regions get a cluster label 
        mask = (labels != -1) # Noise points get a label -1
        return points[mask], colors[mask]

    @staticmethod
    def calculate_y_axis_angle_xy(minor_axis):
        """
        Calculate angles between the principal axis of the block and the principal axis of the camera
        
        Args:
            minor_axis: principal axis vector from PCA analysis
            
        Returns:
            float: angle in degrees representing rotation needed for alignment
        """
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

    @staticmethod
    def get_dominant_color(colors):
        """
        Extract the most dominant color from a set of RGB values
        
        Args:
            colors: numpy array of RGB color values (normalized 0-1)
            
        Returns:
            numpy array: dominant color as RGB values (normalized 0-1)
        """
        c_int = (colors * 255).astype(int)
        counter = Counter(map(tuple, c_int))
        dom = max(counter, key=counter.get)
        return np.array(dom) / 255.0 # Normalizing it to float type

    @staticmethod
    def save_cloud_image(points, colors, image_path):
        """
        Save a rendered image of the point cloud
        
        Args:
            points: numpy array of 3D points
            colors: numpy array of RGB colors
            image_path: path where to save the rendered image
        """
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

    @staticmethod
    def save_point_cloud_as_ply(points, colors, output_file):
        """
        Save point cloud data as PLY file format for compatibility with 3D_Harris_IPD tools
        
        Args:
            points: numpy array of 3D points
            colors: numpy array of RGB colors (normalized 0-1)
            output_file: path where to save the PLY file
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

    @staticmethod
    def save_transformed_point_cloud(points, colors, output_file):
        """
        Save processed point cloud data as text file
        
        Args:
            points: numpy array of 3D points
            colors: numpy array of RGB colors
            output_file: path where to save the text file
        """
        data = np.hstack([points, colors])
        np.savetxt(output_file, data, delimiter=' ', fmt='%f')

    @staticmethod
    def cluster_and_save_summary(transformed_file, summary_file,
                               dbscan_eps=0.01, dbscan_min_samples=10,
                               harris_delta=0.02, harris_k=0.04, harris_fraction=0.15,
                               harris_cluster_threshold=0.008, harris_num_corners=12):
        """
        Perform clustering analysis on transformed point cloud data and save summary results
        
        Args:
            transformed_file: path to input transformed point cloud data
            summary_file: path where to save the analysis summary
            dbscan_eps: DBSCAN epsilon parameter for clustering
            dbscan_min_samples: DBSCAN minimum samples parameter
            harris_delta: Harris detection neighborhood size parameter
            harris_k: Harris corner detection parameter
            harris_fraction: fraction of points to select as corners
            harris_cluster_threshold: minimum distance between corners
            harris_num_corners: maximum number of corners per cluster
        """
        # Load the transformed point cloud data
        data = np.loadtxt(transformed_file, delimiter=' ')
        points, colors = data[:, :3], data[:, 3:6]
        
        # Perform DBSCAN clustering to identify individual LEGO bricks
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points)
        labels = dbscan.labels_

        results = []
        geometries = []

        # Process each cluster (potential LEGO brick)
        for lbl in set(labels):
            if lbl == -1:  # Skip noise points
                continue
            m = (labels == lbl)
            lego_pts = points[m]
            lego_cols = colors[m]
            
            # Filter small clusters that are likely noise
            if len(lego_pts) < 500:
                continue
            
            print(f"Cluster {lbl}: {len(lego_pts)} points found.")

            # Add LEGO cluster as point cloud for visualization
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lego_pts)
            pcd.colors = o3d.utility.Vector3dVector(lego_cols)
            geometries.append(pcd)

            # Harris 3D keypoint detection with improved parameters for LEGO bricks
            if len(lego_pts) > 20:
                print(f"-> Detecting Harris corners for cluster {lbl}...")
                corners = Harris3DDetector.compute_harris_3d_corners(lego_pts, 
                                                     delta=harris_delta, 
                                                     harris_k=harris_k, 
                                                     fraction=harris_fraction, 
                                                     cluster_threshold=harris_cluster_threshold, 
                                                     num_corners=harris_num_corners)
                print(f"-> Found {len(corners)} corners.")

                # Visualize detected corners as orange spheres
                if len(corners) > 0:
                    for corner_pt in corners:
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
                        sphere.translate(corner_pt)
                        sphere.paint_uniform_color([1.0, 0.647, 0.0])  # Orange
                        geometries.append(sphere)

            # Orientation analysis using Principal Component Analysis
            pca = PCA(n_components=3).fit(lego_pts)
            center = np.mean(lego_pts, axis=0)
            angle_deg = ClusteringAnalysis.calculate_y_axis_angle_xy(pca.components_[1])
            dom_color = ClusteringAnalysis.get_dominant_color(lego_cols)
            results.append((center, dom_color, angle_deg))

        # Final interactive visualization of all clusters and detected corners
        if geometries:
            print("Visualizing results...")
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([axes] + geometries, 
                                            window_name="Filtered Clusters with Harris Corners")

        # Save analysis summary to file
        with open(summary_file, "w") as f:
            for center, color, angle in results:
                cx, cy, cz = center * 1000.0  # Convert to millimeters
                r_, g_, b_ = color
                f.write(f"{cx:.2f} {cy:.2f} {cz:.2f} {r_:.6f} {g_:.6f} {b_:.6f} {angle:.2f}\n")
        
        print(f"Saved clustering analysis summary to: {summary_file}")
