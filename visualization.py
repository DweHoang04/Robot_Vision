# -*- coding: utf-8 -*-
"""
Visualization Module
Handles visualization of results including bounding boxes and pose estimation
"""
import numpy as np
import open3d as o3d
import cv2
import os
import time

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Some visualization features will be disabled.")

class BinPickingVisualizer:
    def __init__(self, output_dir="visualization_output"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_bounding_box(self, points, color=[1, 0, 0]):
        """
        Create a bounding box around points
        
        Args:
            points: Point cloud (N x 3)
            color: RGB color for bounding box [r, g, b]
            
        Returns:
            bbox: Open3D LineSet representing bounding box
        """
        if len(points) == 0:
            return None
        
        # Calculate bounding box
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        
        # Define 8 corners of bounding box
        corners = np.array([
            [min_bound[0], min_bound[1], min_bound[2]],  # 0
            [max_bound[0], min_bound[1], min_bound[2]],  # 1
            [max_bound[0], max_bound[1], min_bound[2]],  # 2
            [min_bound[0], max_bound[1], min_bound[2]],  # 3
            [min_bound[0], min_bound[1], max_bound[2]],  # 4
            [max_bound[0], min_bound[1], max_bound[2]],  # 5
            [max_bound[0], max_bound[1], max_bound[2]],  # 6
            [min_bound[0], max_bound[1], max_bound[2]]   # 7
        ])
        
        # Define edges connecting corners
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ])
        
        # Create LineSet
        bbox = o3d.geometry.LineSet()
        bbox.points = o3d.utility.Vector3dVector(corners)
        bbox.lines = o3d.utility.Vector2iVector(edges)
        bbox.colors = o3d.utility.Vector3dVector([color for _ in range(len(edges))])
        
        return bbox
    
    def create_coordinate_frame(self, R, T, size=0.05):
        """
        Create coordinate frame showing pose
        
        Args:
            R: Rotation matrix (3 x 3)
            T: Translation vector (3,)
            size: Size of coordinate frame
            
        Returns:
            coord_frame: Open3D TriangleMesh coordinate frame
        """
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        
        # Apply transformation
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = T
        
        coord_frame.transform(transformation)
        return coord_frame
    
    def create_pose_text(self, position, orientation_euler, scale=0.01):
        """
        Create text annotations for pose information
        
        Args:
            position: 3D position (x, y, z)
            orientation_euler: Euler angles (rx, ry, rz) in degrees
            scale: Text scale
            
        Returns:
            pose_info: Dictionary with pose information
        """
        pose_info = {
            'position': {
                'x': float(position[0]),
                'y': float(position[1]), 
                'z': float(position[2])
            },
            'orientation': {
                'rx': float(orientation_euler[0]),
                'ry': float(orientation_euler[1]),
                'rz': float(orientation_euler[2])
            }
        }
        return pose_info
    
    def visualize_detection_result(self, scene_points, scene_colors, detected_points, 
                                 R=None, T=None, save_image=True, window_name="Detection Result"):
        """
        Visualize detection results with bounding box and pose
        
        Args:
            scene_points: Full scene point cloud
            scene_colors: Scene colors
            detected_points: Detected object points
            R: Rotation matrix (optional)
            T: Translation vector (optional)
            save_image: Whether to save visualization image
            window_name: Window title
            
        Returns:
            pose_info: Dictionary with pose information
        """
        print("Creating detection visualization...")
        
        geometries = []
        
        # Add scene point cloud (gray)
        if len(scene_points) > 0:
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
            if len(scene_colors) == len(scene_points):
                scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors * 0.5)  # Dim the scene
            else:
                scene_pcd.paint_uniform_color([0.7, 0.7, 0.7])
            geometries.append(scene_pcd)
        
        # Add detected object points (original color)
        if len(detected_points) > 0:
            detected_pcd = o3d.geometry.PointCloud()
            detected_pcd.points = o3d.utility.Vector3dVector(detected_points)
            detected_pcd.paint_uniform_color([0, 1, 0])  # Green for detected object
            geometries.append(detected_pcd)
            
            # Add red bounding box around detected object
            bbox = self.create_bounding_box(detected_points, color=[1, 0, 0])
            if bbox is not None:
                geometries.append(bbox)
        
        # Add coordinate frame showing pose
        pose_info = None
        if R is not None and T is not None:
            coord_frame = self.create_coordinate_frame(R, T, size=0.05)
            geometries.append(coord_frame)
            
            # Convert rotation matrix to Euler angles
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_matrix(R)
            euler_angles = rotation.as_euler('xyz', degrees=True)
            
            pose_info = self.create_pose_text(T, euler_angles)
            
            print(f"Detected pose:")
            print(f"  Position: x={T[0]:.3f}, y={T[1]:.3f}, z={T[2]:.3f}")
            print(f"  Orientation: rx={euler_angles[0]:.1f}°, ry={euler_angles[1]:.1f}°, rz={euler_angles[2]:.1f}°")
        
        # Add world coordinate frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(world_frame)
        
        # Visualize
        if save_image:
            # Save visualization as image
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name, visible=False)
            
            for geom in geometries:
                vis.add_geometry(geom)
            
            vis.update_renderer()
            
            # Save screenshot
            timestamp = str(int(time.time()))
            image_path = os.path.join(self.output_dir, f"detection_result_{timestamp}.png")
            vis.capture_screen_image(image_path)
            vis.destroy_window()
            
            print(f"Visualization saved to: {image_path}")
        else:
            # Interactive visualization
            o3d.visualization.draw_geometries(geometries, window_name=window_name)
        
        return pose_info
    
    def plot_keypoints(self, points, keypoints, save_path=None):
        """
        Plot point cloud with highlighted keypoints
        
        Args:
            points: All points (N x 3)
            keypoints: Keypoint positions (M x 3)
            save_path: Path to save plot (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create keypoints plot.")
            return
            
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c='lightblue', s=1, alpha=0.3, label='All points')
        
        # Plot keypoints
        if len(keypoints) > 0:
            ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
                      c='red', s=50, alpha=1.0, label=f'Keypoints ({len(keypoints)})')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('3D Harris Keypoints Detection')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Keypoints plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_correspondences(self, target_keypoints, template_keypoints, 
                                correspondences, save_path=None):
        """
        Visualize point correspondences between target and template
        
        Args:
            target_keypoints: Target keypoint positions
            template_keypoints: Template keypoint positions  
            correspondences: List of (target_idx, template_idx, score) tuples
            save_path: Path to save plot (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create correspondences plot.")
            return
            
        fig = plt.figure(figsize=(15, 6))
        
        # Plot target keypoints
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(target_keypoints[:, 0], target_keypoints[:, 1], target_keypoints[:, 2],
                   c='blue', s=50, label='Target keypoints')
        
        # Highlight corresponding points
        if len(correspondences) > 0:
            target_indices = [c[0] for c in correspondences]
            corr_target = target_keypoints[target_indices]
            ax1.scatter(corr_target[:, 0], corr_target[:, 1], corr_target[:, 2],
                       c='red', s=100, marker='x', label='Correspondences')
        
        ax1.set_title('Target Keypoints')
        ax1.legend()
        
        # Plot template keypoints
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(template_keypoints[:, 0], template_keypoints[:, 1], template_keypoints[:, 2],
                   c='green', s=50, label='Template keypoints')
        
        # Highlight corresponding points
        if len(correspondences) > 0:
            template_indices = [c[1] for c in correspondences]
            corr_template = template_keypoints[template_indices]
            ax2.scatter(corr_template[:, 0], corr_template[:, 1], corr_template[:, 2],
                       c='red', s=100, marker='x', label='Correspondences')
        
        ax2.set_title('Template Keypoints')
        ax2.legend()
        
        plt.suptitle(f'Point Correspondences ({len(correspondences)} matches)')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Correspondences plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_pose_summary(self, pose_results, save_path=None):
        """
        Save pose estimation summary to file
        
        Args:
            pose_results: List of pose estimation results
            save_path: Path to save summary (optional)
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "pose_summary.txt")
        
        with open(save_path, 'w') as f:
            f.write("LEGO BRICK POSE ESTIMATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for i, result in enumerate(pose_results):
                f.write(f"Detection {i+1}:\n")
                f.write(f"  Position (mm): x={result['position']['x']*1000:.1f}, "
                       f"y={result['position']['y']*1000:.1f}, "
                       f"z={result['position']['z']*1000:.1f}\n")
                f.write(f"  Orientation (deg): rx={result['orientation']['rx']:.1f}, "
                       f"ry={result['orientation']['ry']:.1f}, "
                       f"rz={result['orientation']['rz']:.1f}\n")
                
                if 'confidence' in result:
                    f.write(f"  Confidence: {result['confidence']:.3f}\n")
                
                f.write("\n")
        
        print(f"Pose summary saved to: {save_path}")
        return save_path
