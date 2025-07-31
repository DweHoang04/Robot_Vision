import os
import socket
import time
import cv2
import numpy as np
import open3d as o3d
import struct
from datetime import datetime
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from pykinect2 import PyKinectRuntime, PyKinectV2
from pykinect2.PyKinectV2 import *

# Import our new modules
from bin_picking_filter import BinPickingFilter
from keypoint_detector import Harris3DDetector
from spin_image import SpinImageDescriptor
from pose_estimation import RANSACPoseEstimator
from template_manager import TemplateLibraryManager
from visualization import BinPickingVisualizer

class BinPickingSystem:

    def __init__(self, wdf_path, template_dir="C:\\Users\\FILAB\\Desktop\\DUY\\templates"):
        self.output_dir = "C:\\Binpicking"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize all the new modules
        self.filter = BinPickingFilter()
        self.harris_detector = Harris3DDetector(
            n_neighbours=3, 
            delta=0.025, 
            k=0.04, 
            fraction=0.1
        )
        self.spin_descriptor = SpinImageDescriptor(
            image_width=8,
            image_height=8,
            support_angle=60.0,
            bin_size=0.01,
            correlation_threshold=0.5
        )
        self.pose_estimator = RANSACPoseEstimator(
            epsilon1=0.005,
            epsilon2=0.01,
            max_iterations=1000,
            max_outer_iterations=50
        )
        self.template_manager = TemplateLibraryManager(template_dir)
        self.visualizer = BinPickingVisualizer(
            output_dir=os.path.join(self.output_dir, "visualizations")
        )
        
        print(f"BinPickingSystem initialized with template directory: {template_dir}")
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
        time.sleep(0.1)  # Rendering stabilization
        vis.capture_screen_image(image_path)
        vis.destroy_window()

    def calculate_y_axis_angle_xy(self, minor_axis):
        # Normalize to unit vector
        v2d = minor_axis[:2] / np.linalg.norm(minor_axis[:2])

        def clockwise_angle_from_y(vec):
            # Clockwise rotation angle from Y-axis (0~360)
            angle = np.degrees(np.arctan2(vec[0], vec[1])) % 360
            return angle

        angle1 = clockwise_angle_from_y(v2d)
        angle2 = clockwise_angle_from_y(-v2d)

        # Remove directionality for line → smaller rotation angle is the actual clockwise rotation of the line
        angle_deg = min(angle1, angle2)
        # Convert to complementary angle if exceeds 90°
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
            angle_deg = - angle_deg
        return angle_deg  # Finally return with negative sign

    def get_dominant_color(self, colors):
        c_int = (colors * 255).astype(int)
        counter = Counter(map(tuple, c_int))
        dom = max(counter, key=counter.get)
        return np.array(dom) / 255.0

    def cluster_and_save_summary(self, transformed_file, summary_file,
                                 dbscan_eps=0.01, dbscan_min_samples=10):
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
            print(f"Before : {len(lego_pts)}")
            # 🔍 Cluster size filtering
            if len(lego_pts) < 500:
                continue
            print(f"After : {len(lego_pts)}")

            # ⬇️ Generate geometry for visualization
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lego_pts)
            pcd.colors = o3d.utility.Vector3dVector(lego_cols)
            geometries.append(pcd)

            # ✅ PCA and center/direction calculation
            pca = PCA(n_components=3).fit(lego_pts)
            center = np.mean(lego_pts, axis=0)
            angle_deg = self.calculate_y_axis_angle_xy(pca.components_[1])
            dom_color = self.get_dominant_color(lego_cols)
            results.append((center, dom_color, angle_deg))

        # # 🧩 Visualization
        # if geometries:
        #     axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        #     o3d.visualization.draw_geometries([axes] + geometries, window_name="Filtered Clusters")

        # 💾 Save to file
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

    def run_pipeline(self):
        print("0")
        points, colors = self.capture_and_preprocess_kinect_data()
        if len(points) == 0:
            print("[PIPELINE] No valid points, terminating.")
            return
        print("1")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        transformed_file = os.path.join(self.output_dir, f"Transformed_ROI_point_cloud_{timestamp_str}.txt")
        summary_file = os.path.join(self.output_dir, f"Cluster_Summary_{timestamp_str}.txt")
        image_file = os.path.join(self.output_dir, f"PointCloud_Img_{timestamp_str}.png")
        print("2")
        self.save_transformed_point_cloud(points, colors, transformed_file)
        print("3")
        self.save_cloud_image(points, colors, image_file)
        print("4")
        self.cluster_and_save_summary(transformed_file, summary_file)
        print("5")
        # 🔻 Server transmission functionality is commented out
        # self.send_file_via_tcp(summary_file)
        # print("[PIPELINE] All processes completed.")
    
    # ========== ENHANCED BIN PICKING PIPELINE FOR STACKED LEGO BRICKS ==========
    
    def generate_template_library_from_stl(self, stl_file_path, template_name=None):
        """
        Generate template library from STL file
        
        Args:
            stl_file_path: Path to STL file (e.g., LegoBrick_4_2.stl)
            template_name: Name for the template
            
        Returns:
            template_info: Information about generated templates
        """
        print("="*60)
        print("TEMPLATE LIBRARY GENERATION")
        print("="*60)
        
        return self.template_manager.generate_templates_from_stl(stl_file_path, template_name)
    
    def enhanced_bin_picking_pipeline(self, roi_x=195, roi_y=50, roi_w=245, roi_h=300,
                                    template_name=None, enable_visualization=True):
        """
        Complete enhanced bin picking pipeline following the research paper
        
        Args:
            roi_x, roi_y, roi_w, roi_h: Region of interest parameters
            template_name: Name of template to match against
            enable_visualization: Whether to create visualizations
            
        Returns:
            result: Dictionary with pose estimation results
        """
        print("="*60)
        print("ENHANCED BIN PICKING PIPELINE")
        print("="*60)
        
        # Data acquisition and preprocessing
        print("\n[STEP] Data Acquisition and Preprocessing")
        points, colors = self.capture_and_preprocess_kinect_data(
            roi_x=roi_x, roi_y=roi_y, roi_w=roi_w, roi_h=roi_h
        )
        
        if len(points) == 0:
            print("ERROR: No valid points captured from Kinect")
            return {'success': False, 'error': 'No points captured'}
        
        print(f"Captured {len(points)} points from Kinect")
        
        # Step 0: Closest brick cluster filtering
        print("\n[STEP 0] Closest Brick Cluster Filtering")
        filter_result = self.filter.apply_bin_picking_filters(points, colors)
        
        if not filter_result['success'] or len(filter_result['points']) == 0:
            print("ERROR: No valid LEGO brick detected after filtering")
            return {'success': False, 'error': 'No valid brick found', 'filter_result': filter_result}
        
        filtered_points = filter_result['points']
        filtered_colors = filter_result['colors']
        dominant_color = filter_result['dominant_color']
        
        print(f"Filtered to {len(filtered_points)} points with {dominant_color} color")
        
        # Step 1: 3D Keypoint Detection
        print("\n[STEP 1] 3D Keypoint Detection")
        target_keypoints, keypoint_indices, harris_response = self.harris_detector.detect_keypoints(
            filtered_points, neighborhood_method='adaptive', selection_method='fraction'
        )
        
        if len(target_keypoints) == 0:
            print("ERROR: No keypoints detected")
            return {'success': False, 'error': 'No keypoints detected'}
        
        print(f"Detected {len(target_keypoints)} Harris 3D keypoints")
        
        # Load template library
        print("\n[STEP] Loading Template Library")
        templates = self.template_manager.load_template_library(template_name)
        
        if len(templates) == 0:
            print("ERROR: No templates found in library")
            return {'success': False, 'error': 'No templates found'}
        
        print(f"Loaded {len(templates)} templates from library")
        
        # Step 2: Spin Image Computation for target
        print("\n[STEP 2] Spin Image Computation - Target")
        target_spin_images, target_normals = self.spin_descriptor.compute_spin_images_for_keypoints(
            target_keypoints, filtered_points
        )
        
        if len(target_spin_images) == 0:
            print("ERROR: Failed to compute target spin images")
            return {'success': False, 'error': 'Spin image computation failed'}
        
        # Save target spin images for debugging
        if enable_visualization:
            target_spin_dir = os.path.join(self.output_dir, "spin_images", "target")
            os.makedirs(target_spin_dir, exist_ok=True)
            
            for i, spin_img in enumerate(target_spin_images):
                spin_path = os.path.join(target_spin_dir, f"target_spin_{i:03d}.png")
                self.spin_descriptor.save_spin_image(spin_img, spin_path)
            
            # Save spin image grid
            grid_path = os.path.join(target_spin_dir, "target_spin_grid.png")
            self.spin_descriptor.save_spin_image_grid(target_spin_images, grid_path, "Target Spin Images")
        
        # Step 3-6: Template matching with spin images
        print("\n[STEP 3-6] Template Matching with Spin Images")
        best_match = None
        best_score = -1
        
        for template in templates:
            try:
                print(f"\nProcessing template: {template['id']}")
                
                # Load template point cloud
                template_pcd = o3d.io.read_point_cloud(template['file'])
                template_points = np.asarray(template_pcd.points)
                
                if len(template_points) < 10:
                    print(f"  Skipping template {template['id']}: too few points")
                    continue
                
                # Detect keypoints in template
                template_keypoints, _, _ = self.harris_detector.detect_keypoints(
                    template_points, neighborhood_method='adaptive', selection_method='fraction'
                )
                
                if len(template_keypoints) == 0:
                    print(f"  Skipping template {template['id']}: no keypoints detected")
                    continue
                
                # Compute spin images for template
                template_spin_images, template_normals = self.spin_descriptor.compute_spin_images_for_keypoints(
                    template_keypoints, template_points
                )
                
                if len(template_spin_images) == 0:
                    print(f"  Skipping template {template['id']}: spin image computation failed")
                    continue
                
                # Save template spin images for debugging
                if enable_visualization:
                    template_spin_dir = os.path.join(self.output_dir, "spin_images", "templates", template['id'])
                    os.makedirs(template_spin_dir, exist_ok=True)
                    
                    for j, spin_img in enumerate(template_spin_images):
                        spin_path = os.path.join(template_spin_dir, f"template_spin_{j:03d}.png")
                        self.spin_descriptor.save_spin_image(spin_img, spin_path, cmap='plasma')
                    
                    # Save template spin image grid
                    grid_path = os.path.join(template_spin_dir, f"template_{template['id']}_spin_grid.png")
                    self.spin_descriptor.save_spin_image_grid(template_spin_images, grid_path, 
                                                           f"Template {template['id']} Spin Images", cmap='plasma')
                
                # Step 4: Spin Image Matching
                correspondences = self.spin_descriptor.find_correspondences(
                    target_spin_images, template_spin_images, target_keypoints, template_keypoints
                )
                
                if len(correspondences) < 3:
                    print(f"  Template {template['id']}: Insufficient correspondences ({len(correspondences)} < 3)")
                    continue
                
                print(f"  Template {template['id']}: Found {len(correspondences)} correspondences")
                
                # Step 5: RANSAC-based pose estimation
                R, T, inliers = self.pose_estimator.ransac_pose_estimation(
                    template_keypoints, target_keypoints, correspondences
                )
                
                if R is not None and T is not None:
                    # Evaluate pose
                    target_indices = [c[0] for c in correspondences]
                    template_indices = [c[1] for c in correspondences]
                    
                    Pe, Ce = self.pose_estimator.compute_pose_error(
                        template_keypoints[template_indices], 
                        target_keypoints[target_indices], 
                        R, T
                    )
                    
                    print(f"  Template {template['id']}: Pe={Pe:.6f}, Ce={Ce:.6f}, inliers={len(inliers)}")
                    
                    # Step 6: ICP Refinement (if pose is good enough)
                    if Pe < self.pose_estimator.epsilon1 and Ce < self.pose_estimator.epsilon2:
                        print(f"  Template {template['id']}: Applying ICP refinement...")
                        R_refined, T_refined = self.refine_pose_with_icp(template_points, filtered_points, R, T)
                        if R_refined is not None:
                            R, T = R_refined, T_refined
                            print(f"  Template {template['id']}: ICP refinement completed")
                    
                    # Calculate match score
                    match_score = len(inliers) / (1.0 + Pe + Ce)
                    
                    if match_score > best_score:
                        best_score = match_score
                        
                        # Convert rotation to Euler angles
                        euler_angles = self.pose_estimator.pose_to_euler_angles(R)
                        
                        best_match = {
                            'template_id': template['id'],
                            'rotation_matrix': R,
                            'translation': T,
                            'position': {'x': T[0], 'y': T[1], 'z': T[2]},
                            'orientation': {'rx': euler_angles[0], 'ry': euler_angles[1], 'rz': euler_angles[2]},
                            'match_score': match_score,
                            'Pe': Pe,
                            'Ce': Ce,
                            'num_correspondences': len(correspondences),
                            'num_inliers': len(inliers),
                            'dominant_color': dominant_color,
                            'method': 'enhanced_pipeline'
                        }
                        
                        print(f"  Template {template['id']}: NEW BEST MATCH (score={match_score:.3f})")
                
            except Exception as e:
                print(f"  Error processing template {template['id']}: {e}")
                continue
        
        # Final results
        if best_match:
            print(f"\n[RESULT] Best match found: {best_match['template_id']}")
            print(f"  Position: x={best_match['position']['x']:.3f}, y={best_match['position']['y']:.3f}, z={best_match['position']['z']:.3f}")
            print(f"  Orientation: rx={best_match['orientation']['rx']:.1f}°, ry={best_match['orientation']['ry']:.1f}°, rz={best_match['orientation']['rz']:.1f}°")
            print(f"  Color: {best_match['dominant_color']}")
            print(f"  Match score: {best_match['match_score']:.3f}")
            
            # Visualization
            if enable_visualization:
                print("\n[VISUALIZATION] Creating result visualization...")
                
                # Plot keypoints
                keypoints_plot_path = os.path.join(self.output_dir, "keypoints_detection.png")
                self.visualizer.plot_keypoints(filtered_points, target_keypoints, keypoints_plot_path)
                
                # Visualize detection result with bounding box and pose
                pose_info = self.visualizer.visualize_detection_result(
                    points, colors, filtered_points,
                    best_match['rotation_matrix'], best_match['translation'],
                    save_image=True, window_name="LEGO Brick Detection Result"
                )
                
                # Save pose summary
                pose_summary_path = self.visualizer.save_pose_summary([best_match])
                
                print(f"  Keypoints plot saved to: {keypoints_plot_path}")
                print(f"  Pose summary saved to: {pose_summary_path}")
            
            return {
                'success': True,
                'detection': best_match,
                'filter_result': filter_result,
                'num_keypoints': len(target_keypoints),
                'num_templates_processed': len(templates)
            }
        else:
            print("\n[RESULT] No suitable match found")
            return {
                'success': False,
                'error': 'No suitable template match found',
                'filter_result': filter_result,
                'num_keypoints': len(target_keypoints),
                'num_templates_processed': len(templates)
            }
    
    def refine_pose_with_icp(self, template_points, target_points, initial_R, initial_T, 
                           max_iterations=50, tolerance=1e-6):
        """
        Refine pose estimation using ICP (Iterative Closest Points)
        
        Args:
            template_points: Template point cloud
            target_points: Target point cloud
            initial_R: Initial rotation matrix
            initial_T: Initial translation vector
            max_iterations: Maximum ICP iterations
            tolerance: Convergence tolerance
            
        Returns:
            R_refined: Refined rotation matrix
            T_refined: Refined translation vector
        """
        try:
            # Create point clouds
            template_pcd = o3d.geometry.PointCloud()
            template_pcd.points = o3d.utility.Vector3dVector(template_points)
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_points)
            
            # Apply initial transformation to template
            initial_transform = np.eye(4)
            initial_transform[:3, :3] = initial_R
            initial_transform[:3, 3] = initial_T
            
            # Run ICP
            result = o3d.pipelines.registration.registration_icp(
                template_pcd, target_pcd, 0.02,  # max_correspondence_distance
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
            )
            
            if result.fitness > 0.1:  # Minimum fitness threshold
                refined_transform = result.transformation
                R_refined = refined_transform[:3, :3]
                T_refined = refined_transform[:3, 3]
                
                print(f"ICP refinement: fitness={result.fitness:.3f}, inlier_rmse={result.inlier_rmse:.6f}")
                return R_refined, T_refined
            else:
                print(f"ICP refinement failed: low fitness ({result.fitness:.3f})")
                return None, None
                
        except Exception as e:
            print(f"ICP refinement error: {e}")
            return None, None
        
if __name__ == "__main__":
    # Example usage of the enhanced bin picking system
    
    # Initialize system with template directory
    system = BinPickingSystem(wdf_path="", template_dir="C:\\Users\\FILAB\\Desktop\\DUY\\templates")
    
    # Check if we have the STL file for template generation
    stl_file = "C:\\Users\\FILAB\\Desktop\\DUY\\LegoBrick_4_2.stl"
    
    if os.path.exists(stl_file):
        print("STL file found. Generating template library...")
        
        # Generate template library from STL file (only need to do this once)
        template_info = system.generate_template_library_from_stl(
            stl_file_path=stl_file,
            template_name="LegoBrick_4_2"
        )
        
        print(f"Generated {template_info['num_templates']} templates")
        
        # Run the enhanced bin picking pipeline
        print("\nRunning enhanced bin picking pipeline...")
        result = system.enhanced_bin_picking_pipeline(
            template_name="LegoBrick_4_2",
            enable_visualization=True
        )
        
        if result['success']:
            detection = result['detection']
            print("\n" + "="*60)
            print("DETECTION SUCCESSFUL!")
            print("="*60)
            print(f"Template: {detection['template_id']}")
            print(f"Position (m): x={detection['position']['x']:.3f}, y={detection['position']['y']:.3f}, z={detection['position']['z']:.3f}")
            print(f"Orientation (deg): rx={detection['orientation']['rx']:.1f}, ry={detection['orientation']['ry']:.1f}, rz={detection['orientation']['rz']:.1f}")
            print(f"Color: {detection['dominant_color']}")
            print(f"Confidence: {detection['match_score']:.3f}")
        else:
            print("\n" + "="*60)
            print("DETECTION FAILED!")
            print("="*60)
            print(f"Error: {result['error']}")
            
    else:
        print(f"STL file '{stl_file}' not found. Please ensure the STL file is in the current directory.")
        print("Running basic pipeline instead...")
        
        # Fall back to original pipeline
        system.run_pipeline()