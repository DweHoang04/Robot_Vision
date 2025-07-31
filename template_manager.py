# -*- coding: utf-8 -*-
"""
Template Library Manager
Handles template generation from CAD models and template loading
"""
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import json
import glob
import matplotlib.pyplot as plt

class TemplateLibraryManager:
    def __init__(self, template_dir="templates"):
        """
        Initialize template library manager
        
        Args:
            template_dir: Directory to store template library
        """
        self.template_dir = template_dir
        os.makedirs(template_dir, exist_ok=True)
    
    def generate_viewpoints(self, x_angles=None, y_angles=None, z_angles=None):
        """
        Generate viewpoint rotations for template generation
        Based on paper: "x-axis from 0° to 180° and y-axis from 0° to 360°"
        
        Args:
            x_angles: X-axis rotation angles in degrees
            y_angles: Y-axis rotation angles in degrees  
            z_angles: Z-axis rotation angles in degrees
            
        Returns:
            viewpoints: List of rotation matrices
        """
        if x_angles is None:
            x_angles = np.arange(0, 181, 20)  # 0° to 180° with 20° steps
        if y_angles is None:
            y_angles = np.arange(0, 360, 20)  # 0° to 360° with 20° steps
        if z_angles is None:
            z_angles = [0]  # Usually keep Z fixed
        
        viewpoints = []
        for x_angle in x_angles:
            for y_angle in y_angles:
                for z_angle in z_angles:
                    # Create rotation matrix
                    rotation = Rotation.from_euler('xyz', [x_angle, y_angle, z_angle], degrees=True)
                    viewpoints.append({
                        'rotation_matrix': rotation.as_matrix(),
                        'euler_angles': [x_angle, y_angle, z_angle],
                        'id': f"x{x_angle:03d}_y{y_angle:03d}_z{z_angle:03d}"
                    })
        
        print(f"Generated {len(viewpoints)} viewpoints")
        return viewpoints
    
    def load_stl_model(self, stl_file_path):
        """
        Load STL model and convert to point cloud
        
        Args:
            stl_file_path: Path to STL file
            
        Returns:
            mesh: Open3D mesh object
            point_cloud: Open3D point cloud object
        """
        print(f"Loading STL model: {stl_file_path}")
        
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(stl_file_path)
        
        if len(mesh.vertices) == 0:
            raise ValueError(f"Failed to load STL file: {stl_file_path}")
        
        # Normalize mesh size
        mesh.scale(1.0 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), 
                   center=mesh.get_center())
        
        # Sample points from mesh surface
        point_cloud = mesh.sample_points_uniformly(number_of_points=5000)
        
        print(f"Loaded mesh with {len(mesh.vertices)} vertices, "
              f"sampled {len(point_cloud.points)} points")
        
        return mesh, point_cloud
    
    def apply_rotation_to_pointcloud(self, point_cloud, rotation_matrix):
        """Apply rotation to point cloud"""
        rotated_pcd = o3d.geometry.PointCloud(point_cloud)
        rotated_pcd.rotate(rotation_matrix, center=[0, 0, 0])
        return rotated_pcd
    
    def simulate_depth_view(self, point_cloud, viewpoint_rotation):
        """
        Simulate depth camera view of the object
        This simulates the partial view that would be captured by a depth camera
        
        Args:
            point_cloud: Open3D point cloud
            viewpoint_rotation: Rotation matrix for viewpoint
            
        Returns:
            visible_points: Points visible from the viewpoint
        """
        # Apply inverse rotation to simulate camera viewpoint
        rotated_pcd = self.apply_rotation_to_pointcloud(point_cloud, viewpoint_rotation.T)
        points = np.asarray(rotated_pcd.points)
        
        if len(points) == 0:
            return np.array([])
        
        # Simulate depth camera visibility (keep points facing camera)
        # Points with positive Z are facing the camera
        visible_mask = points[:, 2] > 0
        visible_points = points[visible_mask]
        
        # Add some random sampling to simulate partial occlusion
        if len(visible_points) > 1000:
            indices = np.random.choice(len(visible_points), 1000, replace=False)
            visible_points = visible_points[indices]
        
        return visible_points
    
    def generate_templates_from_stl(self, stl_file_path, template_name=None, 
                                  viewpoint_subset=None):
        """
        Generate template library from STL file
        
        Args:
            stl_file_path: Path to STL file
            template_name: Name for template (derived from filename if None)
            viewpoint_subset: Subset of viewpoints to use (use all if None)
            
        Returns:
            template_info: Dictionary with template generation info
        """
        if template_name is None:
            template_name = os.path.splitext(os.path.basename(stl_file_path))[0]
        
        print(f"Generating templates for {template_name}...")
        
        # Load STL model
        mesh, point_cloud = self.load_stl_model(stl_file_path)
        
        # Generate viewpoints
        viewpoints = self.generate_viewpoints()
        
        if viewpoint_subset is not None:
            viewpoints = viewpoints[:viewpoint_subset]
        
        # Create template directory
        template_subdir = os.path.join(self.template_dir, template_name)
        os.makedirs(template_subdir, exist_ok=True)
        
        # Generate templates for each viewpoint
        template_files = []
        for i, viewpoint in enumerate(viewpoints):
            try:
                # Simulate depth view
                visible_points = self.simulate_depth_view(point_cloud, viewpoint['rotation_matrix'])
                
                if len(visible_points) < 50:  # Skip viewpoints with too few points
                    continue
                
                # Create point cloud object
                template_pcd = o3d.geometry.PointCloud()
                template_pcd.points = o3d.utility.Vector3dVector(visible_points)
                
                # Add some noise to simulate real sensor data
                noise = np.random.normal(0, 0.001, visible_points.shape)
                noisy_points = visible_points + noise
                template_pcd.points = o3d.utility.Vector3dVector(noisy_points)
                
                # Save template
                template_filename = f"{template_name}_{viewpoint['id']}.pcd"
                template_filepath = os.path.join(template_subdir, template_filename)
                o3d.io.write_point_cloud(template_filepath, template_pcd)
                
                template_files.append({
                    'file': template_filepath,
                    'id': f"{template_name}_{viewpoint['id']}",
                    'viewpoint': viewpoint,
                    'num_points': len(visible_points)
                })
                
                if (i + 1) % 50 == 0:
                    print(f"  Generated {i + 1}/{len(viewpoints)} templates...")
                    
            except Exception as e:
                print(f"Error generating template for viewpoint {i}: {e}")
                continue
        
        # Save template metadata
        metadata = {
            'template_name': template_name,
            'source_stl': stl_file_path,
            'num_templates': len(template_files),
            'templates': template_files
        }
        
        metadata_file = os.path.join(template_subdir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        print(f"Generated {len(template_files)} templates for {template_name}")
        print(f"Templates saved to: {template_subdir}")
        
        return metadata
    
    def load_template_library(self, template_name=None):
        """
        Load template library from disk
        
        Args:
            template_name: Specific template to load (load all if None)
            
        Returns:
            templates: List of template dictionaries
        """
        templates = []
        
        if template_name is not None:
            # Load specific template
            template_subdir = os.path.join(self.template_dir, template_name)
            if os.path.exists(template_subdir):
                templates.extend(self._load_template_from_directory(template_subdir))
        else:
            # Load all templates
            for subdir in os.listdir(self.template_dir):
                template_subdir = os.path.join(self.template_dir, subdir)
                if os.path.isdir(template_subdir):
                    templates.extend(self._load_template_from_directory(template_subdir))
        
        print(f"Loaded {len(templates)} templates from library")
        return templates
    
    def _load_template_from_directory(self, template_dir):
        """Load templates from a specific directory"""
        templates = []
        
        # Try to load metadata first
        metadata_file = os.path.join(template_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                for template_info in metadata['templates']:
                    if os.path.exists(template_info['file']):
                        templates.append(template_info)
                
                return templates
            except Exception as e:
                print(f"Error loading metadata from {metadata_file}: {e}")
        
        # Fallback: scan for PCD files directly
        pcd_files = glob.glob(os.path.join(template_dir, "*.pcd"))
        for pcd_file in pcd_files:
            template_id = os.path.splitext(os.path.basename(pcd_file))[0]
            templates.append({
                'file': pcd_file,
                'id': template_id,
                'viewpoint': None,
                'num_points': None
            })
        
        return templates
    
    def visualize_templates(self, template_name, max_templates=10):
        """
        Visualize templates from the library
        
        Args:
            template_name: Name of template to visualize
            max_templates: Maximum number of templates to show
        """
        templates = self.load_template_library(template_name)
        
        if len(templates) == 0:
            print(f"No templates found for {template_name}")
            return
        
        # Load and visualize templates
        geometries = []
        templates_to_show = templates[:max_templates]
        
        for i, template in enumerate(templates_to_show):
            try:
                pcd = o3d.io.read_point_cloud(template['file'])
                if len(pcd.points) > 0:
                    # Color each template differently
                    color = plt.cm.tab10(i % 10)[:3]
                    pcd.paint_uniform_color(color)
                    
                    # Offset templates for visualization
                    offset = np.array([i * 0.1, 0, 0])
                    pcd.translate(offset)
                    
                    geometries.append(pcd)
            except Exception as e:
                print(f"Error loading template {template['file']}: {e}")
        
        if geometries:
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            geometries.append(coord_frame)
            
            print(f"Visualizing {len(geometries)-1} templates for {template_name}")
            o3d.visualization.draw_geometries(geometries, 
                                            window_name=f"Templates: {template_name}")
        else:
            print("No valid templates to visualize")
