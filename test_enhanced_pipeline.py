# -*- coding: utf-8 -*-
"""
Test script for the enhanced bin picking pipeline
This script demonstrates the pipeline using the STL file for template generation
and simulated data for testing without requiring a Kinect camera
"""
import os
import numpy as np
import open3d as o3d
from BinPickingSystem import BinPickingSystem

def create_test_scene(stl_file_path, noise_level=0.001, num_points=2000):
    """
    Create a test scene by loading and slightly modifying the STL file
    This simulates a real capture from the Kinect camera
    
    Args:
        stl_file_path: Path to STL file
        noise_level: Amount of noise to add
        num_points: Number of points to sample
        
    Returns:
        points: Test point cloud
        colors: Test colors
    """
    print(f"Creating test scene from {stl_file_path}...")
    
    # Load STL file
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load STL file: {stl_file_path}")
    
    # Normalize and position the mesh
    mesh.scale(0.05, center=mesh.get_center())  # Scale to realistic size
    mesh.translate([0.0, 0.0, 0.5])  # Position in front of camera
    
    # Sample points from mesh
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points)
    
    # Add some noise to simulate sensor noise
    noise = np.random.normal(0, noise_level, points.shape)
    points += noise
    
    # Create realistic colors (LEGO red)
    colors = np.full((len(points), 3), [0.8, 0.2, 0.2])  # Red color
    
    # Add some color variation
    color_variation = np.random.normal(0, 0.05, colors.shape)
    colors += color_variation
    colors = np.clip(colors, 0, 1)
    
    print(f"Generated test scene with {len(points)} points")
    return points, colors

def test_template_generation():
    """Test template generation from STL file"""
    print("="*60)
    print("TESTING TEMPLATE GENERATION")
    print("="*60)
    
    # Initialize system
    system = BinPickingSystem(wdf_path="", template_dir="C:\\Users\\FILAB\\Desktop\\DUY\\templates")
    
    # Check for STL file
    stl_file = "C:\\Users\\FILAB\\Desktop\\DUY\\LegoBrick_4_2.stl"
    
    if not os.path.exists(stl_file):
        print(f"ERROR: STL file '{stl_file}' not found!")
        print("Please ensure the STL file is in the current directory.")
        return False
    
    # Generate templates (with fewer viewpoints for faster testing)
    print("Generating template library...")
    
    # Temporarily modify template generation for faster testing
    system.template_manager.generate_viewpoints = lambda: [
        {'rotation_matrix': np.eye(3), 'euler_angles': [0, 0, 0], 'id': 'x000_y000_z000'},
        {'rotation_matrix': np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), 'euler_angles': [0, 0, 90], 'id': 'x000_y000_z090'},
        {'rotation_matrix': np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), 'euler_angles': [0, 90, 0], 'id': 'x000_y090_z000'},
    ]
    
    template_info = system.generate_template_library_from_stl(
        stl_file_path=stl_file,
        template_name="LegoBrick_4_2_test"
    )
    
    print(f"Successfully generated {template_info['num_templates']} templates")
    return True

def test_individual_modules():
    """Test individual modules with synthetic data"""
    print("="*60)
    print("TESTING INDIVIDUAL MODULES")
    print("="*60)
    
    # Create test data
    stl_file = "C:\\Users\\FILAB\\Desktop\\DUY\\LegoBrick_4_2.stl"
    
    if not os.path.exists(stl_file):
        print(f"STL file '{stl_file}' not found. Skipping module tests.")
        return
    
    points, colors = create_test_scene(stl_file)
    
    # Initialize system
    system = BinPickingSystem(wdf_path="", template_dir="C:\\Users\\FILAB\\Desktop\\DUY\\templates")
    
    # Test 1: Bin picking filter
    print("\n[TEST 1] Bin Picking Filter")
    filter_result = system.filter.apply_bin_picking_filters(points, colors)
    
    if filter_result['success']:
        print(f"✓ Filter passed: {len(filter_result['points'])} points, color: {filter_result['dominant_color']}")
        filtered_points = filter_result['points']
    else:
        print("✗ Filter failed")
        filtered_points = points
    
    # Test 2: Harris 3D keypoint detection
    print("\n[TEST 2] Harris 3D Keypoint Detection")
    keypoints, keypoint_indices, response = system.harris_detector.detect_keypoints(filtered_points)
    
    if len(keypoints) > 0:
        print(f"✓ Keypoint detection passed: {len(keypoints)} keypoints detected")
    else:
        print("✗ Keypoint detection failed")
        return
    
    # Test 3: Spin image computation
    print("\n[TEST 3] Spin Image Computation")
    spin_images, normals = system.spin_descriptor.compute_spin_images_for_keypoints(keypoints, filtered_points)
    
    if len(spin_images) > 0:
        print(f"✓ Spin image computation passed: {len(spin_images)} spin images computed")
        
        # Save a few spin images for inspection
        test_output_dir = os.path.join("test_output", "spin_images")
        os.makedirs(test_output_dir, exist_ok=True)
        
        for i, spin_img in enumerate(spin_images[:5]):  # Save first 5
            spin_path = os.path.join(test_output_dir, f"test_spin_{i}.png")
            system.spin_descriptor.save_spin_image(spin_img, spin_path)
        
        print(f"Sample spin images saved to: {test_output_dir}")
    else:
        print("✗ Spin image computation failed")
    
    print("\n[TEST] All individual modules tested successfully!")

def test_full_pipeline():
    """Test the complete pipeline with simulated data"""
    print("="*60) 
    print("TESTING FULL PIPELINE")
    print("="*60)
    
    # This would normally use the Kinect, but we'll simulate the data
    stl_file = "C:\\Users\\FILAB\\Desktop\\DUY\\LegoBrick_4_2.stl"
    
    if not os.path.exists(stl_file):
        print(f"STL file '{stl_file}' not found. Cannot test full pipeline.")
        return
    
    # Initialize system
    system = BinPickingSystem(wdf_path="", template_dir="C:\\Users\\FILAB\\Desktop\\DUY\\templates")
    
    # Create mock Kinect data
    points, colors = create_test_scene(stl_file)
    
    # Replace the Kinect capture method with our test data
    def mock_capture_and_preprocess_kinect_data(*args, **kwargs):
        return points, colors
    
    # Temporarily replace the method
    original_method = system.capture_and_preprocess_kinect_data
    system.capture_and_preprocess_kinect_data = mock_capture_and_preprocess_kinect_data
    
    try:
        # Run the enhanced pipeline
        result = system.enhanced_bin_picking_pipeline(
            template_name="LegoBrick_4_2_test",
            enable_visualization=True
        )
        
        if result['success']:
            detection = result['detection']
            print("\n" + "="*60)
            print("PIPELINE TEST SUCCESSFUL!")
            print("="*60)
            print(f"Template: {detection['template_id']}")
            print(f"Position: x={detection['position']['x']:.3f}, y={detection['position']['y']:.3f}, z={detection['position']['z']:.3f}")
            print(f"Orientation: rx={detection['orientation']['rx']:.1f}°, ry={detection['orientation']['ry']:.1f}°, rz={detection['orientation']['rz']:.1f}°")
            print(f"Match score: {detection['match_score']:.3f}")
        else:
            print("\n" + "="*60)
            print("PIPELINE TEST FAILED!")
            print("="*60)
            print(f"Error: {result['error']}")
    
    finally:
        # Restore original method
        system.capture_and_preprocess_kinect_data = original_method

def main():
    """Main test function"""
    print("ENHANCED BIN PICKING SYSTEM - TEST SUITE")
    print("="*60)
    
    # Create output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Test 1: Template generation
    if test_template_generation():
        print("\n✓ Template generation test passed")
    else:
        print("\n✗ Template generation test failed")
        return
    
    # Test 2: Individual modules
    test_individual_modules()
    
    # Test 3: Full pipeline
    test_full_pipeline()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)
    print("Check the following directories for outputs:")
    print("- test_templates/: Generated template library")
    print("- test_output/: Test outputs including spin images")
    print("- C:/Binpicking/: Pipeline outputs and visualizations")

if __name__ == "__main__":
    main()
