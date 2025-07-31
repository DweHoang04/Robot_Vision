# Enhanced 3D Bin Picking System for LEGO Duplo Bricks

This project implements a complete 3D bin picking pipeline for detecting and measuring the coordinates (x,y,z) and rotation (rx,ry,rz) of symmetrical LEGO Duplo 2x3 bricks using depth and RGB data from a Kinect camera.

## Overview

The system follows the pipeline described in the research paper "3D object detection and pose estimation from depth image for robotic bin picking" with modifications specific to LEGO brick detection.

### Pipeline Steps

1. **Step 0**: Closest Brick Cluster Filtering
2. **Step 1**: 3D Harris Keypoint Detection  
3. **Step 2**: Spin Image Feature Computation
4. **Step 3**: Spin Image Matching
5. **Step 4**: RANSAC-based Pose Estimation
6. **Step 5**: ICP Refinement
7. **Step 6**: Visualization and Results

## System Architecture

### Core Modules

- **`BinPickingSystem.py`**: Main system class integrating all components
- **`bin_picking_filter.py`**: Step 0 - Closest brick cluster filtering with LEGO color detection
- **`keypoint_detector.py`**: Step 1 - 3D Harris corner detection implementation
- **`spin_image.py`**: Step 2-3 - Spin image computation and matching
- **`pose_estimation.py`**: Step 4-5 - RANSAC pose estimation and ICP refinement
- **`template_manager.py`**: Template library generation from STL files
- **`visualization.py`**: Result visualization with bounding boxes and pose display

### Supporting Files

- **`neighborhoords.py`**: Neighborhood computation for Harris detection
- **`transformation.py`**: Point cloud transformation utilities
- **`utils/ply.py`**: PLY file I/O compatibility layer

## Installation

### Requirements

```bash
pip install numpy scipy scikit-learn open3d opencv-python matplotlib
pip install pykinect2  # For Kinect support
```

### Hardware Requirements

- Microsoft Kinect v2 camera
- Windows environment (for Kinect SDK)
- LEGO Duplo 2x3 brick STL file (`LegoBrick_4_2.stl`)

## Usage

### 1. Template Generation

First, generate the template library from the STL file:

```python
from BinPickingSystem import BinPickingSystem

# Initialize system
system = BinPickingSystem(wdf_path="", template_dir="templates")

# Generate templates from STL file
template_info = system.generate_template_library_from_stl(
    stl_file_path="LegoBrick_4_2.stl",
    template_name="LegoBrick_4_2"
)
```

### 2. Real-time Detection

Run the complete bin picking pipeline:

```python
# Run enhanced pipeline with Kinect data
result = system.enhanced_bin_picking_pipeline(
    template_name="LegoBrick_4_2",
    enable_visualization=True
)

if result['success']:
    detection = result['detection']
    print(f"Position: x={detection['position']['x']:.3f}m")
    print(f"Orientation: rx={detection['orientation']['rx']:.1f}°")
    print(f"Color: {detection['dominant_color']}")
```

### 3. Testing Without Kinect

Use the test script to verify the system without hardware:

```bash
python test_enhanced_pipeline.py
```

## Step-by-Step Pipeline Description

### Step 0: Closest Brick Cluster Filtering

The filtering process follows a three-stage approach to isolate the target brick:

1. **Closest Cluster Detection**: 
   - Applies DBSCAN clustering to the entire input point cloud (no color pre-filtering)
   - Identifies the cluster containing the point closest to the camera (minimum Z-distance)
   - Ensures the truly closest physical cluster is selected, not just the closest colored cluster

2. **Reference Color Extraction**:
   - Analyzes the N closest points within the identified cluster (default: 30% closest points)
   - Classifies their colors using HSV-based LEGO color detection
   - Determines the dominant color among these closest points as the "reference color"

3. **Color-based Cluster Filtering**:
   - Filters the entire closest cluster to keep only points matching the reference color
   - Removes outlier colors and noise within the cluster
   - Outputs a clean point cloud of the target brick with consistent color

**LEGO Color Detection**: Supports red, orange, yellow, light green, light blue, and dark blue using calibrated HSV ranges with special handling for red hue wrap-around.

**Output**: Clean, color-consistent point cloud containing only the target brick closest to the camera.

### Step 1: 3D Harris Keypoint Detection

- **Neighborhood Computation**: Uses adaptive k-ring Delaunay triangulation
- **Surface Fitting**: Applies PCA for best-fit plane computation
- **Harris Response**: Computes 3D Harris corner response using quadratic surface fitting
- **Keypoint Selection**: Selects top keypoints using fraction method
- **Output**: 3D keypoint coordinates

### Step 2: Spin Image Computation

- **Surface Normals**: Computes surface normals using local PCA
- **Spin Image Generation**: Creates 2D spin images using cylindrical coordinates:
  - α = perpendicular distance to spin axis
  - β = signed distance along spin axis
- **Correlation**: Computes correlation coefficients between spin images
- **Output**: Spin image descriptors for each keypoint

### Step 3: Template Matching

- **Template Loading**: Loads pre-generated templates from library
- **Correspondence Finding**: Matches target and template spin images
- **Correlation Threshold**: Filters matches above correlation threshold
- **Output**: Point correspondences between target and template

### Step 4: RANSAC Pose Estimation

- **Random Sampling**: Selects 3 correspondence pairs randomly
- **Transformation Estimation**: Computes rigid transformation (R, T)
- **Pose Evaluation**: 
  - Pe = mean distance error (equation 6)
  - Ce = centroid distance error (equation 7)
- **Iterative Refinement**: Repeats until convergence criteria met
- **Output**: Best rotation matrix R and translation vector T

### Step 5: ICP Refinement

- **Initial Alignment**: Uses RANSAC result as starting point
- **Iterative Closest Points**: Refines pose using point-to-point ICP
- **Convergence**: Stops when fitness threshold reached
- **Output**: Refined pose estimation

### Step 6: Visualization

- **Bounding Box**: Creates red bounding box around detected brick
- **Coordinate Frame**: Shows pose as 3D coordinate system
- **Pose Information**: Displays position (x,y,z) and orientation (rx,ry,rz)
- **Output**: Visual confirmation and pose summary

## Key Features

### LEGO-Specific Adaptations

1. **Color-based Filtering**: HSV color space filtering for accurate LEGO color detection
2. **Symmetrical Object Handling**: Fraction-based keypoint selection suitable for symmetric bricks
3. **Multi-view Templates**: Comprehensive template library covering various orientations
4. **Robust Matching**: RANSAC + ICP combination for accurate pose estimation

### Advanced Capabilities

1. **Adaptive Neighborhoods**: Dynamic neighborhood size based on point density
2. **Spin Image Visualization**: PNG output of spin images for debugging
3. **Multiple Template Support**: Library-based approach for different brick types
4. **Comprehensive Logging**: Detailed progress and error reporting

## Output Files

The system generates several output files:

### Template Library
- `templates/LegoBrick_4_2/`: Template point clouds and metadata
- `templates/LegoBrick_4_2/metadata.json`: Template information

### Detection Results
- `C:/Binpicking/visualizations/`: Detection visualizations
- `C:/Binpicking/spin_images/`: Spin image visualizations
- `C:/Binpicking/pose_summary.txt`: Pose estimation results

### Debug Information
- `C:/Binpicking/keypoints_detection.png`: Keypoint visualization
- `C:/Binpicking/spin_images/target/`: Target spin images
- `C:/Binpicking/spin_images/templates/`: Template spin images

## Configuration Parameters

### Harris Detector
- `n_neighbours=3`: Number of neighbors for fixed methods
- `delta=0.025`: Adaptive neighborhood parameter
- `k=0.04`: Harris response parameter
- `fraction=0.1`: Fraction of points to select as keypoints

### Spin Images
- `image_width=8`: Spin image width in bins
- `image_height=8`: Spin image height in bins
- `support_angle=60.0`: Maximum angle for point inclusion (degrees)
- `correlation_threshold=0.5`: Minimum correlation for matching

### RANSAC
- `epsilon1=0.005`: Distance threshold for pose evaluation
- `epsilon2=0.01`: Centroid distance threshold
- `max_iterations=1000`: Maximum RANSAC iterations
- `max_outer_iterations=50`: Maximum outer loop iterations

## Troubleshooting

### Common Issues

1. **No keypoints detected**: Increase fraction parameter or adjust delta
2. **Poor spin image matching**: Lower correlation threshold or increase image resolution
3. **RANSAC failure**: Increase iteration limits or relax epsilon thresholds
4. **Template loading errors**: Check template directory and file permissions

### Performance Optimization

1. **Reduce template count**: Use fewer viewpoints for faster processing
2. **Adjust neighborhood size**: Smaller neighborhoods for faster Harris detection
3. **Lower spin image resolution**: Smaller images for faster correlation
4. **Enable early termination**: Stop RANSAC when good solution found

## Research Paper Implementation

This implementation follows the methodology described in:
"3D object detection and pose estimation from depth image for robotic bin picking"

### Key Equations Implemented

- **Equation 1**: 3D Harris keypoint detection υ = H(U), ω = H(W)
- **Equation 2-3**: Spin image formulation S_o → (α, β)
- **Equation 4**: Correlation coefficient R(P, Q)
- **Equation 5**: Rigid transformation U' = RU + T
- **Equation 6-7**: Pose evaluation metrics Pe and Ce

### Algorithm Adaptations

- **Adaptive neighborhood**: Modified k-ring Delaunay for variable density
- **Fraction selection**: Replaces clustering for symmetric objects
- **Color filtering**: Added LEGO-specific color detection
- **Enhanced visualization**: Real-time feedback with bounding boxes

## Future Enhancements

1. **Multi-brick detection**: Detect multiple bricks simultaneously
2. **Real-time processing**: Optimize for real-time performance
3. **Learning-based refinement**: Use ML to improve matching accuracy
4. **Robotic integration**: Add robot control and gripper positioning
5. **Quality assessment**: Confidence scoring for detection reliability

## License

This project is developed for research purposes at POSTECH Summer Program 2025.
