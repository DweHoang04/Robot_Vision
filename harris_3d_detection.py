# harris_3d_detection.py
# 3D Harris corner detection module for LEGO brick analysis

import numpy as np
import itertools
import scipy.spatial.distance
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from collections import Counter

class Harris3DDetection:
    """
    3D Harris corner detection specialized for LEGO brick analysis
    Includes LEGO-specific filtering and multi-hypothesis generation
    """
    
    def __init__(self):
        pass
    
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
        if len(points) < 4:
            return {}
            
        try:
            triangulation = Delaunay(points)
        except:
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
            if len(neighborhood_direct[v]) > 0:
                dist = np.max(np.linalg.norm(query - points[neighborhood_direct[v]], axis=1))
                if dist >= delta:
                    bigger_ring = False
                    neighborhood[v] = neighborhood_direct[v]
                else:
                    bigger_ring = True
            else:
                bigger_ring = False
                neighborhood[v] = [v]

            iteration = 1
            while bigger_ring and iteration <= max_iter:
                iteration += 1
                for neighbor in neighborhood_direct[v]:
                    if neighbor in neighborhood_direct:
                        if v in neighborhood.keys():
                            neighborhood[v] = list(np.unique(neighborhood[v] + neighborhood_direct[neighbor]))
                        else:
                            neighborhood[v] = list(np.unique(neighborhood_direct[v] + neighborhood_direct[neighbor]))

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
        """Center the point cloud on its centroid"""
        centred_points = points.copy()
        centroid = np.mean(centred_points, axis=0)
        centred_points = centred_points - centroid
        return centred_points, centroid

    def scale_point_cloud(self, points, target_scale=1.0):
        """Scale point cloud to a target scale for consistent processing"""
        scaled_points = points.copy()
        
        distances = np.linalg.norm(scaled_points, axis=1)
        current_scale = np.max(distances)
        
        if current_scale > 0:
            scaling_factor = target_scale / current_scale
            scaled_points = scaled_points * scaling_factor
            return scaled_points, scaling_factor
        else:
            return scaled_points, 1.0

    def preprocess_for_harris_detection(self, points):
        """Simplified preprocessing pipeline for Harris 3D corner detection"""
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

    def compute_harris_3d_corners_multi_hypothesis(self, points, delta=0.025, harris_k=0.04, 
                                                 cluster_threshold=0.008, num_corners=8, num_hypotheses=3):
        """Enhanced Harris corner detection that returns multiple valid corner hypotheses"""
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
        
        # STEP 5: Select corner candidates based on local maxima
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
        
        print(f"Generated {len(all_hypotheses)} corner hypotheses using direct Harris 3D detection")
        return all_hypotheses

    def validate_brick_cluster(self, corner_points, min_corners=4):
        """Simple validation for brick clusters"""
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
        """Select the best pose hypothesis from multiple candidates"""
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
