# -*- coding: utf-8 -*-
"""
3D Harris Keypoint Detection Module
Based on the 3D_harris.py implementation with adaptations for bin picking
"""
import itertools
import numpy as np
from sklearn.decomposition import PCA
from math import sqrt
import neighborhoods
import transformation

class Harris3DDetector:
    def __init__(self, n_neighbours=3, delta=0.025, k=0.04, fraction=0.1):
        """
        Initialize Harris 3D keypoint detector
        
        Args:
            n_neighbours: Number of neighbors for fixed neighborhood methods
            delta: Parameter for adaptive neighborhood
            k: Harris corner response parameter
            fraction: Fraction of points to select as keypoints
        """
        self.n_neighbours = n_neighbours
        self.delta = delta
        self.k = k
        self.fraction = fraction
    
    def dot_product(self, vector_1, vector_2):
        return sum((a*b) for a, b in zip(vector_1, vector_2))

    def length(self, vector):
        return sqrt(self.dot_product(vector, vector))

    def angle(self, v1, v2):
        a = self.dot_product(v1, v2)/(self.length(v1)*self.length(v2))
        a = np.clip(a, -1, 1)
        return np.arccos(a)

    def polyfit3d(self, x, y, z, order=3):
        """Fit 3D polynomial surface"""
        ncols = (order + 1)**2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order+1), range(order+1))
        for k, (i,j) in enumerate(ij):
            G[:,k] = x**i * y**j
        m, _, _, _ = np.linalg.lstsq(G, z, rcond=None)
        return m

    def compute_harris_response(self, points, neighborhood_method='adaptive'):
        """
        Compute Harris corner response for all points
        
        Args:
            points: Input point cloud as numpy array (N x 3)
            neighborhood_method: 'k_ring_adaptive' or 'k_ring'
            
        Returns:
            resp: Harris response values for each point
            neighborhood: Dictionary of neighborhood indices
        """
        print(f"Computing Harris 3D keypoints for {len(points)} points...")
        
        # Initialize response array
        resp = np.zeros(len(points))
        
        # Compute neighborhood
        if neighborhood_method == 'adaptive':
            print("Using adaptive k-ring Delaunay neighborhood...")
            neighborhood = neighborhoods.k_ring_delaunay_adaptive(points, self.delta)
        else:
            print("Using k-ring Delaunay neighborhood...")
            neighborhood = neighborhoods.k_ring_delaunay(points, self.n_neighbours)
        
        print(f"Computed neighborhoods for {len(neighborhood)} points")
        
        # Compute Harris response for each point
        for point_idx in neighborhood.keys():
            try:
                neighbors = points[neighborhood[point_idx], :]
                
                if len(neighbors) < 3:
                    continue
                    
                # Center the ENTIRE point cloud (following original algorithm)
                points_centred, _ = transformation.centering_centroid(points)
                
                # Best fitting plane using PCA on ENTIRE point cloud
                pca = PCA(n_components=3) # Principal Component Analysis
                points_pca = pca.fit_transform(np.transpose(points_centred))
                eigenvalues, eigenvectors = np.linalg.eigh(points_pca)
                idx = np.argmin(eigenvalues, axis=0)
                best_fit_normal = eigenvectors[idx,:]
                
                # Rotate the ENTIRE point cloud to align with principal components
                points_rotated = points.copy()
                for i in range(points.shape[0]):
                    points_rotated[i, :] = np.dot(np.transpose(eigenvectors), points[i, :])

                # Restrict to XY plane and translate (using rotated entire point cloud)
                points_2D = points_rotated[:,:2] - points_rotated[point_idx,:2]

                # Fit a quadratic surface using entire transformed point cloud
                m = self.polyfit3d(points_2D[:,0], points_2D[:,1], points_rotated[:,2], order=2)
                m = m.reshape((3,3))

                # Compute the derivative components
                fx2  = m[1, 0]*m[1, 0] + 2*m[2, 0]*m[2, 0] + 2*m[1, 1]*m[1, 1]  # A
                fy2  = m[1, 0]*m[1, 0] + 2*m[1, 1]*m[1, 1] + 2*m[0, 2]*m[0, 2]  # B
                fxfy = m[1, 0]*m[0, 1] + 2*m[2, 0]*m[1, 1] + 2*m[1, 1]*m[0, 2]  # C

                # Compute Harris corner response
                resp[point_idx] = fx2*fy2 - fxfy*fxfy - self.k*(fx2 + fy2)*(fx2 + fy2)
                    
            except Exception as e:
                # If any error occurs in processing this point, skip it
                continue
        
        return resp, neighborhood

    def select_keypoints(self, points, resp, neighborhood, method='fraction'):
        """
        Select keypoints based on Harris response values
        
        Args:
            points: Input point cloud
            resp: Harris response values
            neighborhood: Neighborhood dictionary
            method: Selection method ('fraction' or 'cluster')
            
        Returns:
            keypoints: Selected keypoint coordinates
            keypoint_indices: Indices of selected keypoints
        """
        print("Selecting interest points...")
        
        # Search for local maxima
        candidate = []
        for point_idx in neighborhood.keys():
            if resp[point_idx] >= np.max(resp[neighborhood[point_idx]]):
                candidate.append([point_idx, resp[point_idx]])
        
        # Sort by decreasing response value
        candidate.sort(reverse=True, key=lambda x: x[1])
        candidate = np.array(candidate)
        
        if len(candidate) == 0:
            print("No keypoint candidates found!")
            return np.array([]), np.array([])
        
        if method == 'fraction':
            # Method 1: Select top fraction of points
            num_keypoints = max(1, int(self.fraction * len(points)))
            num_keypoints = min(num_keypoints, len(candidate))
            keypoint_indices = candidate[:num_keypoints, 0].astype(int)
            
        else:
            # Method 2: Cluster-based selection (for reference)
            cluster_threshold = 0.01
            selected_indices = [int(candidate[0, 0])]
            Q = points[int(candidate[0, 0]), :].reshape((1, -1))
            
            for i in range(1, len(candidate)):
                query = points[int(candidate[i, 0]), :].reshape((1, -1))
                from scipy.spatial.distance import cdist
                distances = cdist(query, Q, metric='euclidean')
                if np.min(distances) > cluster_threshold:
                    Q = np.concatenate((Q, query), axis=0)
                    selected_indices.append(int(candidate[i, 0]))
            
            keypoint_indices = np.array(selected_indices)
        
        keypoints = points[keypoint_indices]
        
        print(f"Selected {len(keypoints)} keypoints using {method} method")
        return keypoints, keypoint_indices

    def detect_keypoints(self, points, neighborhood_method='adaptive', selection_method='fraction'):
        """
        Main function to detect 3D Harris keypoints
        
        Args:
            points: Input point cloud (N x 3)
            neighborhood_method: 'adaptive' or 'k_ring'
            selection_method: 'fraction' or 'cluster'
            
        Returns:
            keypoints: Detected keypoint coordinates
            keypoint_indices: Indices of keypoints in original point cloud
            resp: Harris response values for all points
        """
        if len(points) < 10:
            print("Insufficient points for keypoint detection")
            return np.array([]), np.array([]), np.array([])
        
        # Compute Harris response
        resp, neighborhood = self.compute_harris_response(points, neighborhood_method)
        
        # Select keypoints
        keypoints, keypoint_indices = self.select_keypoints(points, resp, neighborhood, selection_method)
        
        return keypoints, keypoint_indices, resp
