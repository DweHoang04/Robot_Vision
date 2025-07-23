# harris_detection.py
# Harris 3D Corner Detection Library
# Contains algorithms for 3D corner detection and neighborhood computation

import numpy as np
import itertools
import scipy.spatial
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from transformation import PointCloudTransformation


class Harris3DDetector:
    """
    A class implementing improved 3D Harris corner detection based on the 3D_Harris_IPD algorithm.
    Includes adaptive neighborhood computation and robust corner response calculation.
    """

    @staticmethod
    def polyfit3d(x, y, z, order=2):
        """
        Fit a 3D polynomial surface to the data points
        
        Args:
            x, y, z: coordinate arrays of the same length
            order: polynomial order (default: 2 for quadratic surface)
            
        Returns:
            numpy array: polynomial coefficients for the fitted surface
        """
        ncols = (order + 1)**2 # Number of coefficients for a polynomial of degree 'order'
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order+1), range(order+1))
        for k, (i, j) in enumerate(ij):
            G[:, k] = x**i * y**j
        m, _, _, _ = np.linalg.lstsq(G, z, rcond=None) # Solve the least squares problem
        return m

    @staticmethod
    def compute_delaunay_neighborhood(points, delta=0.025, max_iter=5):
        """
        Compute adaptive neighborhoods using Delaunay triangulation
        
        Args:
            points: numpy array of 3D points
            delta: neighborhood size parameter for adaptive k-ring expansion
            max_iter: maximum iterations for k-ring expansion
            
        Returns:
            dict: neighborhood mapping where keys are point indices and values are neighbor indices
        """
        if len(points) < 4:  # Need at least 4 points for Delaunay triangulation
            return {}
            
        try:
            triangulation = Delaunay(points)
        except:
            # Fall back to simple k-NN if Delaunay fails
            return Harris3DDetector.compute_knn_neighborhood(points, k=6)

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

    @staticmethod
    def compute_knn_neighborhood(points, k=6):
        """
        Fallback k-NN neighborhood computation
        
        Args:
            points: numpy array of 3D points
            k: number of nearest neighbors to find
            
        Returns:
            dict: neighborhood mapping for k-nearest neighbors
        """
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

    @staticmethod
    def compute_harris_3d_corners(points, delta=0.025, harris_k=0.04, fraction=0.1, 
                                cluster_threshold=0.01, num_corners=8):
        """
        Improved 3D Harris corner detection based on the 3D_Harris_IPD implementation.
        Now includes proper preprocessing steps: centering, scaling, and alignment.
        
        Args:
            points: numpy array of 3D points
            delta: neighborhood size parameter for adaptive k-ring
            harris_k: Harris corner detection parameter
            fraction: fraction of points to select as corners
            cluster_threshold: minimum distance between corners for clustering
            num_corners: maximum number of corners to return
            
        Returns:
            numpy array: detected corner points in original coordinate system
        """
        if len(points) < 10:
            print("Not enough points for Harris corner detection")
            return np.array([])

        print(f"Computing Harris corners for {len(points)} points...")
        
        # STEP 1: Preprocess points for stable Harris detection
        preprocessed_points, transform_params = PointCloudTransformation.preprocess_for_harris_detection(points)
        
        # STEP 2: Compute neighborhoods using adaptive Delaunay triangulation
        neighborhood = Harris3DDetector.compute_delaunay_neighborhood(preprocessed_points, delta=delta)
        
        if len(neighborhood) == 0:
            print("Failed to compute neighborhoods, falling back to k-NN")
            neighborhood = Harris3DDetector.compute_knn_neighborhood(preprocessed_points, k=6)
        
        # STEP 3: Initialize response array
        resp = np.zeros(len(preprocessed_points))
        
        # STEP 4: Compute Harris response for each point using preprocessed coordinates
        for point_idx in neighborhood.keys():
            try:
                if len(neighborhood[point_idx]) < 3:
                    resp[point_idx] = -np.inf
                    continue
                    
                neighbors = preprocessed_points[neighborhood[point_idx], :]
                
                # Center the neighbors around their local centroid
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
                    m = Harris3DDetector.polyfit3d(neighbors_2D[:, 0], neighbors_2D[:, 1], 
                                                 rotated_neighbors[:, 2], order=2)
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
        
        # STEP 5: Select interest points - find local maxima in preprocessed space
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
        num_fraction = max(1, int(fraction * len(preprocessed_points)))
        interest_points_fraction = candidate[:num_fraction, 0].astype(int)
        
        # Method 2: Cluster-based selection (avoid points too close to each other)
        selected_corners = []
        if len(candidate) > 0:
            # Start with the best corner
            selected_corners.append(int(candidate[0, 0]))
            Q = preprocessed_points[int(candidate[0, 0]), :].reshape((1, -1))
            
            # Add corners that are far enough from existing ones
            for i in range(1, len(candidate)):
                query = preprocessed_points[int(candidate[i, 0]), :].reshape((1, -1))
                distances = scipy.spatial.distance.cdist(query, Q, metric='euclidean')
                if np.min(distances) > cluster_threshold:
                    selected_corners.append(int(candidate[i, 0]))
                    Q = np.concatenate((Q, query), axis=0)
                    
                    # Stop if we have enough corners
                    if len(selected_corners) >= num_corners:
                        break
        
        # STEP 6: Return corner points in ORIGINAL coordinate system
        if len(selected_corners) > 0:
            # Get corner points from original (non-preprocessed) coordinates
            corner_points = points[selected_corners]
            
            # Optional: Apply inverse transformation if needed for specific applications
            # For now, we return points in original coordinates since preprocessing
            # was mainly for numerical stability during computation
            
            print(f"Found {len(corner_points)} Harris corners with improved preprocessing")
            return corner_points
        else:
            print("No valid corners found after clustering")
            return np.array([])
