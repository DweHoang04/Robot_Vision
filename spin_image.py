# -*- coding: utf-8 -*-
"""
Spin Image Feature Descriptor Module
Implementation of spin image computation and matching for 3D keypoints
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors
import cv2

class SpinImageDescriptor:
    def __init__(self, image_width=8, image_height=8, support_angle=60.0, 
                 bin_size=0.01, correlation_threshold=0.5):
        """
        Initialize Spin Image descriptor
        
        Args:
            image_width: Width of spin image in bins
            image_height: Height of spin image in bins  
            support_angle: Maximum angle for points to be included (degrees)
            bin_size: Size of each bin in the spin image
            correlation_threshold: Minimum correlation for matching
        """
        self.image_width = image_width
        self.image_height = image_height
        self.support_angle = np.radians(support_angle)
        self.bin_size = bin_size
        self.correlation_threshold = correlation_threshold
    
    def compute_surface_normals(self, points, k=20):
        """
        Compute surface normals for point cloud using PCA on local neighborhoods
        
        Args:
            points: Input point cloud (N x 3)
            k: Number of nearest neighbors for normal estimation
            
        Returns:
            normals: Surface normal vectors (N x 3)
        """
        print(f"Computing surface normals for {len(points)} points...")
        
        if len(points) < k:
            k = len(points) - 1
            
        normals = np.zeros_like(points)
        
        # Use nearest neighbors to find local neighborhoods
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
        
        for i, point in enumerate(points):
            try:
                # Find k nearest neighbors
                distances, indices = nbrs.kneighbors([point])
                neighbors = points[indices[0]]
                
                # Center the neighborhood
                centered = neighbors - np.mean(neighbors, axis=0)
                
                # Compute covariance matrix
                cov_matrix = np.cov(centered.T)
                
                # Compute eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # Normal is the eigenvector corresponding to smallest eigenvalue
                normal_vector = eigenvectors[:, 0]
                
                # Ensure normal points outward (toward camera/positive z generally)
                if normal_vector[2] < 0:
                    normal_vector = -normal_vector
                    
                normals[i] = normal_vector
                
            except Exception as e:
                # Default to z-up normal if computation fails
                normals[i] = [0, 0, 1]
        
        print("Surface normals computed successfully")
        return normals
    
    def compute_spin_image(self, point, normal, surface_points, max_radius=None):
        """
        Compute spin image for a single point
        
        Args:
            point: 3D keypoint position (3,)
            normal: Surface normal at keypoint (3,)
            surface_points: All surface points (N x 3)
            max_radius: Maximum radius for spin image (auto-computed if None)
            
        Returns:
            spin_image: 2D spin image array
        """
        # Normalize the normal vector
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        # Compute relative positions
        relative_pos = surface_points - point
        
        # Compute alpha (perpendicular distance to spin axis)
        # alpha = ||x - p||^2 - (n · (x - p))^2, then take sqrt
        dot_products = np.dot(relative_pos, normal)
        squared_distances = np.sum(relative_pos**2, axis=1)
        alpha_squared = squared_distances - dot_products**2
        
        # Avoid negative values due to numerical errors
        alpha_squared = np.maximum(alpha_squared, 0)
        alpha = np.sqrt(alpha_squared)
        
        # Compute beta (signed distance along spin axis)
        beta = dot_products
        
        # Filter points by support angle
        distances = np.linalg.norm(relative_pos, axis=1)
        valid_mask = distances > 1e-8  # Avoid division by zero
        
        if np.any(valid_mask):
            cos_angles = np.abs(dot_products[valid_mask]) / distances[valid_mask]
            angle_mask = np.arccos(np.clip(cos_angles, 0, 1)) <= self.support_angle
            valid_indices = np.where(valid_mask)[0][angle_mask]
        else:
            valid_indices = []
        
        if len(valid_indices) == 0:
            return np.zeros((self.image_height, self.image_width))
        
        # Get valid alpha and beta values
        valid_alpha = alpha[valid_indices]
        valid_beta = beta[valid_indices]
        
        # Determine image bounds
        if max_radius is None:
            max_alpha = np.max(valid_alpha) if len(valid_alpha) > 0 else self.bin_size
            max_radius = max(max_alpha, self.bin_size)
        
        min_beta = np.min(valid_beta) if len(valid_beta) > 0 else -self.bin_size
        max_beta = np.max(valid_beta) if len(valid_beta) > 0 else self.bin_size
        beta_range = max(max_beta - min_beta, self.bin_size)
        
        # Create spin image
        spin_image = np.zeros((self.image_height, self.image_width))
        
        # Bin the points
        alpha_bins = np.linspace(0, max_radius, self.image_width + 1)
        beta_bins = np.linspace(min_beta, max_beta, self.image_height + 1)
        
        for alpha_val, beta_val in zip(valid_alpha, valid_beta):
            # Find which bin each point belongs to
            alpha_bin = np.digitize(alpha_val, alpha_bins) - 1
            beta_bin = np.digitize(beta_val, beta_bins) - 1
            
            # Make sure bins are within image bounds
            alpha_bin = np.clip(alpha_bin, 0, self.image_width - 1)
            beta_bin = np.clip(beta_bin, 0, self.image_height - 1)
            
            # Increment the bin count
            spin_image[beta_bin, alpha_bin] += 1
        
        return spin_image
    
    def compute_spin_images_for_keypoints(self, keypoints, surface_points, surface_normals=None):
        """
        Compute spin images for all keypoints
        
        Args:
            keypoints: Array of keypoint positions (M x 3)
            surface_points: All surface points (N x 3)
            surface_normals: Precomputed surface normals (N x 3), computed if None
            
        Returns:
            spin_images: List of spin image arrays
            keypoint_normals: Surface normals at keypoints
        """
        print(f"Computing spin images for {len(keypoints)} keypoints...")
        
        if surface_normals is None:
            surface_normals = self.compute_surface_normals(surface_points)
        
        # Find normals for keypoints (closest surface point)
        keypoint_normals = []
        for kp in keypoints:
            distances = np.linalg.norm(surface_points - kp, axis=1)
            closest_idx = np.argmin(distances)
            keypoint_normals.append(surface_normals[closest_idx])
        
        keypoint_normals = np.array(keypoint_normals)
        
        # Compute spin images
        spin_images = []
        for i, (kp, normal) in enumerate(zip(keypoints, keypoint_normals)):
            if i % 5 == 0:  # Progress indicator
                print(f"  Computing spin image {i+1}/{len(keypoints)}")
            
            spin_img = self.compute_spin_image(kp, normal, surface_points)
            spin_images.append(spin_img)
        
        print(f"Computed {len(spin_images)} spin images")
        return spin_images, keypoint_normals
    
    def compute_correlation_coefficient(self, spin_img1, spin_img2):
        """
        Compute correlation coefficient between two spin images
        Using equation (4) from the paper
        
        Args:
            spin_img1, spin_img2: Spin image arrays
            
        Returns:
            correlation: Correlation coefficient R
        """
        # Flatten images
        p = spin_img1.flatten()
        q = spin_img2.flatten()
        
        N = len(p)
        
        if N == 0:
            return 0.0
        
        # Compute sums
        sum_p = np.sum(p)
        sum_q = np.sum(q)
        sum_pq = np.sum(p * q)
        sum_p2 = np.sum(p * p)
        sum_q2 = np.sum(q * q)
        
        # Compute correlation coefficient R(P,Q)
        numerator = N * sum_pq - sum_p * sum_q
        denominator_p = N * sum_p2 - sum_p * sum_p
        denominator_q = N * sum_q2 - sum_q * sum_q
        
        if denominator_p <= 0 or denominator_q <= 0:
            return 0.0
        
        denominator = np.sqrt(denominator_p * denominator_q)
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        return correlation
    
    def find_correspondences(self, target_spin_images, template_spin_images, 
                           target_keypoints, template_keypoints):
        """
        Find point correspondences using spin image matching
        
        Args:
            target_spin_images: Spin images from target object
            template_spin_images: Spin images from template object  
            target_keypoints: Target keypoint positions
            template_keypoints: Template keypoint positions
            
        Returns:
            correspondences: List of (target_idx, template_idx, correlation) tuples
        """
        print("Finding spin image correspondences...")
        
        correspondences = []
        
        for i, target_spin in enumerate(target_spin_images):
            best_correlation = -1
            best_template_idx = -1
            
            for j, template_spin in enumerate(template_spin_images):
                correlation = self.compute_correlation_coefficient(target_spin, template_spin)
                
                if correlation > best_correlation and correlation > self.correlation_threshold:
                    best_correlation = correlation
                    best_template_idx = j
            
            if best_template_idx >= 0:
                correspondences.append((i, best_template_idx, best_correlation))
        
        print(f"Found {len(correspondences)} correspondences above threshold {self.correlation_threshold}")
        return correspondences
    
    def save_spin_image(self, spin_image, filename, cmap='viridis'):
        """Save spin image as PNG file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(spin_image, cmap=cmap, origin='lower')
        plt.colorbar()
        plt.title('Spin Image')
        plt.xlabel('Alpha (radial distance)')
        plt.ylabel('Beta (axial distance)')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_spin_image_grid(self, spin_images, output_path, title="Spin Images", 
                           max_images=16, cmap='viridis'):
        """Save a grid of spin images for visualization"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        num_images = min(len(spin_images), max_images)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(num_images):
            if i < len(spin_images):
                axes[i].imshow(spin_images[i], cmap=cmap, origin='lower')
                axes[i].set_title(f'#{i+1}')
                axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Spin image grid saved to: {output_path}")
