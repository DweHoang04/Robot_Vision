# -*- coding: utf-8 -*-
"""
RANSAC-based Pose Estimation Module
Implementation of rigid transformation estimation with RANSAC
"""
import numpy as np
import random
from scipy.spatial.transform import Rotation

class RANSACPoseEstimator:
    def __init__(self, epsilon1=0.005, epsilon2=0.01, max_iterations=1000, 
                 max_outer_iterations=50, min_inliers=3):
        """
        Initialize RANSAC pose estimator
        
        Args:
            epsilon1: Distance threshold for pose evaluation (Pe)
            epsilon2: Centroid distance threshold (Ce)
            max_iterations: Maximum RANSAC iterations
            max_outer_iterations: Maximum outer loop iterations
            min_inliers: Minimum number of inliers required
        """
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.max_iterations = max_iterations
        self.max_outer_iterations = max_outer_iterations
        self.min_inliers = min_inliers
    
    def estimate_rigid_transformation(self, source_points, target_points):
        """
        Estimate rigid transformation between two point sets using least squares
        
        Args:
            source_points: Source point set (N x 3)
            target_points: Target point set (N x 3)
            
        Returns:
            R: Rotation matrix (3 x 3)
            T: Translation vector (3,)
        """
        if len(source_points) != len(target_points) or len(source_points) < 3:
            return None, None
        
        # Center the point sets
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid
        
        # Compute cross-covariance matrix
        H = source_centered.T @ target_centered
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        T = target_centroid - R @ source_centroid
        
        return R, T
    
    def apply_transformation(self, points, R, T):
        """Apply rigid transformation to points"""
        return (R @ points.T).T + T
    
    def compute_pose_error(self, source_points, target_points, R, T):
        """
        Compute pose evaluation metrics Pe and Ce from equations (6) and (7)
        
        Args:
            source_points: Source keypoints (N x 3)
            target_points: Target keypoints (N x 3)  
            R: Rotation matrix (3 x 3)
            T: Translation vector (3,)
            
        Returns:
            Pe: Mean distance error
            Ce: Centroid distance error
        """
        # Transform source points
        transformed_source = self.apply_transformation(source_points, R, T)
        
        # Compute Pe: mean distance between corresponding points (equation 6)
        distances = np.linalg.norm(target_points - transformed_source, axis=1)
        Pe = np.mean(distances)
        
        # Compute Ce: distance between centroids (equation 7)
        source_centroid = np.mean(transformed_source, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        Ce = np.linalg.norm(target_centroid - source_centroid)
        
        return Pe, Ce
    
    def ransac_pose_estimation(self, template_keypoints, target_keypoints, correspondences):
        """
        RANSAC-based pose estimation following the algorithm in the paper
        
        Args:
            template_keypoints: Template keypoint positions (M x 3)
            target_keypoints: Target keypoint positions (N x 3)
            correspondences: List of (target_idx, template_idx, score) tuples
            
        Returns:
            best_R: Best rotation matrix
            best_T: Best translation vector
            best_inliers: Indices of inlier correspondences
        """
        if len(correspondences) < 3:
            print(f"Insufficient correspondences: {len(correspondences)} < 3")
            return None, None, []
        
        print(f"Running RANSAC with {len(correspondences)} correspondences...")
        
        # Extract corresponding point pairs
        target_indices = [c[0] for c in correspondences]
        template_indices = [c[1] for c in correspondences]
        
        Uc = template_keypoints[template_indices]  # Template correspondences
        Wc = target_keypoints[target_indices]      # Target correspondences
        
        best_R = None
        best_T = None
        best_inliers = []
        best_score = -1
        
        # Outer loop: Check Ce constraint
        for outer_iter in range(self.max_outer_iterations):
            Ce = float('inf')
            Pe = float('inf')
            current_R = None
            current_T = None
            current_inliers = []
            
            # Inner loop: Check Pe constraint
            iteration = 0
            while Pe > self.epsilon1 and iteration < self.max_iterations:
                iteration += 1
                
                # Randomly select 3 correspondence pairs
                if len(correspondences) < 3:
                    break
                    
                sample_indices = random.sample(range(len(correspondences)), 3)
                sample_template = Uc[sample_indices]
                sample_target = Wc[sample_indices]
                
                # Estimate rigid transformation
                R, T = self.estimate_rigid_transformation(sample_template, sample_target)
                
                if R is None or T is None:
                    continue
                
                # Evaluate transformation on all correspondences
                Pe, Ce_temp = self.compute_pose_error(Uc, Wc, R, T)
                
                # Count inliers
                transformed_template = self.apply_transformation(Uc, R, T)
                distances = np.linalg.norm(Wc - transformed_template, axis=1)
                inlier_mask = distances < self.epsilon1
                num_inliers = np.sum(inlier_mask)
                
                if Pe <= self.epsilon1 and num_inliers >= self.min_inliers:
                    current_R = R
                    current_T = T
                    current_inliers = np.where(inlier_mask)[0]
                    Ce = Ce_temp
                    break
            
            # Check if we found a good solution
            if current_R is not None and Ce <= self.epsilon2:
                # Score based on number of inliers and low error
                score = len(current_inliers) / (1.0 + Pe + Ce)
                
                if score > best_score:
                    best_R = current_R
                    best_T = current_T
                    best_inliers = current_inliers
                    best_score = score
                    
                    print(f"Found better solution: Pe={Pe:.6f}, Ce={Ce:.6f}, "
                          f"inliers={len(current_inliers)}, score={score:.3f}")
                
                # Early termination if we found a very good solution
                if len(current_inliers) > len(correspondences) * 0.8:
                    break
        
        if best_R is not None:
            final_Pe, final_Ce = self.compute_pose_error(Uc, Wc, best_R, best_T)
            print(f"Final RANSAC result: Pe={final_Pe:.6f}, Ce={final_Ce:.6f}, "
                  f"inliers={len(best_inliers)}/{len(correspondences)}")
        else:
            print("RANSAC failed to find valid transformation")
        
        return best_R, best_T, best_inliers
    
    def refine_with_all_inliers(self, template_keypoints, target_keypoints, 
                               correspondences, inlier_indices):
        """
        Refine pose estimation using all inlier correspondences
        
        Args:
            template_keypoints: Template keypoint positions  
            target_keypoints: Target keypoint positions
            correspondences: All correspondences
            inlier_indices: Indices of inlier correspondences
            
        Returns:
            R_refined: Refined rotation matrix
            T_refined: Refined translation vector
        """
        if len(inlier_indices) < 3:
            return None, None
        
        # Extract inlier correspondences
        inlier_correspondences = [correspondences[i] for i in inlier_indices]
        target_indices = [c[0] for c in inlier_correspondences]
        template_indices = [c[1] for c in inlier_correspondences]
        
        template_inliers = template_keypoints[template_indices]
        target_inliers = target_keypoints[target_indices]
        
        # Re-estimate transformation using all inliers
        R_refined, T_refined = self.estimate_rigid_transformation(template_inliers, target_inliers)
        
        if R_refined is not None:
            Pe, Ce = self.compute_pose_error(template_inliers, target_inliers, R_refined, T_refined)
            print(f"Refined pose: Pe={Pe:.6f}, Ce={Ce:.6f}")
        
        return R_refined, T_refined
    
    def pose_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles (rx, ry, rz) in degrees"""
        rotation = Rotation.from_matrix(R)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        return euler_angles
