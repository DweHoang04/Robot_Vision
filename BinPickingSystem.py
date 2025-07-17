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

class BinPickingSystem:

    def __init__(self, wdf_path):
        self.output_dir = "C:\\Binpicking"
        self.host = "192.168.1.23"
        self.port = 9999

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
        time.sleep(0.1)  # 렌더링 안정화
        vis.capture_screen_image(image_path)
        vis.destroy_window()
    '''    
    # 마이너스면 왼쪽으로 돌고 플러스면 오른쪽으로 돈다.
    def calculate_y_axis_angle_xy(self, minor_axis):
        ref_axis = np.array([0, 1])
        v2d = minor_axis[:2] / np.linalg.norm(minor_axis[:2])
        angle_deg = np.degrees(np.arccos(np.clip(np.dot(v2d, ref_axis), -1.0, 1.0)))
        if v2d[0] < 0:
            angle_deg = -angle_deg
        return angle_deg
    '''
    def calculate_y_axis_angle_xy(self, minor_axis):
        # 단위 벡터로 정규화
        v2d = minor_axis[:2] / np.linalg.norm(minor_axis[:2])

        def clockwise_angle_from_y(vec):
            # Y축 기준 시계방향 회전각도 (0~360)
            angle = np.degrees(np.arctan2(vec[0], vec[1])) % 360
            return angle

        angle1 = clockwise_angle_from_y(v2d)
        angle2 = clockwise_angle_from_y(-v2d)

        # 선(line)이므로 방향성 제거 → 둘 중 더 작은 회전각이 실제 선의 시계방향 회전각
        angle_deg = min(angle1, angle2)
        # 90° 초과 시 보완각으로 변환
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
            angle_deg = - angle_deg
        return angle_deg  # 최종적으로 음수 부호 붙여 반환

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
            print(f"전 : {len(lego_pts)}")
            # 🔍 클러스터 크기 필터링
            if len(lego_pts) < 500:
                continue
            print(f"후 : {len(lego_pts)}")

            # ⬇️ 시각화용 geometry 생성
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lego_pts)
            pcd.colors = o3d.utility.Vector3dVector(lego_cols)
            geometries.append(pcd)

            # ✅ PCA 및 중심/방향 계산
            pca = PCA(n_components=3).fit(lego_pts)
            center = np.mean(lego_pts, axis=0)
            angle_deg = self.calculate_y_axis_angle_xy(pca.components_[1])
            dom_color = self.get_dominant_color(lego_cols)
            results.append((center, dom_color, angle_deg))

        # # 🧩 시각화
        # if geometries:
        #     axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        #     o3d.visualization.draw_geometries([axes] + geometries, window_name="Filtered Clusters")

        # 💾 파일 저장
        with open(summary_file, "w") as f:
            for center, color, angle in results:
                cx, cy, cz = center * 1000.0
                r_, g_, b_ = color
                f.write(f"{cx:.2f} {cy:.2f} {cz:.2f} {r_:.6f} {g_:.6f} {b_:.6f} {angle:.2f}\n")

    def send_file_via_tcp(self, file_path):
        filename = os.path.basename(file_path).encode('utf-8')
        with open(file_path, "rb") as f:
            file_data = f.read()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            s.sendall(struct.pack('>I', len(filename)))
            s.sendall(filename)
            s.sendall(file_data)

    def run_pipeline(self):
        print("0")
        points, colors = self.capture_and_preprocess_kinect_data()
        if len(points) == 0:
            print("[PIPELINE] 유효 포인트가 없어 종료.")
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
        # 🔻 서버 전송 기능은 주석 처리
        self.send_file_via_tcp(summary_file)
        print("[PIPELINE] 모든 공정이 완료되었습니다.")
        
if __name__ == "__main__":
    # 예: WADF 경로가 필요 없는 경우 빈 문자열로 초기화하거나 필요 시 경로 지정
    system = BinPickingSystem(wdf_path="")
    system.run_pipeline()