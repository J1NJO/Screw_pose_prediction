import os
import json
import numpy as np
import open3d as o3d
import cv2
from ultralytics import YOLO
from PIL import Image
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt

class InputDataHandler:
    def __init__(self, base_path):
        """
        Initialize the InputDataHandler with the base path to the dataset.

        Args:
            base_path (str): Path to the folder containing data folders for each camera snapshot.
        """
        self.base_path = base_path

    def parse_image(self, image_path):
        """
        Load an RGB image using OpenCV.

        Returns:
            np.ndarray: The loaded image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image

    def parse_pointcloud(self, ply_path):
        """
        Load a point cloud from a .ply file using Open3D.

        Returns:
            open3d.geometry.PointCloud: The loaded point cloud.
        """
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Point cloud file not found: {ply_path}")

        pointcloud = o3d.io.read_point_cloud(ply_path)
        if pointcloud.is_empty():
            raise ValueError(f"Point cloud is empty: {ply_path}")
        #print(pointcloud.shape)
        return pointcloud

    def parse_transformation_matrix(self, json_path):
        """
        Load the transformation matrix from a JSON file.

        Returns:
            np.ndarray: The 4x4 transformation matrix as a numpy array.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Transformation matrix file not found: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) != 4 or not all(len(row) == 4 for row in data):
            raise ValueError(f"Invalid transformation matrix format in file: {json_path}")

        return np.array(data, dtype=np.float64)

    def get_file_paths(self, folder_name):
        """
        Get paths for the image, point cloud, and JSON file in a snapshot folder.

        Returns:
            dict: Dictionary containing paths for the image, point cloud, and JSON file.
        """
        snapshot_path = os.path.join(self.base_path, folder_name)
        image_path = os.path.join(snapshot_path, f'{folder_name}.png')
        pointcloud_path = os.path.join(snapshot_path, f'{folder_name}.ply')
        json_path = os.path.join(snapshot_path, f'{folder_name}.json')

        return {
            "image_path": image_path,
            "pointcloud_path": pointcloud_path,
            "json_path": json_path
        }

    def load_snapshot(self, folder_name):
        """
        Load all data for a single snapshot.

        Returns:
            dict: A dictionary containing the image, point cloud, and transformation matrix.
        """
        paths = self.get_file_paths(folder_name)
        return {
            "image": self.parse_image(paths["image_path"]),
            "pointcloud": self.parse_pointcloud(paths["pointcloud_path"]),
            "transformation_matrix": self.parse_transformation_matrix(paths["json_path"]),
            "image_path": paths["image_path"]
        }

    def load_all_snapshots(self):
        """
        Load all snapshots in the base path.

        Returns:
            list: A list of dictionaries, each containing the image, point cloud, and transformation matrix for a snapshot.
        """
        snapshots = []

        for folder_name in sorted(os.listdir(self.base_path)):
            snapshot_path = os.path.join(self.base_path, folder_name)
            if os.path.isdir(snapshot_path):
                try:
                    snapshot_data = self.load_snapshot(folder_name)
                    snapshots.append(snapshot_data)
                except (FileNotFoundError, ValueError) as e:
                    print(f"Skipping snapshot {snapshot_path}: {e}")

        return snapshots


def run_yolo(yolo, image_url, save_path, output_file, conf=0.25, iou=0.7):
    # YOLO processing function

    results = yolo(image_url, conf=conf, iou=iou)
    bbox_centers = []  # To store the centers of bounding boxes for class 0

    # Open the output file to save bounding box coordinates
    with open(output_file, 'w') as f:
        for r in results:
            for box in r.boxes:
                coordinates = (box.xyxy).tolist()[0]
                cls = int(box.cls)  # Get the class index
                left, top, right, bottom = coordinates[0], coordinates[1], coordinates[2], coordinates[3]

                # Write class and bounding box coordinates to the file
                f.write(f"Class: {cls}, Coordinates: {left}, {top}, {right}, {bottom}\n")

                # If the class is 0, calculate the center and add to bbox_centers
                if cls == 0:
                    center_x = (left + right) / 2
                    center_y = (top + bottom) / 2
                    bbox_centers.append([center_x, center_y])

    # Convert bbox_centers to a numpy array
    bbox_centers = np.array(bbox_centers)

    # Plot the results and save the annotated image
    res = results[0].plot()[:, :, [2, 1, 0]]
    image = Image.fromarray(res)
    image.save(save_path)
    return image, bbox_centers


import open3d as o3d
import numpy as np

def add_bounding_box_to_pointcloud(pointcloud, screw_poses, size=20):
    """
    Add bounding boxes around screws in the point cloud for visualization, using screw pose (position and orientation).

    Args:
        pointcloud (open3d.geometry.PointCloud): The point cloud.
        screw_poses (list): List of screw poses, each containing a position and orientation.
        size (float): Size of the bounding box.

    Returns:
        list: List of open3d.geometry.AxisAlignedBoundingBox objects representing bounding boxes around screws.
    """
    bbox_objects = []

    for pose in screw_poses:
        position = pose["position"]
        # Get bounding box dimensions (size could be adjusted based on screw dimensions)
        half_size = size / 2

        # Create axis-aligned bounding box centered at the screw position
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array(position) - half_size,
            max_bound=np.array(position) + half_size
        )
        
        # Optionally, we could align the bounding box according to the screw orientation,
        # but here we keep it axis-aligned to simplify visualization.
        bbox.color = (1, 0, 0)  # Color the bounding boxes red
        bbox_objects.append(bbox)

    return bbox_objects

def project_pointcloud_to_image(pointcloud, intrinsics):
    """
    Project the 3D point cloud to 2D image coordinates using camera intrinsics.

    Returns:
        np.ndarray: 2D pixel coordinates corresponding to 3D points.
        np.ndarray: Depths of the 3D points.
    """
    points = np.asarray(pointcloud.points)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    # Filter out points with Z <= 0 (behind the camera or invalid)
    valid_points = points[points[:, 2] > 0]
    X, Y, Z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]

    # Project to 2D image space
    u = (fx * X / Z + cx).astype(np.int32)
    v = (fy * Y / Z + cy).astype(np.int32)
    image_coords = np.stack((u, v), axis=-1)

    return image_coords, valid_points

def get_screw_pose(bbox_centers, image_coords, valid_points):
    """
    Compute the pose (position and orientation) of screws.

    Args:
        bbox_centers (list): List of bounding box centers from YOLO.
        image_coords (np.ndarray): 2D pixel coordinates of the point cloud.
        valid_points (np.ndarray): 3D points corresponding to the image coordinates.

    Returns:
        list: List of screw poses as dictionaries containing position and orientation.
    """
    screw_poses = []

    for center_x, center_y in bbox_centers:
        # Find the nearest 2D point in the point cloud to the bounding box center
        distances = np.linalg.norm(image_coords - np.array([center_x, center_y]), axis=1)
        nearest_idx = np.argmin(distances)
        position = valid_points[nearest_idx]

        # Define orientation: z-axis points away from the plane
        z_axis = np.array([0, 0, 1])  # Screw head points away from the plane
        x_axis = np.array([1, 0, 0])  # Arbitrary x-axis
        y_axis = np.cross(z_axis, x_axis)  # Compute y-axis

        # Normalize all axes to ensure a valid rotation matrix
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Orientation as a rotation matrix
        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T

        # Store the pose
        screw_poses.append({
            "position": position,
            "orientation": rotation_matrix
        })

    return screw_poses


def write_screw_poses_to_json(screw_poses, output_file):
    """
    Write the list of screw poses to a JSON file.

    Args:
        screw_poses (list): List of screw poses with position and orientation.
        output_file (str): Path to the output JSON file.
    """
    poses_data = []

    for pose in screw_poses:
        poses_data.append({
            "position": pose["position"].tolist(),
            "orientation": pose["orientation"].tolist()  # Convert to list for JSON serialization
        })

    data = {"screws": poses_data}

    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Screw poses have been written to {output_file}")


def transform_screw_pose(screw_poses, transformation_matrix):
    """
    Transform the screw poses from the camera frame to the robot/world frame.

    Args:
        screw_poses (list): List of screw poses in the camera frame.
        transformation_matrix (np.ndarray): 4x4 transformation matrix.

    Returns:
        list: Transformed screw poses in the robot/world frame.
    """
    transformed_poses = []

    for pose in screw_poses:
        # Transform position
        position = np.append(pose["position"], 1)  # Homogeneous coordinates
        transformed_position = np.dot(transformation_matrix, position)[:3]

        # Transform orientation (rotation matrix)
        rotation_matrix = pose["orientation"]
        transformed_rotation_matrix = np.dot(transformation_matrix[:3, :3], rotation_matrix)

        transformed_poses.append({
            "position": transformed_position,
            "orientation": transformed_rotation_matrix
        })

    return transformed_poses



if __name__ == "__main__":
    base_path = "/project/test/dataset"
    handler = InputDataHandler(base_path)

    all_snapshots = handler.load_all_snapshots()

    print(f"Loaded {len(all_snapshots)} snapshots.")
    yolo = YOLO('/project/scripts/best.pt')
    f = 2480 #focal length in pixels
    camera_intrinsic_matrix = np.array([[f, 0, 1240], [0, f, 1034], [0, 0, 1]])
    
    for i, snapshot in enumerate(all_snapshots):
        print(f"Snapshot {i + 1}:")
        print(f"  Image shape: {snapshot['image'].shape}")
        print(f"  Point cloud: {len(snapshot['pointcloud'].points)} points")
        print(f"  Transformation matrix:\n{snapshot['transformation_matrix']}")

        # Run YOLO on the image
        image_url = snapshot['image_path']
        save_path = os.path.join(base_path, f"result_snapshot_{i + 1}.png")
        output_file = os.path.join(base_path, f"result_snapshot_{i + 1}.txt")
        annotated_image, bbox_centers = run_yolo(yolo, image_url, save_path, output_file)
        #print(f"Bounding box centers for class 0 in snapshot {i + 1}:\n{bbox_centers}")

        # Get 3D coordinates of screws
        image_coords, valid_points = project_pointcloud_to_image(snapshot['pointcloud'], camera_intrinsic_matrix)
        screw_poses = get_screw_pose(bbox_centers, image_coords, valid_points)
        print(f"3D coordinates of screws in snapshot {i + 1}:\n{screw_poses}")
        bounding_boxes = add_bounding_box_to_pointcloud(snapshot['pointcloud'], screw_poses)
        o3d.visualization.draw_geometries([snapshot['pointcloud']] + bounding_boxes)

        # Transform screw coordinates
        transformed_screw_poses = transform_screw_pose(screw_poses, snapshot['transformation_matrix'])
        print(f"3D transformed coordinates of screws in snapshot {i + 1}:\n{transformed_screw_poses}")
        
        # Write the screw coordinates to the JSON file
        output_file = os.path.join(base_path, f"result_snapshot_{i + 1}.json")   
        write_screw_poses_to_json(transformed_screw_poses, output_file)