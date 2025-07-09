import numpy as np
import os
import json
import pyvista as pv
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def process_and_save_bbox(space_name, object_id, description, eps=0.2, min_samples=3):
    """
    Process the point cloud data of the specified space and object ID,
    and save the largest bounding box information as a JSON file.

    Args:
        space_name (str): Name of the space to process.
        object_id (str): Target object ID.
        description (str): Description of the object.
        eps (float): DBSCAN neighborhood radius.
        min_samples (int): Minimum number of samples for DBSCAN clustering.
    """
    def load_label_points(class_id):
        """
        Load the labeled point cloud data of a specific class.

        Args:
            class_id (str): The class ID to load.

        Returns:
            dict or None: A dictionary of labeled points if exists, otherwise None.
        """
        file_path = os.path.join(f'../Data/{space_name}/castresults', f"{class_id}_label_points.npy")
        if not os.path.exists(file_path):
            print(f"Error: {file_path} does not exist.")
            return None
        return np.load(file_path, allow_pickle=True).item()

    def compute_obb(points):
        """
        Compute the oriented bounding box (OBB) of the given point cloud using PCA.

        Args:
            points (np.ndarray): The point cloud data.

        Returns:
            tuple: (obb_corners, center, axes), where each is a list.
        """
        pca = PCA(n_components=3)
        pca.fit(points)
        center = np.mean(points, axis=0)
        axes = pca.components_
        transformed_points = np.dot(points - center, axes.T)

        min_vals = np.min(transformed_points, axis=0)
        max_vals = np.max(transformed_points, axis=0)

        corners = np.array([
            [min_vals[0], min_vals[1], min_vals[2]],
            [max_vals[0], min_vals[1], min_vals[2]],
            [max_vals[0], max_vals[1], min_vals[2]],
            [min_vals[0], max_vals[1], min_vals[2]],
            [min_vals[0], min_vals[1], max_vals[2]],
            [max_vals[0], min_vals[1], max_vals[2]],
            [max_vals[0], max_vals[1], max_vals[2]],
            [min_vals[0], max_vals[1], max_vals[2]],
        ])
        obb_corners = np.dot(corners, axes) + center
        return obb_corners.tolist(), center.tolist(), axes.tolist()

    class_id = object_id
    all_label_points = load_label_points(class_id)
    if all_label_points is None:
        return

    bbox_data = {
        "space_name": space_name,
        "object_id": object_id,
        "bounding_boxes": [],
        "description": description
    }

    for idx, (label, data) in enumerate(all_label_points.items()):
        points = np.array(data["points"], dtype=np.float64)
        if points.size == 0:
            continue

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) == 0:
            continue

        max_cluster_label = unique_labels[np.argmax(counts)]
        max_cluster_points = points[labels == max_cluster_label]

        if len(max_cluster_points) >= 3:
            obb_corners, obb_center, obb_axes = compute_obb(max_cluster_points)
            bbox_data["bounding_boxes"].append({
                "label": int(label),
                "obb_corners": obb_corners,
                "obb_center": obb_center,
                "obb_axes": obb_axes
            })
    
    output_dir = f'../Data/{space_name}/bbox_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{object_id}_bbox.json")
    
    with open(output_path, "w") as json_file:
        json.dump(bbox_data, json_file, indent=4)
    
    print(f"Bounding box data saved to {output_path}")

