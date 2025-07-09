import os
import re
import numpy as np
import pyvista as pv
import xml.etree.ElementTree as ET
import glob
from camera_utils import load_camera_parameters_from_xml, correct_camera_matrix

def read_mesh(filepath):
    """
    Load and compute normals for the 3D mesh.
    """
    print("Loading mesh from:", filepath)
    mesh = pv.read(filepath)
    mesh.compute_normals()
    print("Normals computed.")
    return mesh

def pixel_to_3d_coordinates(combined_data, corrected_camera_matrix, img_pos, img_orient, label, depth_scale=1):
    """
    Convert segmentation and depth information to 3D point cloud.
    """
    fx, fy = corrected_camera_matrix[0, 0], corrected_camera_matrix[1, 1]
    cx, cy = corrected_camera_matrix[0, 2], corrected_camera_matrix[1, 2]
    coords_3d = []

    # Select pixels belonging to the specified label
    label_pixels = np.column_stack(np.where(combined_data[..., 0] == label))

    for (py, px) in label_pixels:
        Z = combined_data[py, px, 1] * depth_scale
        X = (px - cx) * Z / fx
        Y = (py - cy) * Z / fy
        point_camera = np.array([X, Y, Z])
        point_world = img_orient @ point_camera + img_pos
        coords_3d.append(point_world)

    return np.array(coords_3d)


def find_correct_depth_scale(mesh, combined_data, corrected_camera_matrix, img_pos, img_orient, labels, 
                             threshold=0.17, scale_step=1.1, max_iterations=100, stop_threshold=0.01, bad_frame_limit=50,
                             max_depth_scale=25.0):
    """
    Optimize depth scale using ray tracing to maximize intersection ratio.
    """
    depth_scale = 1.01  # initial scale
    all_data = {label: {"original_points": None, "scaled_points": None, "intersections": []} for label in labels}
    
    prev_intersection_ratios = []
    last_ratio = 0.0

    for iteration in range(max_iterations):
        if depth_scale > max_depth_scale:
            print(f"Depth scale exceeded maximum of {max_depth_scale}. Skipping frame.")
            return None, None

        intersection_count = 0
        total_points = 0

        for label in labels:
            original_points = pixel_to_3d_coordinates(
                combined_data, corrected_camera_matrix, img_pos, img_orient, label, depth_scale=1
            )
            scaled_points = pixel_to_3d_coordinates(
                combined_data, corrected_camera_matrix, img_pos, img_orient, label, depth_scale
            )

            label_intersections = []
            for origin, target in zip(original_points, scaled_points):
                ray_direction = (target - origin) / np.linalg.norm(target - origin)
                intersections, _ = mesh.ray_trace(origin, origin + ray_direction * np.linalg.norm(target - origin))
                if len(intersections) > 0:
                    intersection_count += 1
                    label_intersections.extend(intersections)

            all_data[label]["original_points"] = original_points
            all_data[label]["scaled_points"] = scaled_points
            all_data[label]["intersections"] = label_intersections
            total_points += len(scaled_points)

        intersection_ratio = intersection_count / total_points if total_points > 0 else 0
        print(f"Iteration {iteration + 1}: depth_scale={depth_scale:.3f}, intersection_ratio={intersection_ratio:.2%}")

        # Track intersection ratios of the last `bad_frame_limit` iterations
        prev_intersection_ratios.append(intersection_ratio)
        if len(prev_intersection_ratios) > bad_frame_limit:
            prev_intersection_ratios.pop(0)
        
        # If improvement is insignificant, skip the frame
        if len(prev_intersection_ratios) == bad_frame_limit:
            max_change = max(prev_intersection_ratios) - min(prev_intersection_ratios)
            if max_change < stop_threshold:
                print(f"Frame has poor performance, skipping. No significant improvement after {bad_frame_limit} iterations.")
                return None, None
        
        # Stop if desired threshold reached
        if intersection_ratio >= threshold:
            print(f"Found correct depth scale: {depth_scale:.3f}")
            return depth_scale, all_data
        
        depth_scale *= scale_step

    print("No valid depth scale found within max iterations.")
    return None, None


def visualize_all_frames(mesh, all_label_points):
    """
    Visualize the mesh and all label points in a PyVista plot.
    """
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="lightgray", opacity=0.5, label="3D Mesh")

    colors = ["blue", "green", "yellow", "purple", "orange", "cyan"]
    for idx, (label, data) in enumerate(all_label_points.items()):
        color = colors[idx % len(colors)]
        points = np.array(data["points"])
        if points.size > 0:
            plotter.add_points(points, color=color, point_size=5, label=f"Label {label}")

    plotter.add_legend()
    plotter.show()

   
def process_depth_segmentation(space_name, object_id):
    """
    Process depth and segmentation data for a given space and object,
    perform point cloud computation and optionally visualization.

    Args:
        space_name (str): Name of the space to process.
        object_id (str): Target object ID.
    """
    xml_path = f'../Data/{space_name}/{space_name}.xml'
    segdep_folder = f'../Data/{space_name}/segdep/{object_id}'
    model_path = f'../Data/{space_name}/testnocolor.obj'
    class_id = object_id

    img_position, img_orientation, camera_matrix = load_camera_parameters_from_xml(xml_path)
    mesh = read_mesh(model_path)

    frame_files = glob.glob(os.path.join(segdep_folder, '*.npy'))
    frame_files = frame_files[:4]
    all_label_points = {}

    for frame_file in sorted(frame_files):
        base_name = os.path.basename(frame_file)
        match = re.search(r'frame_(\d+)_', base_name)

        if not match:
            continue

        frame_number = int(match.group(1))
        print(f"Now processing frame {frame_number}...")
        if frame_number not in img_position:
            print(f"Frame {frame_number} not found in camera parameters. Skipping.")
            continue

        combined_data = np.load(frame_file)
        img_pos = img_position[frame_number]
        img_orient = img_orientation[frame_number]

        unique_labels = np.unique(combined_data[..., 0])
        unique_labels = unique_labels[unique_labels > 0]

        corrected_camera_matrix = correct_camera_matrix(camera_matrix, 3840, 2160)

        depth_scale, all_data = find_correct_depth_scale(
            mesh, combined_data, corrected_camera_matrix, img_pos, img_orient, unique_labels
        )

        if depth_scale:  # Save data only if depth_scale is valid
            for label, data in all_data.items():
                if label not in all_label_points:
                    all_label_points[label] = {"class_id": class_id, "points": []}
                all_label_points[label]["points"].extend(data["scaled_points"])
        else:
            print(f"Skipping frame {frame_number} due to poor intersection ratio.")

    # Save all label points to file
    output_dir = f'../Data/{space_name}/castresults'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{class_id}_label_points.npy")
    np.save(output_path, all_label_points)
    print(f"Saved all label points for class {class_id} to {output_path}")

    # Uncomment below to enable visualization
    # visualize_all_frames(mesh, all_label_points)