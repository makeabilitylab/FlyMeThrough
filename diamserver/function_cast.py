import os
import re
import numpy as np
import pyvista as pv
import xml.etree.ElementTree as ET
import glob


# 从XML中提取相机数据
def parse_camera_data(xml_path):
    img_position = {}
    img_orientation = {}
    camera_matrix = None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    sensor = root.find('.//sensor')
    if sensor is not None:
        calibration = sensor.find('calibration')
        if calibration is not None:
            f = float(calibration.find('f').text)
            cx = float(calibration.find('cx').text)
            cy = float(calibration.find('cy').text)
            camera_matrix = np.array([
                [f, 0.0, cx],
                [0.0, f, cy],
                [0.0, 0.0, 1.0]
            ])

    for camera in root.findall('.//camera'):
        label = camera.get('label')
        frame_number = int(label)
        #if label and 'frame_' in label:
            #frame_number = int(label.split('_')[1])
        component_id = camera.get('component_id')
        if component_id == '0':  # 只处理component_id为0的相机
            transform_element = camera.find('transform')
            if transform_element is not None:
                transform = np.fromstring(transform_element.text, sep=' ').reshape(4, 4)
                img_position[frame_number] = transform[:3, 3]
                img_orientation[frame_number] = transform[:3, :3]

    return img_position, img_orientation, camera_matrix


def read_mesh(filepath):
    print("Loading mesh from:", filepath)
    mesh = pv.read(filepath)
    mesh.compute_normals()
    print("Normals computed.")
    return mesh


# 使用图像宽高计算正确的相机矩阵
def calculate_correct_camera_matrix(camera_matrix, image_width, image_height):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    offset_cx = camera_matrix[0, 2]
    offset_cy = camera_matrix[1, 2]

    # 计算图像中心
    cx = (image_width / 2) + offset_cx
    cy = (image_height / 2) + offset_cy

    # 构建相机矩阵
    corrected_camera_matrix = np.array([[fx, 0.0, cx],
                                        [0.0, fy, cy],
                                        [0.0, 0.0, 1.0]], dtype=np.float32)
    return corrected_camera_matrix


# 将分割和深度信息转换为三维点云
def pixel_to_3d_coordinates(combined_data, corrected_camera_matrix, img_pos, img_orient, label, depth_scale=1):
    fx, fy = corrected_camera_matrix[0, 0], corrected_camera_matrix[1, 1]
    cx, cy = corrected_camera_matrix[0, 2], corrected_camera_matrix[1, 2]
    coords_3d = []

    # 提取对应类别的像素坐标和深度（仅限于分割区域）
    label_pixels = np.column_stack(np.where(combined_data[..., 0] == label))

    for (py, px) in label_pixels:
        Z = combined_data[py, px, 1] * depth_scale  # 根据depth_scale调整深度信息
        X = (px - cx) * Z / fx
        Y = (py - cy) * Z / fy
        point_camera = np.array([X, Y, Z])
        point_world = img_orient @ point_camera + img_pos
        coords_3d.append(point_world)

    return np.array(coords_3d)


# 使用射线追踪优化深度比例
def find_correct_depth_scale(mesh, combined_data, corrected_camera_matrix, img_pos, img_orient, labels, 
                             threshold=0.17, scale_step=1.1, max_iterations=100, stop_threshold=0.01, bad_frame_limit=50,
                             max_depth_scale=25.0):
    depth_scale = 1.01  # 初始深度比例
    all_data = {label: {"original_points": None, "scaled_points": None, "intersections": []} for label in labels}
    
    prev_intersection_ratios = []  # 存储过去几轮的 intersection_ratio
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

        # 记录最近 bad_frame_limit 轮的交点比率
        prev_intersection_ratios.append(intersection_ratio)
        if len(prev_intersection_ratios) > bad_frame_limit:
            prev_intersection_ratios.pop(0)  # 只保留最近 bad_frame_limit 轮的记录
        
        # 判断是否需要跳过该帧（变化太小）
        if len(prev_intersection_ratios) == bad_frame_limit:
            max_change = max(prev_intersection_ratios) - min(prev_intersection_ratios)
            if max_change < stop_threshold:  # 变化小于 stop_threshold（默认 0.01）
                print(f"Frame has poor performance, skipping. No significant improvement after {bad_frame_limit} iterations.")
                return None, None  # 直接返回 None，表示该帧无效
        
        # 检查是否达到阈值
        if intersection_ratio >= threshold:
            print(f"Found correct depth scale: {depth_scale:.3f}")
            return depth_scale, all_data
        
        # 继续调整 depth scale
        depth_scale *= scale_step

    print("No valid depth scale found within max iterations.")
    return None, None  # 该帧没有找到合适的 depth scale



# 可视化功能
def visualize_all_frames(mesh, all_label_points):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="lightgray", opacity=0.5, label="3D Mesh")

    colors = ["blue", "green", "yellow", "purple", "orange", "cyan"]
    for idx, (label, data) in enumerate(all_label_points.items()):
        color = colors[idx % len(colors)]
        points = np.array(data["points"])  # 读取嵌套字典中的 "points" 键
        if points.size > 0:
            plotter.add_points(points, color=color, point_size=5, label=f"Label {label}")

    plotter.add_legend()
    plotter.show()

   

def process_depth_segmentation(space_name, object_id):
    """
    处理指定空间和对象 ID 的深度和分割数据，并进行点云计算和可视化。
    
    参数：
    space_name (str): 处理的空间名称。
    object_id (str): 目标对象的 ID。
    """
    xml_path = f'../Data/{space_name}/{space_name}.xml'
    segdep_folder = f'../Data/{space_name}/segdep/{object_id}'
    model_path = f'../Data/{space_name}/testnocolor.obj'
    class_id = object_id  # 指定类别 ID
    
    img_position, img_orientation, camera_matrix = parse_camera_data(xml_path)
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

        corrected_camera_matrix = calculate_correct_camera_matrix(camera_matrix, 3840, 2160)

        depth_scale, all_data = find_correct_depth_scale(
            mesh, combined_data, corrected_camera_matrix, img_pos, img_orient, unique_labels
        )

        if depth_scale:  # 只有当 depth_scale 有效时，才保存数据
            for label, data in all_data.items():
                if label not in all_label_points:
                    all_label_points[label] = {"class_id": class_id, "points": []}
                all_label_points[label]["points"].extend(data["scaled_points"])
        else:
            print(f"Skipping frame {frame_number} due to poor intersection ratio.")

    # 保存所有点数据
    output_dir = f'../Data/{space_name}/castresults'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{class_id}_label_points.npy")
    np.save(output_path, all_label_points)
    print(f"Saved all label points for class {class_id} to {output_path}")
    
    # visualize_all_frames(mesh, all_label_points)
