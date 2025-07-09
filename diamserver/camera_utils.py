import numpy as np
import xml.etree.ElementTree as ET

def load_camera_parameters_from_xml(xml_path):
    """
    Load camera parameters (positions, orientations, intrinsic matrix) from an XML file.

    The XML is expected to contain:
    - <sensor> section with calibration parameters: focal length (f), offsets (cx, cy)
    - multiple <camera> nodes with a 4×4 transform matrix and component_id

    Only cameras with `component_id == '0'` are considered.

    Args:
        xml_path (str): Path to the XML file.

    Returns:
        tuple:
            img_position (dict[int, np.ndarray]): mapping frame_number → 3D position (3,)
            img_orientation (dict[int, np.ndarray]): mapping frame_number → 3×3 rotation matrix
            camera_matrix (np.ndarray): 3×3 intrinsic camera matrix
    """
    img_position = {}
    img_orientation = {}
    camera_matrix = None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Parse intrinsic parameters
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
            ], dtype=np.float32)

    # Parse per-frame extrinsic parameters
    for camera in root.findall('.//camera'):
        label = camera.get('label')
        frame_number = int(label)

        component_id = camera.get('component_id')
        if component_id == '0':  # Only handle cameras with component_id == 0
            transform_element = camera.find('transform')
            if transform_element is not None:
                transform = np.fromstring(transform_element.text, sep=' ').reshape(4, 4)
                img_position[frame_number] = transform[:3, 3]
                img_orientation[frame_number] = transform[:3, :3]

    return img_position, img_orientation, camera_matrix


def correct_camera_matrix(camera_matrix, image_width, image_height):
    """
    Correct a camera intrinsic matrix if the cx/cy in the XML are offsets from the image center
    instead of absolute pixel coordinates.

    This is usually needed if the dataset you use records cx/cy incorrectly.

    Args:
        camera_matrix (np.ndarray): 3×3 intrinsic camera matrix from XML.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.

    Returns:
        np.ndarray: Corrected 3×3 camera intrinsic matrix.
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    offset_cx = camera_matrix[0, 2]
    offset_cy = camera_matrix[1, 2]

    # adjust principal point assuming cx/cy are relative offsets
    cx = (image_width / 2) + offset_cx
    cy = (image_height / 2) + offset_cy

    corrected = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return corrected
