import os
import json
import numpy as np
import argparse

def get_bounding_box_ratio(mask):
    """
    Compute the bounding box of a binary mask and convert it to normalized coordinates [x1, y1, x2, y2].

    Args:
        mask (np.ndarray): Binary mask of shape (H, W), where non-zero pixels belong to the object.

    Returns:
        list or None: Normalized bounding box [x1, y1, x2, y2] if the mask has non-zero area,
                      otherwise None.
    """
    mask_indices = np.argwhere(mask > 0)
    if mask_indices.size == 0:
        return None  # No object region present

    y_min, x_min = mask_indices.min(axis=0)
    y_max, x_max = mask_indices.max(axis=0)

    height, width = mask.shape  # get mask dimensions

    # Normalize coordinates to the [0, 1] range
    x1, y1 = x_min / width, y_min / height
    x2, y2 = x_max / width, y_max / height

    return [x1, y1, x2, y2]


def process_mask_files(space_name, object_id, description):
    """
    Traverse all .npy mask files for a given space and object,
    compute bounding boxes, and save results as a JSON file.

    Args:
        space_name (str): Name of the space.
        object_id (str): Object ID.
        description (str): Task description.

    Returns:
        None
    """
    mask_path = f"../Data/results/{space_name}/{object_id}/"
    results = []

    if not os.path.exists(mask_path):
        print(f"Directory {mask_path} does not exist!")
        return

    # Iterate over mask files sorted by name
    for file_name in sorted(os.listdir(mask_path)):
        if file_name.endswith(".npy"):
            file_path = os.path.join(mask_path, file_name)
            mask = np.load(file_path)  # load mask data

            bounding_box_ratio = get_bounding_box_ratio(mask)
            if bounding_box_ratio:
                # parse frame name
                frame_id = file_name.replace("seg_", "").replace(".npy", "")
                frame_name = f"frame_{frame_id}.jpg"

                results.append({
                    "frame": frame_name,
                    "bbox": bounding_box_ratio
                })

    # Construct output JSON structure
    output_data = {
        "space_name": space_name,
        "object_id": object_id,
        "results": results,
        "description": description
    }

    # Save to JSON file
    output_json_path = f"../Data/results/{space_name}/{object_id}/bbox_results.json"
    with open(output_json_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Bounding box results saved to: {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert segmentation masks to bounding boxes")
    parser.add_argument("--space_name", type=str, required=True, help="Name of the space")
    parser.add_argument("--object_id", type=str, required=True, help="Object ID")
    parser.add_argument("--description", type=str, required=True, help="Task description")

    args = parser.parse_args()

    process_mask_files(args.space_name, args.object_id, args.description)
