import os
import json

def merge_bbox_results(space_name):
    """
    Merge all bounding box data from JSON files under the given space_name.
    
    Parameters:
        space_name (str): The name of the space to process.
    
    Returns:
        dict: A dictionary containing merged bounding box data.
    """
    base_path = f"../Data/{space_name}/bbox_results"
    merged_data = {space_name: {}}
    
    if not os.path.isdir(base_path):
        print(f"Error: Directory {base_path} does not exist.")
        return merged_data
    
    for file_name in os.listdir(base_path):
        if not file_name.endswith("bbox.json"):
            continue
        
        file_path = os.path.join(base_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                object_id = data.get("object_id")
                bounding_boxes = data.get("bounding_boxes", [])
                
                if object_id not in merged_data[space_name]:
                    merged_data[space_name][object_id] = []
                
                merged_data[space_name][object_id].extend(bounding_boxes)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse {file_path}")
    
    return merged_data

def save_merged_data(space_name):
    """
    Save merged bounding box data to a JSON file in the corresponding bbox_results directory.
    
    Parameters:
        space_name (str): The name of the space to process.
    """
    base_path = f"../Data/{space_name}/bbox_results"
    output_path = os.path.join(base_path, f"merged_{space_name}_bbox_results.json")
    
    merged_data = merge_bbox_results(space_name)
    
    # Overwrite existing merged file if it exists
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4)
    
    print(f"Merged JSON file saved to {output_path}")
    return output_path


