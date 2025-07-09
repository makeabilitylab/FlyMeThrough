import os
import numpy as np

def combine_segmentation_and_depth(depth_folder, segmentation_folder, output_folder):
    """
    Combine segmentation data and depth data, and save the result to the specified output folder.

    Args:
        depth_folder (str): Path to the folder containing depth data (.npz files).
        segmentation_folder (str): Path to the folder containing segmentation data (.npy files).
        output_folder (str): Path to the folder where combined results will be saved.
    """
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all segmentation files in the folder
    seg_files = [
        f for f in os.listdir(segmentation_folder)
        if f.startswith("seg__") and f.endswith('.npy')
    ]

    # Iterate over segmentation files
    for seg_file in seg_files:
        # Extract frame number (e.g., seg__00003.npy -> 00003)
        frame_number = seg_file.replace("seg__", "").replace(".npy", "")

        # Construct the corresponding depth file name (e.g., 00003.npz)
        depth_file = f"{frame_number}.npz"
        depth_path = os.path.join(depth_folder, depth_file)

        # Check if the depth file exists
        if not os.path.exists(depth_path):
            print(f"Depth file {depth_path} not found for {seg_file}. Skipping...")
            continue

        # Load segmentation and depth data
        seg_path = os.path.join(segmentation_folder, seg_file)
        segmentation_data = np.load(seg_path)
        depth_data = np.load(depth_path)['depth']

        # Make sure the dimensions match
        if segmentation_data.shape[:2] != depth_data.shape[:2]:
            print(f"Shape mismatch between {seg_file} and {depth_file}. Skipping...")
            continue

        # Combine the data along a new axis
        combined_data = np.stack((segmentation_data, depth_data), axis=-1)

        # Save the combined data
        output_path = os.path.join(output_folder, f"{frame_number}_depth_segmentation.npy")
        np.save(output_path, combined_data)
        print(f"Saved combined data to {output_path}")

    print("Processing complete.")
