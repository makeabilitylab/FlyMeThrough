import os
from PIL import Image
import depth_pro  # ensure this module is installed and available
import numpy as np
import matplotlib.pyplot as plt

# Input and output folder paths
space_name = "scene_example"  # replace with your actual space name
input_folder = os.path.join("..", "Data", space_name, "frame")
output_folder = os.path.join("..", "Data", space_name, "depth")

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the model and preprocessing transforms
model, transform = depth_pro.create_model_and_transforms()
model.eval()  # set model to inference mode

# Iterate over all image files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # check file extension
        input_path = os.path.join(input_folder, filename).replace("\\", "/")
        print(f"Processing: {input_path}")
        
        # Load and preprocess the image
        image, _, f_px = depth_pro.load_rgb(input_path)
        image = transform(image)
        
        # Run inference
        prediction = model.infer(image, f_px=f_px)
        
        # Extract results
        depth = prediction["depth"]             # depth map (in meters)
        focallength_px = prediction["focallength_px"]  # focal length (in pixels)
        
        # Convert depth to NumPy array
        depth_np = depth.cpu().numpy()
        
        # Prepare output filenames
        base_name = os.path.splitext(filename)[0]
        npz_path = os.path.join(output_folder, f"{base_name}.npz").replace("\\", "/")
        png_path = os.path.join(output_folder, f"{base_name}_depth.png").replace("\\", "/")
        
        # Save depth and focal length to .npz file
        np.savez(npz_path, depth=depth_np, focallength_px=focallength_px.cpu().numpy())
        print(f"Saved depth map and focal length as {npz_path}")
        
        # Save depth map as pseudo-color PNG
        plt.imshow(depth_np, cmap='plasma')
        plt.colorbar(label="Depth (m)")
        plt.title(f"Depth Map: {base_name}")
        plt.savefig(png_path)
        plt.close()
        print(f"Saved depth map image as {png_path}")

print("Processing complete.")
