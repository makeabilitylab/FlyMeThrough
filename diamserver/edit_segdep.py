import os
import numpy as np

def combine_segmentation_and_depth(depth_folder, segmentation_folder, output_folder):
    """
    将分割数据和深度数据合并，并保存到指定的输出文件夹。
    
    参数：
    depth_folder (str): 存储深度数据的文件夹路径。
    segmentation_folder (str): 存储分割数据的文件夹路径。
    output_folder (str): 结果输出文件夹路径。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有分割文件
    seg_files = [f for f in os.listdir(segmentation_folder) if f.startswith("seg_") and f.endswith('.npy')]

    # 遍历分割文件
    for seg_file in seg_files:
        # 提取编号（例如 seg__0003.npy -> 0003）
        frame_number = seg_file.split('_')[-1].split('.')[0]

        # 构造对应的深度文件名，例如 frame_0003.npz
        depth_file = f"frame_{frame_number}.npz"
        depth_path = os.path.join(depth_folder, depth_file)

        # 检查对应的深度文件是否存在
        if not os.path.exists(depth_path):
            print(f"Depth file {depth_path} not found for {seg_file}. Skipping...")
            continue

        # 加载深度和分割数据
        seg_path = os.path.join(segmentation_folder, seg_file)
        segmentation_data = np.load(seg_path)
        depth_data = np.load(depth_path)['depth']

        # 确保数据尺寸匹配
        if segmentation_data.shape[:2] != depth_data.shape[:2]:
            print(f"Shape mismatch for {seg_file} and {depth_file}. Skipping...")
            continue

        # 合并数据
        combined_data = np.stack((segmentation_data, depth_data), axis=-1)

        # 保存新数据
        output_path = os.path.join(output_folder, f"frame_{frame_number}_depth_segmentation.npy")
        np.save(output_path, combined_data)
        print(f"Saved combined data to {output_path}")

    print("Processing complete.")