import os
import json
import numpy as np
import argparse

def get_bounding_box_ratio(mask):
    """获取二值mask的bounding box，并转换为比例格式 [x1, y1, x2, y2]"""
    mask_indices = np.argwhere(mask > 0)
    if mask_indices.size == 0:
        return None  # 没有目标区域
    
    y_min, x_min = mask_indices.min(axis=0)
    y_max, x_max = mask_indices.max(axis=0)

    height, width = mask.shape  # 获取 mask 的尺寸

    # 归一化到 0-1 范围
    x1, y1 = x_min / width, y_min / height
    x2, y2 = x_max / width, y_max / height

    return [x1, y1, x2, y2]

def process_mask_files(space_name, object_id, description):
    """遍历指定路径下的所有 .npy 文件，计算 bounding box，并输出 JSON"""
    mask_path = f"../Data/results/{space_name}/{object_id}/"
    
    results = []
    
    if not os.path.exists(mask_path):
        print(f"目录 {mask_path} 不存在！")
        return
    
    for file_name in sorted(os.listdir(mask_path)):  # 按名称排序
        if file_name.endswith(".npy"):
            file_path = os.path.join(mask_path, file_name)
            mask = np.load(file_path)  # 加载 mask 数据

            bounding_box_ratio = get_bounding_box_ratio(mask)
            if bounding_box_ratio:
                # 解析 frame 名称
                frame_id = file_name.replace("seg_", "").replace(".npy", "")
                frame_name = f"frame_{frame_id}.jpg"
                
                results.append({
                    "frame": frame_name,
                    "bbox": bounding_box_ratio
                })
    
    # 生成 JSON 结构
    output_data = {
        "space_name": space_name,
        "object_id": object_id,
        "results": results,
        "description": description
    }
    
    # 保存为 JSON 文件
    output_json_path = f"../Data/results/{space_name}/{object_id}/bbox_results.json"
    with open(output_json_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)
    
    print(f"Bounding box 结果已保存至: {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert segmentation masks to bounding boxes")
    parser.add_argument("--space_name", type=str, required=True, help="Name of the space")
    parser.add_argument("--object_id", type=str, required=True, help="Object ID")
    parser.add_argument("--description", type=str, required=True, help="Task description")  # 新增参数

    args = parser.parse_args()

    process_mask_files(args.space_name, args.object_id, args.description)

# 运行代码
#space_name = "OceanTeachingBuilding"
#bject_id = "test"
#process_mask_files(space_name, object_id)
