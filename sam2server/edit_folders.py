import os

def generate_output_paths(video_path):
    """
    根据视频文件路径生成对应的输出目录。

    参数：
        video_path (str): 视频文件路径。

    返回：
        dict: 包含不同类型输出路径的字典。
    """
    # 获取视频文件名（不带扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 基础目录（与视频路径同级的 `../Data`）
    base_dir = os.path.abspath(os.path.join(os.path.dirname(video_path), ".."))

    # 定义各类输出路径
    paths = {
        "frames_dir": os.path.join(base_dir, "frames", video_name),
        "mask_dir": os.path.join(base_dir, "mask", video_name),
        "video_masks_dir": os.path.join(base_dir, "mask", video_name)
    }

    # 确保所有目录存在
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)

    return paths

if __name__ == "__main__":
    # 已知视频路径
    video_path = '../Data/videos/OceanTeachingBuilding.MP4'

    # 自动生成目录
    output_paths = generate_output_paths(video_path)

    # 打印生成的路径
    print("生成的路径：")
    for key, path in output_paths.items():
        print(f"{key}: {path}")