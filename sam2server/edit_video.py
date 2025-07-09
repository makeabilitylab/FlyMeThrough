import cv2
import os

def extract_frames(video_path, output_dir, fps):
    """
    从指定视频中抽取帧并保存到指定目录。

    参数：
        video_path (str): 视频文件路径。
        output_dir (str): 输出帧保存的目录。
        fps (int): 每秒要抽取的帧数。

    返回：
        int: 抽取并保存的帧的总数量。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 frame_interval 帧保存一帧
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f'{extracted_count:05d}.jpg')
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"从视频 {video_path} 抽取了 {extracted_count} 帧并保存到 {output_dir}")
    return extracted_count

# 使用示例
if __name__ == "__main__":
    video_path = '../Data/videos/OceanTeachingBuilding.MP4'  # 视频文件路径
    output_dir = '../Data/frames/OceanTeachingBuilding'           # 输出的帧保存路径
    fps = 2                                # 每秒要抽取的帧数

    total_frames = extract_frames(video_path, output_dir, fps)
    print(f"总共抽取了 {total_frames} 帧。")

