import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
import argparse
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 🔥 **初始化 PyTorch 设备**
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    # 使用 bfloat16 精度
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # 对于 Ampere 及以上 GPU，启用 `TF32`
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def load_video_frames(video_dir):
    """
    加载帧文件名并按索引排序
    :param frames_dir: 帧存储目录
    :return: 排序后的帧文件名列表
    """
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names

def build_index_to_frame_id(frame_names):
    frame_ids = [int(os.path.splitext(name)[0]) for name in frame_names]
    return {i: frame_id for i, frame_id in enumerate(frame_ids)}

def load_mask_from_csv(mask_csv_path):
    """
    从 CSV 文件中加载掩码，并转换为 PyTorch Tensor 格式的二值掩码。
    """
    data = np.genfromtxt(mask_csv_path, delimiter=',', skip_header=2)

    with open(mask_csv_path, 'r') as file:
        lines = file.readlines()
        width, height = map(int, lines[1].strip().split(','))  # 读取掩码尺寸

    mask = data.reshape((height, width))
    mask_boolean = mask.astype(bool)
    mask_tensor = torch.tensor(mask_boolean, dtype=torch.bool)

    return mask_tensor, height, width

def initialize_predictor(model_cfg, sam2_checkpoint, video_path):
    """
    初始化推理状态
    :param predictor: 分割模型预测器
    :param frames_dir: 帧存储目录
    :return: 推理状态
    """
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)
    return predictor, inference_state


def add_new_mask_to_predictor(predictor, inference_state, frame_idx, obj_id, mask_tensor):
    """
    添加掩码到指定帧
    :param predictor: 分割模型预测器
    :param inference_state: 推理状态
    :param mask_path: 掩码路径
    :param ann_frame_idx: 帧索引
    :param ann_obj_id: 对象 ID
    :return: 输出对象 ID 和掩码 logits
    """
    """ 在 `SAM2` 预测器中添加新掩码 """
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        mask=mask_tensor,
    )
    return out_obj_ids, out_mask_logits


def propagate_masks(predictor, inference_state):
    """
    在视频帧中传播掩码
    :param predictor: 分割模型预测器
    :param inference_state: 推理状态
    :return: 包含每帧分割结果的字典
    """
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments

def downsample_mask(mask, target_count=800):
    """ 对目标掩码进行降采样，最多保留 `target_count` 个像素点 """
    non_zero_coords = np.column_stack(np.where(mask > 0))
    
    if len(non_zero_coords) > target_count:
        sampled_indices = np.linspace(0, len(non_zero_coords) - 1, target_count, dtype=int)
        sampled_coords = non_zero_coords[sampled_indices]
    else:
        sampled_coords = non_zero_coords  # 如果目标少于 800，则不采样

    # 创建新的降采样掩码
    new_mask = np.zeros_like(mask)
    for coord in sampled_coords:
        new_mask[tuple(coord)] = mask[tuple(coord)]  # 复制原来的标签值

    return new_mask

def save_filtered_masks(video_segments, index_to_frame_id, base_output_dir, class_id, target_count=800, max_saves=10):
    output_dir = os.path.join(base_output_dir, class_id)
    os.makedirs(output_dir, exist_ok=True)

    save_count = 0
    started = False
    consecutive = True

    frame_items = list(video_segments.items())

    for idx, (frame_idx, masks) in enumerate(frame_items):
        mask = list(masks.values())[0]
        if len(mask.shape) == 3:
            mask = mask.squeeze(0)
        mask = np.array(mask, dtype=np.uint8)

        if np.any(mask > 0):
            if not consecutive:
                print(f"Stopped at frame index {frame_idx} (not contiguous).")
                break

            mask = downsample_mask(mask, target_count)
            original_frame_id = index_to_frame_id[frame_idx]
            frame_path = os.path.join(output_dir, f"seg_{original_frame_id:04d}.npy")
            np.save(frame_path, mask)
            print(f"Saved: {frame_path} (Downsampled to {target_count} pixels)")

            save_count += 1
            started = True

            if save_count >= max_saves:
                break
        else:
            print(f"Skipped: Frame index {frame_idx} (only background)")
            if started:
                consecutive = False

    print(f"Saved {save_count} valid masks into {output_dir}")



def process_video(video_dir, mask_csv, model_cfg, checkpoint, class_id, ann_frame_id, base_output_dir):
    """
    运行完整的视频掩码处理流程：
    1. 初始化 SAM2 预测器
    2. 读取视频帧并构建映射
    3. 将原始帧编号映射为 index
    4. 添加新掩码
    5. 传播掩码
    6. 保存有目标的掩码（使用原始编号命名）
    """
    print("Initializing predictor...")
    predictor, inference_state = initialize_predictor(model_cfg, checkpoint, video_dir)

    print("Loading video frames...")
    frame_names = load_video_frames(video_dir)
    index_to_frame_id = build_index_to_frame_id(frame_names)
    frame_id_to_index = {v: k for k, v in index_to_frame_id.items()}  # 反向映射

    # 🚨 将原始帧编号（如 824）转换为 index
    if ann_frame_id not in frame_id_to_index:
        raise ValueError(f"Frame ID {ann_frame_id} not found in loaded frames.")
    ann_frame_idx = frame_id_to_index[ann_frame_id]

    print(f"Loading mask from CSV: {mask_csv}")
    mask_tensor, height, width = load_mask_from_csv(mask_csv)

    ann_obj_id = 1  # 统一 obj_id
    print(f"Adding new mask at frame index {ann_frame_idx} (original frame {ann_frame_id}) for class_id {class_id}...")
    add_new_mask_to_predictor(predictor, inference_state, ann_frame_idx, ann_obj_id, mask_tensor)

    print("Propagating masks across video...")
    video_segments = propagate_masks(predictor, inference_state)

    print(f"Saving filtered masks for {class_id}...")
    save_filtered_masks(video_segments, index_to_frame_id, base_output_dir, class_id, max_saves=10)

    print("✅ Processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Mask Processing")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to video frames directory")
    parser.add_argument("--mask_csv", type=str, required=True, help="Path to CSV mask file")
    parser.add_argument("--model_cfg", type=str, required=True, help="Model configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--class_id", type=str, required=True, help="Class ID for object segmentation")
    parser.add_argument("--ann_frame_idx", type=int, required=True, help="Original frame number ")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory for results")

    args = parser.parse_args()

    process_video(
    video_dir=args.video_dir,
    mask_csv=args.mask_csv,
    model_cfg=args.model_cfg,
    checkpoint=args.checkpoint,
    class_id=args.class_id,
    ann_frame_id=args.ann_frame_idx,  # 👈 改这里
    base_output_dir=args.output_dir,
)


#if __name__ == "__main__":
    # 配置路径和参数
    #video_dir = "../Data/frames/OceanTeachingBuilding"  # 视频帧目录
    #mask_csv = "../Data/mask/OceanTeachingBuilding/mask.csv"
    #model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    #checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    #class_id = "test"  # `class_id` 既是类别 ID，也是 `ann_obj_id`
    #ann_frame_idx = 32  # 指定初始帧
    #output_dir="../Data/results/OceanTeachingBuilding"

    #process_video(video_dir, mask_csv, model_cfg, checkpoint, class_id, ann_frame_idx, output_dir)

