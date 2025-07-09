import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
import argparse

# Enable fallback for PyTorch MPS backend
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 🔥 Initialize PyTorch device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    # Use bfloat16 precision
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # Enable TF32 for Ampere and later GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

elif device.type == "mps":
    print(
        "\n⚠️ Support for MPS devices is experimental. SAM 2 is trained with CUDA and might "
        "produce numerically different outputs and sometimes degraded performance on MPS. "
        "See: https://github.com/pytorch/pytorch/issues/84936 for details."
    )


def load_video_frames(video_dir):
    """
    Load frame filenames from a directory and sort them by index.

    Args:
        video_dir (str): Directory where video frames are stored.

    Returns:
        list of str: Sorted list of frame filenames.
    """
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names


def build_index_to_frame_id(frame_names):
    """
    Build a mapping from sequential index to frame ID (integer).

    Args:
        frame_names (list of str): List of frame filenames.

    Returns:
        dict: Mapping {index: frame_id}.
    """
    frame_ids = [int(os.path.splitext(name)[0]) for name in frame_names]
    return {i: frame_id for i, frame_id in enumerate(frame_ids)}


def load_mask_from_csv(mask_csv_path):
    """
    Load a mask from a CSV file and convert it into a binary PyTorch tensor.

    Args:
        mask_csv_path (str): Path to the CSV file containing the mask.

    Returns:
        tuple:
            torch.Tensor: Boolean mask tensor of shape (H, W).
            int: Height of the mask.
            int: Width of the mask.
    """
    # load mask data (skip first two header rows)
    data = np.genfromtxt(mask_csv_path, delimiter=',', skip_header=2)

    with open(mask_csv_path, 'r') as file:
        lines = file.readlines()
        # read mask dimensions from the second line
        width, height = map(int, lines[1].strip().split(','))

    mask = data.reshape((height, width))
    mask_boolean = mask.astype(bool)
    mask_tensor = torch.tensor(mask_boolean, dtype=torch.bool)

    return mask_tensor, height, width


def initialize_predictor(model_cfg, sam2_checkpoint, video_path):
    """
    Initialize the SAM2 predictor and inference state.

    Args:
        model_cfg (str): Path to model configuration file.
        sam2_checkpoint (str): Path to model checkpoint.
        video_path (str): Path to the video frames directory or video file.

    Returns:
        tuple:
            predictor: Initialized SAM2 predictor.
            inference_state: Predictor's inference state.
    """
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)
    return predictor, inference_state


def add_new_mask_to_predictor(predictor, inference_state, frame_idx, obj_id, mask_tensor):
    """
    Add a mask to the predictor at the specified frame and object ID.

    Args:
        predictor: SAM2 predictor.
        inference_state: Current inference state.
        frame_idx (int): Frame index where the mask is added.
        obj_id (str): Object ID.
        mask_tensor (torch.Tensor): Boolean mask tensor of shape (H, W).

    Returns:
        tuple:
            out_obj_ids: List of object IDs after addition.
            out_mask_logits: List of mask logits after addition.
    """
    # Add a new mask in the SAM2 predictor
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        mask=mask_tensor,
    )
    return out_obj_ids, out_mask_logits


def propagate_masks(predictor, inference_state):
    """
    Propagate masks throughout the video frames.

    Args:
        predictor: SAM2 predictor.
        inference_state: Current inference state.

    Returns:
        dict: Mapping of frame index to a dict {object_id: binary mask array}.
    """
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments


def downsample_mask(mask, target_count=800):
    """
    Downsample the target mask to at most `target_count` non-zero pixels.

    Args:
        mask (np.ndarray): Input mask.
        target_count (int): Maximum number of pixels to keep.

    Returns:
        np.ndarray: New mask with at most `target_count` non-zero pixels.
    """
    non_zero_coords = np.column_stack(np.where(mask > 0))

    if len(non_zero_coords) > target_count:
        sampled_indices = np.linspace(0, len(non_zero_coords) - 1, target_count, dtype=int)
        sampled_coords = non_zero_coords[sampled_indices]
    else:
        # If there are fewer than `target_count` pixels, keep all
        sampled_coords = non_zero_coords

    # Create a new downsampled mask
    new_mask = np.zeros_like(mask)
    for coord in sampled_coords:
        new_mask[tuple(coord)] = mask[tuple(coord)]  # preserve original label value

    return new_mask


def save_filtered_masks(video_segments, index_to_frame_id, base_output_dir, class_id, target_count=800, max_saves=10):
    """
    Save filtered masks from video segments to disk, downsampled and limited to max_saves.

    Args:
        video_segments (dict): Segmentation results from propagate_masks().
        index_to_frame_id (dict): Mapping from frame index to frame ID.
        base_output_dir (str): Base directory to save masks.
        class_id (str): Object/class ID.
        target_count (int): Max number of pixels per mask.
        max_saves (int): Max number of masks to save.

    Returns:
        None
    """
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
    Run the complete video mask processing pipeline:
    1. Initialize SAM2 predictor
    2. Load video frames and build mappings
    3. Map the original frame ID to index
    4. Add the initial mask
    5. Propagate the mask
    6. Save the masks containing objects (named by original frame IDs)

    Args:
        video_dir (str): Directory containing video frames.
        mask_csv (str): Path to the CSV mask file.
        model_cfg (str): Path to the SAM2 model config file.
        checkpoint (str): Path to the SAM2 model checkpoint.
        class_id (str): Class ID for the object being segmented.
        ann_frame_id (int): Original frame ID where annotation starts.
        base_output_dir (str): Base directory where output masks will be saved.

    Returns:
        None
    """
    print("Initializing predictor...")
    predictor, inference_state = initialize_predictor(model_cfg, checkpoint, video_dir)

    print("Loading video frames...")
    frame_names = load_video_frames(video_dir)
    index_to_frame_id = build_index_to_frame_id(frame_names)
    frame_id_to_index = {v: k for k, v in index_to_frame_id.items()}  # reverse mapping

    # 🚨 Map original frame ID (e.g., 824) to internal index
    if ann_frame_id not in frame_id_to_index:
        raise ValueError(f"Frame ID {ann_frame_id} not found in loaded frames.")
    ann_frame_idx = frame_id_to_index[ann_frame_id]

    print(f"Loading mask from CSV: {mask_csv}")
    mask_tensor, height, width = load_mask_from_csv(mask_csv)

    ann_obj_id = 1  # use a fixed object ID
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
    ann_frame_id=args.ann_frame_idx,  
    base_output_dir=args.output_dir,
)


