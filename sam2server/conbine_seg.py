import numpy as np
import os

# 掩码文件存放目录
mask_dir = "./videos/results/door1"

# 获取所有掩码文件
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".npy")])

# 创建一个空的最终掩码，初始化为 0（背景）
height, width = np.load(os.path.join(mask_dir, mask_files[0])).shape  # 读取第一个掩码尺寸
final_mask = np.zeros((height, width), dtype=np.uint8)

# 遍历所有掩码文件，并自动编号
for idx, file_name in enumerate(mask_files, start=1):  # ID 从 1 开始
    mask_path = os.path.join(mask_dir, file_name)
    mask = np.load(mask_path)  # 读取掩码

    # 只更新掩码像素为1的区域，防止覆盖其他对象
    final_mask[mask > 0] = idx  

print(f"Final mask contains {len(mask_files)} objects.")

# 保存合并后的掩码
np.save(os.path.join(mask_dir, "seg_combined.npy"), final_mask)


import matplotlib.pyplot as plt

# 读取合并后的掩码
combined_mask = np.load(os.path.join(mask_dir, "seg_combined.npy"))

plt.figure(figsize=(8, 6))
plt.imshow(combined_mask, cmap="jet")  # 用不同颜色显示不同 ID
plt.colorbar(label="Object ID")
plt.title("Final Segmentation Mask")
plt.show()
