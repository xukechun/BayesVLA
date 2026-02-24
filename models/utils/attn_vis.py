import numpy as np
import cv2
from matplotlib.cm import get_cmap

def attn_vis(attn_weights, image, patch_size=16, cmap_name='jet', alpha=0.5, win_name="attn_vis", show=True):
    """
    可视化 attention 权重并 overlay 到原图上。
    
    参数:
    - attn_weights: shape (256,), 即 16x16 attention token 权重
    - image: 原始图像 (256, 256, 3)，像素范围 [0, 255]
    - patch_size: 每个 patch 的边长
    - cmap_name: matplotlib colormap, 例如 'jet', 'viridis'
    - alpha: heatmap 与原图的融合强度
    """

    # 检查维度
    assert attn_weights.shape[0] == (image.shape[0] // patch_size) ** 2, "attention shape mismatch"
    
    # 归一化 attention 权重并 reshape 为 heatmap
    attn_map = attn_weights.reshape(image.shape[0] // patch_size, image.shape[1] // patch_size)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    # print(attn_map)

    # 插值放大到原图大小
    attn_map_resized = cv2.resize(attn_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 获取颜色映射（例如 jet colormap）
    cmap = get_cmap(cmap_name)
    attn_color = cmap(attn_map_resized)[..., :3]  # shape: (256, 256, 3)，RGB
    attn_color = np.ascontiguousarray(attn_color[:, :, ::-1]) # bgr
    attn_color = (attn_color * 255).astype(np.uint8)

    # 如果原图是灰度图，转换成 3 通道
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # 转换为 uint8
    image = image.astype(np.uint8)
    image = np.ascontiguousarray(image[:, :, ::-1])

    # 叠加 heatmap 到原图
    blended = cv2.addWeighted(image, 1 - alpha, attn_color, alpha, 0)

    # 可视化
    if show:
        cv2.imshow(win_name, blended)
        key = cv2.waitKey(1)
    
    return blended
