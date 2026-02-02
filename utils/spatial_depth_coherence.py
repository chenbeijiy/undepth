"""
创新1：空间-深度联合一致性损失（Spatial-Depth Coherence Loss）
不仅约束单条射线上高斯的深度一致性，还约束空间邻域内的深度一致性
"""

import torch
import torch.nn.functional as F


def spatial_depth_coherence_loss(surf_depth, rgb, lambda_spatial=0.1, kernel_size=3):
    """
    空间-深度联合一致性损失
    
    约束空间邻域内的深度平滑性，使用RGB相似性作为自适应权重
    
    Args:
        surf_depth: 表面深度图 (1, H, W) 或 (H, W)
        rgb: RGB图像 (3, H, W)
        lambda_spatial: RGB相似性权重系数（用于计算权重）
        kernel_size: 邻域大小（3或5）
    
    Returns:
        loss: 空间-深度一致性损失（标量）
    """
    # 确保输入格式正确
    if surf_depth.dim() == 3:
        surf_depth = surf_depth.squeeze(0)  # (H, W)
    if rgb.dim() == 3:
        rgb = rgb  # (3, H, W)
    else:
        raise ValueError(f"RGB should be (3, H, W), got {rgb.shape}")
    
    H, W = surf_depth.shape[-2:]
    
    # 准备输入：添加batch维度
    surf_depth_batch = surf_depth.unsqueeze(0)  # (1, 1, H, W)
    rgb_batch = rgb.unsqueeze(0)  # (1, 3, H, W)
    
    # 使用unfold提取邻域
    padding = kernel_size // 2
    rgb_unfold = F.unfold(rgb_batch, kernel_size=kernel_size, padding=padding)  # (1, 3*kernel_size^2, H*W)
    depth_unfold = F.unfold(surf_depth_batch, kernel_size=kernel_size, padding=padding)  # (1, kernel_size^2, H*W)
    
    # 重塑RGB unfold: (1, 3*kernel_size^2, H*W) -> (1, 3, kernel_size^2, H*W)
    rgb_unfold = rgb_unfold.view(1, 3, kernel_size * kernel_size, H * W)
    
    # 获取中心像素的RGB和深度
    rgb_center = rgb.view(3, H * W).unsqueeze(1)  # (3, 1, H*W) - 扩展维度以匹配邻域
    depth_center = surf_depth.view(1, H * W).unsqueeze(1)  # (1, 1, H*W) - 扩展维度以匹配邻域
    
    # 计算RGB差异：每个邻域像素与中心像素的RGB差异
    rgb_unfold_reshaped = rgb_unfold.squeeze(0)  # (3, kernel_size^2, H*W)
    # rgb_center: (3, 1, H*W), rgb_unfold_reshaped: (3, kernel_size^2, H*W)
    # 广播后相减: (3, kernel_size^2, H*W)
    rgb_diff = (rgb_unfold_reshaped - rgb_center).norm(dim=0)  # (kernel_size^2, H*W)
    
    # 计算RGB相似性权重：w = exp(-lambda * ||I(x) - I(x')||^2)
    rgb_weights = torch.exp(-lambda_spatial * rgb_diff)  # (kernel_size^2, H*W)
    
    # 计算深度差异：每个邻域像素与中心像素的深度差异
    depth_unfold_reshaped = depth_unfold.squeeze(0)  # (kernel_size^2, H*W)
    depth_center_reshaped = depth_center.squeeze(0)  # (1, H*W)
    depth_diff = (depth_unfold_reshaped - depth_center_reshaped).abs()  # (kernel_size^2, H*W)
    
    # 加权深度一致性损失：L = sum(w * |d(x) - d(x')|^2)
    weighted_depth_diff = rgb_weights * depth_diff.pow(2)  # (kernel_size^2, H*W)
    loss = weighted_depth_diff.mean()
    
    return loss

