"""
改进2.3：空间深度平滑损失（Spatial Depth Smoothness Loss）
在空间邻域内强制深度平滑，直接改善TSDF融合质量
"""

import torch


def spatial_depth_smoothness_loss(surf_depth, lambda_smooth=0.1):
    """
    空间深度平滑损失：惩罚相邻像素的深度差异
    
    Args:
        surf_depth: 表面深度图 (1, H, W) 或 (H, W)
        lambda_smooth: 平滑权重
    
    Returns:
        loss: 空间深度平滑损失（标量）
    """
    # 确保输入格式正确
    if surf_depth.dim() == 3:
        surf_depth = surf_depth.squeeze(0)  # (H, W)
    
    # 计算水平和垂直方向的深度梯度
    depth_grad_x = torch.abs(surf_depth[:, 1:] - surf_depth[:, :-1])  # (H, W-1)
    depth_grad_y = torch.abs(surf_depth[1:, :] - surf_depth[:-1, :])  # (H-1, W)
    
    # 计算平均梯度（深度平滑损失）
    smooth_loss = lambda_smooth * (depth_grad_x.mean() + depth_grad_y.mean())
    
    return smooth_loss

