"""
创新4：深度-法线联合优化（Depth-Normal Joint Optimization）
深度和法线应该几何一致：深度梯度应该与法线方向一致
"""

import torch


def depth_normal_consistency_loss(surf_depth, surf_normal, lambda_dn=0.1):
    """
    深度-法线一致性损失
    
    约束深度梯度方向与法线方向一致
    
    Args:
        surf_depth: 表面深度图 (1, H, W) 或 (H, W)
        surf_normal: 表面法线图 (3, H, W)
        lambda_dn: 损失权重
    
    Returns:
        loss: 深度-法线一致性损失（标量）
    """
    # 确保输入格式正确
    if surf_depth.dim() == 3:
        surf_depth = surf_depth.squeeze(0)  # (H, W)
    if surf_normal.dim() == 3:
        surf_normal = surf_normal  # (3, H, W)
    else:
        raise ValueError(f"surf_normal should be (3, H, W), got {surf_normal.shape}")
    
    H, W = surf_depth.shape[-2:]
    
    # 计算深度梯度（空间梯度）
    # x方向梯度：depth_grad_x[i, j] = depth[i, j+1] - depth[i, j]
    depth_grad_x = surf_depth[:, 1:] - surf_depth[:, :-1]  # (H, W-1)
    # y方向梯度：depth_grad_y[i, j] = depth[i+1, j] - depth[i, j]
    depth_grad_y = surf_depth[1:, :] - surf_depth[:-1, :]  # (H-1, W)
    
    # 计算梯度的大小（用于归一化）
    # 需要对齐到相同的空间位置
    # 取较小的尺寸以匹配两个梯度
    min_h = min(depth_grad_x.shape[0], depth_grad_y.shape[0])
    min_w = min(depth_grad_x.shape[1], depth_grad_y.shape[1])
    
    depth_grad_x_aligned = depth_grad_x[:min_h, :min_w]  # (min_h, min_w)
    depth_grad_y_aligned = depth_grad_y[:min_h, :min_w]  # (min_h, min_w)
    
    # 计算梯度大小（在xy平面上的梯度大小）
    depth_grad_norm_xy = torch.sqrt(depth_grad_x_aligned.pow(2) + depth_grad_y_aligned.pow(2) + 1e-8)  # (min_h, min_w)
    
    # 构建3D梯度方向向量
    # 梯度方向 = (grad_x, grad_y, 1) 归一化
    # 这里1表示深度方向（z方向）
    depth_grad_dir = torch.stack([
        depth_grad_x_aligned / (depth_grad_norm_xy + 1e-8),  # x方向分量
        depth_grad_y_aligned / (depth_grad_norm_xy + 1e-8),  # y方向分量
        torch.ones_like(depth_grad_x_aligned) / (depth_grad_norm_xy + 1e-8)  # z方向分量（归一化）
    ], dim=0)  # (3, min_h, min_w)
    
    # 法线需要对齐到梯度位置（裁剪到相同尺寸）
    normal_aligned = surf_normal[:, :min_h, :min_w]  # (3, min_h, min_w)
    
    # 归一化法线（确保是单位向量）
    normal_norm = torch.sqrt(normal_aligned.pow(2).sum(dim=0) + 1e-8)  # (min_h, min_w)
    normal_aligned = normal_aligned / (normal_norm.unsqueeze(0) + 1e-8)  # (3, min_h, min_w)
    
    # 计算方向差异：1 - dot(depth_grad_dir, normal)
    # dot product: (depth_grad_dir * normal_aligned).sum(dim=0)
    dot_product = (depth_grad_dir * normal_aligned).sum(dim=0)  # (min_h, min_w)
    
    # 方向差异：1 - dot_product（当方向一致时，dot_product接近1，差异接近0）
    # 使用绝对值确保差异为正
    dir_diff = (1.0 - dot_product).abs()  # (min_h, min_w)
    
    # 计算平均损失
    loss = lambda_dn * dir_diff.mean()
    
    return loss

