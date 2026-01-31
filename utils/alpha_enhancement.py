"""
创新3：自适应Alpha增强（Adaptive Alpha Enhancement）
在坑洞风险区域（深度方差大、alpha分散），自动增强alpha值，强制alpha饱和
"""

import torch
import torch.nn.functional as F


def compute_spatial_variance(depth, kernel_size=5):
    """
    计算空间深度方差
    
    Args:
        depth: 深度图 (1, H, W) 或 (H, W)
        kernel_size: 邻域大小
    
    Returns:
        depth_var: 深度方差图 (H, W)
    """
    if depth.dim() == 3:
        depth = depth.squeeze(0)  # (H, W)
    
    # 使用unfold提取邻域
    depth_padded = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    depth_unfold = F.unfold(depth_padded, kernel_size=kernel_size, padding=kernel_size//2)
    
    # 计算方差
    depth_mean = depth_unfold.mean(dim=1, keepdim=True)  # (1, 1, H*W)
    depth_var = ((depth_unfold - depth_mean) ** 2).mean(dim=1, keepdim=True)  # (1, 1, H*W)
    
    # 恢复空间维度
    H, W = depth.shape
    depth_var = F.fold(depth_var, (H, W), (1, 1)).squeeze()  # (H, W)
    
    return depth_var


def adaptive_alpha_enhancement_loss(rend_alpha, surf_depth, lambda_enhance=0.2, 
                                   depth_var_scale=10.0):
    """
    自适应Alpha增强损失
    
    在深度方差大且alpha低的区域，强制alpha饱和
    
    Args:
        rend_alpha: 渲染alpha图 (1, H, W) 或 (H, W)
        surf_depth: 表面深度图 (1, H, W) 或 (H, W)
        lambda_enhance: 损失权重
        depth_var_scale: 深度方差缩放因子，用于调整坑洞检测的敏感度
    
    Returns:
        loss: Alpha增强损失（标量）
    """
    # 确保是2D
    if rend_alpha.dim() == 3:
        rend_alpha = rend_alpha.squeeze(0)
    if surf_depth.dim() == 3:
        surf_depth = surf_depth.squeeze(0)
    
    # 计算深度方差（空间邻域）
    depth_var = compute_spatial_variance(surf_depth, kernel_size=5)
    
    # 计算坑洞风险权重
    # 深度方差大 + alpha低 → 高权重
    # hole_risk = depth_var * (1.0 - rend_alpha)
    # 使用sigmoid归一化到[0,1]，depth_var_scale控制敏感度
    hole_risk = depth_var * (1.0 - rend_alpha)
    hole_weight = torch.sigmoid(hole_risk * depth_var_scale)  # 归一化到[0,1]
    
    # Alpha增强损失：在坑洞区域强制alpha饱和
    # loss = hole_weight * (1.0 - rend_alpha)^2
    alpha_enhance_loss = hole_weight * (1.0 - rend_alpha).pow(2)
    
    return lambda_enhance * alpha_enhance_loss.mean()

