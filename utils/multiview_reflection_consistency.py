"""
多视角反射一致性约束损失（Multi-View Reflection Consistency Loss）

创新点1：直接约束多视角下的反射一致性，而非仅约束深度。
如果反射是一致的，那么高斯就不需要通过深度偏差来拟合高光。

数学公式：
    L_reflection = Σ_{i,j} w_{i,j} · ||R_i(p) - R_j(p)||²
    
其中：
    R_i(p) = (I_i(p) - I_diffuse(p)) / (I_diffuse(p) + ε)
    w_{i,j} = exp(-λ_view_weight · |cos(θ_i) - cos(θ_j)|)
"""

import torch
import torch.nn.functional as F
import random
from utils.view_dependent_depth_constraint import compute_view_direction, compute_view_normal_angle


def compute_reflection_strength(rgb_image, diffuse_estimate=None, epsilon=1e-6):
    """
    计算反射强度（镜面反射分量）
    
    反射强度定义：
        R(p) = (I(p) - I_diffuse(p)) / (I_diffuse(p) + ε)
    
    其中：
        I(p) 是RGB亮度
        I_diffuse(p) 是估计的漫反射分量
    
    实现方法：
        使用RGB最小值作为漫反射估计（因为漫反射在所有视角下相似）
    
    Args:
        rgb_image: [3, H, W] RGB图像（归一化到[0, 1]）
        diffuse_estimate: [3, H, W] 或 [H, W] 漫反射估计（可选，当前未使用）
        epsilon: 小值，避免除零
    
    Returns:
        reflection_strength: [H, W] 反射强度
    """
    # 计算RGB亮度（加权平均）
    # 使用标准RGB到亮度的转换：Y = 0.299*R + 0.587*G + 0.114*B
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_image.device, dtype=rgb_image.dtype)
    luminance = torch.sum(rgb_image * weights.view(3, 1, 1), dim=0)  # [H, W]
    
    # 估计漫反射分量
    # 方法：使用RGB最小值作为漫反射估计
    # 假设：漫反射在所有视角下相似，使用最小值可以近似漫反射
    # 注意：这里我们使用每个像素的最小值（更准确）
    # 但实际上，我们需要一个稳定的参考，所以使用全局最小值
    # 更准确的方法是：使用SH的0阶项（漫反射项），但这里简化处理
    diffuse_luminance = torch.min(luminance)  # 标量，全局最小值
    
    # 计算反射强度
    # R = (I - I_diffuse) / (I_diffuse + ε)
    # 如果I < I_diffuse，反射强度为负（不太可能，但处理一下）
    reflection_strength = (luminance - diffuse_luminance) / (diffuse_luminance + epsilon)
    
    # 限制反射强度在合理范围内（避免异常值）
    reflection_strength = torch.clamp(reflection_strength, -1.0, 10.0)
    
    return reflection_strength  # [H, W]


def compute_view_weight_pairwise(cos_theta_i, cos_theta_j, lambda_view_weight=2.0):
    """
    计算视角对的权重
    
    权重公式：
        w_{i,j} = exp(-λ_view_weight · |cos(θ_i) - cos(θ_j)|)
    
    当两个视角的视角-法线夹角相似时，权重接近1.0
    当两个视角的视角-法线夹角差异大时，权重接近0
    
    Args:
        cos_theta_i: [H, W] 视角i的视角-法线夹角余弦值
        cos_theta_j: [H, W] 视角j的视角-法线夹角余弦值
        lambda_view_weight: 权重参数（默认2.0）
    
    Returns:
        weight: [H, W] 视角对的权重
    """
    # 计算视角-法线夹角差异
    cos_diff = torch.abs(cos_theta_i - cos_theta_j)  # [H, W]
    
    # 计算权重
    weight = torch.exp(-lambda_view_weight * cos_diff)  # [H, W]
    
    return weight


# 注意：multiview_reflection_consistency_loss函数已移除
# 使用multiview_reflection_consistency_loss_simple代替
# 因为它接受已经渲染好的render_pkgs，更灵活且避免重复渲染


def multiview_reflection_consistency_loss_simple(
    render_pkgs,
    viewpoint_cameras,
    lambda_view_weight=2.0,
    mask_background=True,
    epsilon=1e-6
):
    """
    简化版本的多视角反射一致性损失
    
    这个版本接受已经渲染好的render_pkgs，避免在函数内部渲染
    
    Args:
        render_pkgs: 渲染结果字典列表，每个元素包含：
            - 'render': [3, H, W] RGB图像
            - 'surf_normal': [3, H, W] 表面法线（可选）
            - 'rend_alpha': [1, H, W] alpha值（可选）
        viewpoint_cameras: 视角相机列表
        lambda_view_weight: 视角权重参数
        mask_background: 是否mask背景区域
        epsilon: 小值，避免除零
    
    Returns:
        loss: 标量tensor，多视角反射一致性损失
    """
    if len(render_pkgs) < 2 or len(viewpoint_cameras) < 2:
        return torch.tensor(0.0, device="cuda", requires_grad=True)
    
    # 确保render_pkgs和viewpoint_cameras长度一致
    assert len(render_pkgs) == len(viewpoint_cameras), \
        f"render_pkgs ({len(render_pkgs)}) and viewpoint_cameras ({len(viewpoint_cameras)}) must have the same length"
    
    # 计算每个视角的反射强度和视角-法线夹角
    reflection_strengths = []
    cos_thetas = []
    alpha_masks = []
    
    for render_pkg, viewpoint_cam in zip(render_pkgs, viewpoint_cameras):
        # 获取RGB图像
        rgb_image = render_pkg['render']  # [3, H, W]
        H, W = rgb_image.shape[1], rgb_image.shape[2]
        
        # 计算反射强度
        reflection_strength = compute_reflection_strength(rgb_image, epsilon=epsilon)  # [H, W]
        reflection_strengths.append(reflection_strength)
        
        # 计算视角-法线夹角
        surf_normal = render_pkg.get('surf_normal', None)  # [3, H, W]
        if surf_normal is not None:
            view_dirs = compute_view_direction(viewpoint_cam, H, W)  # [H, W, 3]
            cos_theta = compute_view_normal_angle(view_dirs, surf_normal)  # [H, W]
            cos_thetas.append(cos_theta)
        else:
            # 如果没有法线，使用默认值（所有视角权重相同）
            cos_thetas.append(torch.zeros(H, W, device=rgb_image.device))
        
        # 获取alpha mask（用于mask背景）
        if mask_background:
            rend_alpha = render_pkg.get('rend_alpha', None)
            if rend_alpha is not None:
                alpha_mask = (rend_alpha.squeeze(0) > 0.5).float()  # [H, W]
            else:
                alpha_mask = torch.ones(H, W, device=rgb_image.device)
            alpha_masks.append(alpha_mask)
        else:
            alpha_masks.append(torch.ones(H, W, device=rgb_image.device))
    
    # 计算视角间的反射一致性损失
    total_loss = 0.0
    num_pairs = 0
    
    for i in range(len(viewpoint_cameras)):
        for j in range(i + 1, len(viewpoint_cameras)):
            # 获取两个视角的反射强度
            R_i = reflection_strengths[i]  # [H, W]
            R_j = reflection_strengths[j]  # [H, W]
            
            # 确保两个视角的图像尺寸相同
            if R_i.shape != R_j.shape:
                # 如果尺寸不同，进行resize（使用双线性插值）
                min_H = min(R_i.shape[0], R_j.shape[0])
                min_W = min(R_i.shape[1], R_j.shape[1])
                R_i = R_i[:min_H, :min_W]
                R_j = R_j[:min_H, :min_W]
                cos_theta_i = cos_thetas[i][:min_H, :min_W]
                cos_theta_j = cos_thetas[j][:min_H, :min_W]
                alpha_mask_i = alpha_masks[i][:min_H, :min_W]
                alpha_mask_j = alpha_masks[j][:min_H, :min_W]
            else:
                cos_theta_i = cos_thetas[i]
                cos_theta_j = cos_thetas[j]
                alpha_mask_i = alpha_masks[i]
                alpha_mask_j = alpha_masks[j]
            
            # 计算反射强度差异
            reflection_diff = R_i - R_j  # [H, W]
            reflection_diff_sq = reflection_diff ** 2  # [H, W]
            
            # 计算视角权重
            view_weight = compute_view_weight_pairwise(
                cos_theta_i, cos_theta_j, lambda_view_weight
            )  # [H, W]
            
            # 应用视角权重
            weighted_diff_sq = view_weight * reflection_diff_sq  # [H, W]
            
            # Mask背景区域
            if mask_background:
                # 两个视角都必须是前景
                combined_mask = alpha_mask_i * alpha_mask_j  # [H, W]
                weighted_diff_sq = weighted_diff_sq * combined_mask
            
            # 累加损失
            pair_loss = weighted_diff_sq.mean()
            total_loss = total_loss + pair_loss
            num_pairs += 1
    
    # 计算平均损失
    if num_pairs > 0:
        avg_loss = total_loss / num_pairs
    else:
        avg_loss = torch.tensor(0.0, device="cuda", requires_grad=True)
    
    return avg_loss
