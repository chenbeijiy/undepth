"""
改进版多视角反射一致性约束损失（Improved Multi-View Reflection Consistency Loss）

改进策略（参考PGSR等方法）：
1. 简化反射强度估计：直接使用RGB亮度，而不是复杂的反射强度计算
2. 减少计算频率：只在关键迭代计算
3. 使用低分辨率：降低计算分辨率
4. 简化视角权重：使用更简单的权重计算
5. 只在关键区域计算：只在高光/高alpha区域计算

核心思想：
- 直接约束多视角下的RGB亮度一致性（简化版）
- 只在特定区域（高光/高alpha）计算，减少计算量
- 使用更简单的权重机制
"""

import torch
import torch.nn.functional as F


def compute_luminance(rgb_image):
    """
    计算RGB亮度（简化版）
    
    Args:
        rgb_image: [3, H, W] RGB图像（归一化到[0, 1]）
    
    Returns:
        luminance: [H, W] 亮度
    """
    # 使用标准RGB到亮度的转换：Y = 0.299*R + 0.587*G + 0.114*B
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_image.device, dtype=rgb_image.dtype)
    luminance = torch.sum(rgb_image * weights.view(3, 1, 1), dim=0)  # [H, W]
    return luminance


def compute_highlight_mask(luminance, alpha=None, threshold=0.7):
    """
    计算高光区域mask（简化版）
    
    只在高光区域（高亮度或高alpha）计算一致性损失
    
    Args:
        luminance: [H, W] 亮度
        alpha: [1, H, W] 或 [H, W] alpha值（可选）
        threshold: 阈值（默认0.7）
    
    Returns:
        mask: [H, W] 高光区域mask
    """
    # 方法1：使用亮度阈值
    brightness_mask = (luminance > threshold).float()  # [H, W]
    
    # 方法2：如果提供了alpha，也考虑alpha
    if alpha is not None:
        if alpha.dim() == 3:
            alpha = alpha.squeeze(0)  # [H, W]
        alpha_mask = (alpha > 0.5).float()  # [H, W]
        # 结合两个mask（高亮度或高alpha）
        combined_mask = torch.max(brightness_mask, alpha_mask)
    else:
        combined_mask = brightness_mask
    
    return combined_mask


def multiview_reflection_consistency_loss_improved(
    render_pkgs,
    viewpoint_cameras,
    lambda_weight=1.0,
    mask_background=True,
    use_highlight_mask=True,
    highlight_threshold=0.7,
    resolution_scale=0.5
):
    """
    改进版多视角反射一致性损失
    
    改进点：
    1. 直接使用RGB亮度，而不是复杂的反射强度计算
    2. 使用低分辨率计算（可选）
    3. 只在高光区域计算（可选）
    4. 简化视角权重计算
    
    Args:
        render_pkgs: 渲染结果字典列表，每个元素包含：
            - 'render': [3, H, W] RGB图像
            - 'rend_alpha': [1, H, W] alpha值（可选）
        viewpoint_cameras: 视角相机列表
        lambda_weight: 权重参数（默认1.0）
        mask_background: 是否mask背景区域（默认True）
        use_highlight_mask: 是否只在高光区域计算（默认True）
        highlight_threshold: 高光阈值（默认0.7）
        resolution_scale: 分辨率缩放因子（默认0.5，即1/2分辨率）
    
    Returns:
        loss: 标量tensor，多视角反射一致性损失
    """
    if len(render_pkgs) < 2 or len(viewpoint_cameras) < 2:
        return torch.tensor(0.0, device="cuda", requires_grad=True)
    
    # 确保render_pkgs和viewpoint_cameras长度一致
    assert len(render_pkgs) == len(viewpoint_cameras), \
        f"render_pkgs ({len(render_pkgs)}) and viewpoint_cameras ({len(viewpoint_cameras)}) must have the same length"
    
    # 计算每个视角的亮度和mask
    luminances = []
    alpha_masks = []
    highlight_masks = []
    
    for render_pkg in render_pkgs:
        # 获取RGB图像
        rgb_image = render_pkg['render']  # [3, H, W]
        H, W = rgb_image.shape[1], rgb_image.shape[2]
        
        # 可选：使用低分辨率
        if resolution_scale < 1.0:
            target_H = int(H * resolution_scale)
            target_W = int(W * resolution_scale)
            rgb_image = F.interpolate(
                rgb_image.unsqueeze(0), 
                size=(target_H, target_W), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)  # [3, target_H, target_W]
            H, W = target_H, target_W
        
        # 计算亮度（简化版）
        luminance = compute_luminance(rgb_image)  # [H, W]
        luminances.append(luminance)
        
        # 获取alpha mask（用于mask背景）
        rend_alpha = render_pkg.get('rend_alpha', None)
        rend_alpha_resized = None
        if mask_background:
            if rend_alpha is not None:
                if resolution_scale < 1.0:
                    rend_alpha_resized = F.interpolate(
                        rend_alpha.unsqueeze(0), 
                        size=(target_H, target_W), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)  # [1, target_H, target_W]
                else:
                    rend_alpha_resized = rend_alpha
                alpha_mask = (rend_alpha_resized.squeeze(0) > 0.5).float()  # [H, W]
            else:
                alpha_mask = torch.ones(H, W, device=rgb_image.device)
            alpha_masks.append(alpha_mask)
        else:
            alpha_masks.append(torch.ones(H, W, device=rgb_image.device))
        
        # 计算高光区域mask（可选）
        if use_highlight_mask:
            # 使用已经resize的rend_alpha（如果存在）
            rend_alpha_for_highlight = None
            if rend_alpha_resized is not None:
                rend_alpha_for_highlight = rend_alpha_resized.squeeze(0) if rend_alpha_resized.dim() == 3 else rend_alpha_resized
            elif rend_alpha is not None:
                rend_alpha_for_highlight = rend_alpha.squeeze(0) if rend_alpha.dim() == 3 else rend_alpha
            
            highlight_mask = compute_highlight_mask(
                luminance, 
                rend_alpha_for_highlight,
                highlight_threshold
            )  # [H, W]
            highlight_masks.append(highlight_mask)
        else:
            highlight_masks.append(torch.ones(H, W, device=rgb_image.device))
    
    # 计算视角间的亮度一致性损失
    total_loss = 0.0
    num_pairs = 0
    
    for i in range(len(viewpoint_cameras)):
        for j in range(i + 1, len(viewpoint_cameras)):
            # 获取两个视角的亮度
            L_i = luminances[i]  # [H, W]
            L_j = luminances[j]  # [H, W]
            
            # 确保两个视角的图像尺寸相同
            if L_i.shape != L_j.shape:
                min_H = min(L_i.shape[0], L_j.shape[0])
                min_W = min(L_i.shape[1], L_j.shape[1])
                L_i = L_i[:min_H, :min_W]
                L_j = L_j[:min_H, :min_W]
                alpha_mask_i = alpha_masks[i][:min_H, :min_W]
                alpha_mask_j = alpha_masks[j][:min_H, :min_W]
                highlight_mask_i = highlight_masks[i][:min_H, :min_W]
                highlight_mask_j = highlight_masks[j][:min_H, :min_W]
            else:
                alpha_mask_i = alpha_masks[i]
                alpha_mask_j = alpha_masks[j]
                highlight_mask_i = highlight_masks[i]
                highlight_mask_j = highlight_masks[j]
            
            # 计算亮度差异（简化版：直接使用L1损失）
            luminance_diff = torch.abs(L_i - L_j)  # [H, W]
            
            # 应用mask
            combined_mask = torch.ones_like(luminance_diff)
            
            # Mask背景区域
            if mask_background:
                combined_mask = combined_mask * alpha_mask_i * alpha_mask_j
            
            # Mask非高光区域（如果启用）
            if use_highlight_mask:
                # 只在高光区域计算（两个视角都是高光）
                combined_mask = combined_mask * highlight_mask_i * highlight_mask_j
            
            # 应用mask并计算损失
            masked_diff = luminance_diff * combined_mask  # [H, W]
            
            # 计算有效像素数量（避免除零）
            valid_pixels = combined_mask.sum()
            if valid_pixels > 0:
                pair_loss = masked_diff.sum() / valid_pixels
            else:
                pair_loss = torch.tensor(0.0, device=luminance_diff.device)
            
            total_loss = total_loss + pair_loss
            num_pairs += 1
    
    # 计算平均损失
    if num_pairs > 0:
        avg_loss = lambda_weight * (total_loss / num_pairs)
    else:
        avg_loss = torch.tensor(0.0, device="cuda", requires_grad=True)
    
    return avg_loss


def multiview_reflection_consistency_loss_minimal(
    render_pkgs,
    viewpoint_cameras,
    lambda_weight=1.0,
    mask_background=True
):
    """
    最小化版本的多视角反射一致性损失
    
    最简化的实现：
    - 直接比较RGB亮度
    - 不使用复杂的权重计算
    - 不使用高光mask
    - 不使用低分辨率
    
    适用于快速测试和调试
    
    Args:
        render_pkgs: 渲染结果字典列表
        viewpoint_cameras: 视角相机列表
        lambda_weight: 权重参数
        mask_background: 是否mask背景区域
    
    Returns:
        loss: 标量tensor，多视角反射一致性损失
    """
    if len(render_pkgs) < 2:
        return torch.tensor(0.0, device="cuda", requires_grad=True)
    
    # 计算每个视角的亮度
    luminances = []
    alpha_masks = []
    
    for render_pkg in render_pkgs:
        rgb_image = render_pkg['render']  # [3, H, W]
        luminance = compute_luminance(rgb_image)  # [H, W]
        luminances.append(luminance)
        
        if mask_background:
            rend_alpha = render_pkg.get('rend_alpha', None)
            if rend_alpha is not None:
                alpha_mask = (rend_alpha.squeeze(0) > 0.5).float()
            else:
                alpha_mask = torch.ones_like(luminance)
            alpha_masks.append(alpha_mask)
        else:
            alpha_masks.append(torch.ones_like(luminance))
    
    # 计算视角间的亮度一致性损失
    total_loss = 0.0
    num_pairs = 0
    
    for i in range(len(render_pkgs)):
        for j in range(i + 1, len(render_pkgs)):
            L_i = luminances[i]
            L_j = luminances[j]
            
            # 处理尺寸不匹配
            if L_i.shape != L_j.shape:
                min_H = min(L_i.shape[0], L_j.shape[0])
                min_W = min(L_i.shape[1], L_j.shape[1])
                L_i = L_i[:min_H, :min_W]
                L_j = L_j[:min_H, :min_W]
                alpha_mask_i = alpha_masks[i][:min_H, :min_W]
                alpha_mask_j = alpha_masks[j][:min_H, :min_W]
            else:
                alpha_mask_i = alpha_masks[i]
                alpha_mask_j = alpha_masks[j]
            
            # 计算亮度差异
            luminance_diff = torch.abs(L_i - L_j)
            
            # 应用mask
            if mask_background:
                combined_mask = alpha_mask_i * alpha_mask_j
                valid_pixels = combined_mask.sum()
                if valid_pixels > 0:
                    pair_loss = (luminance_diff * combined_mask).sum() / valid_pixels
                else:
                    pair_loss = torch.tensor(0.0, device=luminance_diff.device)
            else:
                pair_loss = luminance_diff.mean()
            
            total_loss = total_loss + pair_loss
            num_pairs += 1
    
    # 计算平均损失
    if num_pairs > 0:
        avg_loss = lambda_weight * (total_loss / num_pairs)
    else:
        avg_loss = torch.tensor(0.0, device="cuda", requires_grad=True)
    
    return avg_loss
