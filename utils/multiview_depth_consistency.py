"""
多视角深度一致性损失（Multi-View Depth Consistency Loss）

实现标准的多视角深度一致性约束，确保同一3D点在所有视角下的深度一致。
这对于改善TSDF融合质量和减少坑洞问题非常重要。

参考：INNOVATIVE_IMPROVEMENTS.md 中的创新2
"""

import torch
import torch.nn.functional as F
import random


def project_depth_to_view(depth_src, viewpoint_src, viewpoint_tgt):
    """
    将源视角的深度图投影到目标视角
    
    核心思想：
    1. 使用depths_to_points将源视角的深度图转换为3D世界坐标点
    2. 将3D世界坐标点投影到目标视角的像素坐标
    3. 返回投影后的深度和有效掩码
    
    Args:
        depth_src: 源视角的深度图 (1, H, W) 或 (H, W)
        viewpoint_src: 源视角相机对象
        viewpoint_tgt: 目标视角相机对象
    
    Returns:
        depth_proj: 投影到目标视角的深度（相机坐标系z坐标）(H_src, W_src)
        pixel_coords_tgt: 投影到目标视角的像素坐标 (H_src, W_src, 2)，归一化到[-1,1]
        valid_mask: 有效投影区域掩码 (H_src, W_src)
    """
    from utils.point_utils import depths_to_points
    
    device = depth_src.device
    
    # 确保深度图格式正确
    if depth_src.dim() == 3:
        depth_src = depth_src.squeeze(0)  # (H, W)
    
    H_src, W_src = depth_src.shape
    H_tgt, W_tgt = viewpoint_tgt.image_height, viewpoint_tgt.image_width
    
    # 步骤1: 将源视角的深度图转换为3D世界坐标点
    points_world = depths_to_points(viewpoint_src, depth_src)  # (H*W, 3)
    points_world = points_world.reshape(H_src, W_src, 3)  # (H, W, 3)
    
    # 步骤2: 将3D世界坐标点投影到目标视角
    # 将3D点转换为齐次坐标
    points_homo = torch.cat([
        points_world,
        torch.ones(H_src, W_src, 1, device=device)
    ], dim=-1)  # (H, W, 4)
    
    # 使用full_proj_transform直接投影（包含world_view和projection）
    proj_matrix_tgt = viewpoint_tgt.full_proj_transform  # (4, 4)
    points_proj_homo = points_homo @ proj_matrix_tgt.T  # (H, W, 4)
    
    # 透视除法，得到NDC坐标
    w = points_proj_homo[:, :, 3:4]  # (H, W, 1)
    points_ndc = points_proj_homo[:, :, :3] / (w + 1e-8)  # (H, W, 3)
    
    # 提取NDC坐标的xy（用于grid_sample，范围[-1, 1]）
    pixel_coords_tgt_normalized = points_ndc[:, :, :2]  # (H, W, 2)
    
    # 计算相机坐标系中的深度（用于比较）
    # 需要从世界坐标转换到相机坐标
    w2c_tgt = viewpoint_tgt.world_view_transform.T  # (4, 4)
    points_cam_tgt = points_homo @ w2c_tgt.T  # (H, W, 4)
    depth_cam_tgt = points_cam_tgt[:, :, 2]  # (H, W)
    
    # 检查深度有效性（必须在近远平面之间）
    near_plane = viewpoint_tgt.znear if hasattr(viewpoint_tgt, 'znear') else 0.01
    far_plane = viewpoint_tgt.zfar if hasattr(viewpoint_tgt, 'zfar') else 100.0
    valid_depth_mask = (depth_cam_tgt > near_plane) & (depth_cam_tgt < far_plane)
    
    # 检查像素坐标是否在有效范围内（NDC空间[-1, 1]）
    valid_pixel_mask = (
        (pixel_coords_tgt_normalized[:, :, 0] >= -1.0) & 
        (pixel_coords_tgt_normalized[:, :, 0] <= 1.0) &
        (pixel_coords_tgt_normalized[:, :, 1] >= -1.0) & 
        (pixel_coords_tgt_normalized[:, :, 1] <= 1.0)
    )
    
    # 组合有效掩码
    valid_mask = valid_depth_mask & valid_pixel_mask
    
    return depth_cam_tgt, pixel_coords_tgt_normalized, valid_mask


def multiview_depth_consistency_loss(
    gaussians,
    render_func,
    viewpoint_stack,
    pipe,
    background,
    lambda_multiview=0.05,
    n_views=3,
    sample_random=True,
    return_details=False
):
    """
    多视角深度一致性损失
    
    核心思想：在训练时，同时渲染多个视角，约束同一3D点在所有视角下的深度一致性。
    
    数学公式：
    L_multiview = Σ_{i,j} w_{i,j} · |d_i(π_j(p)) - d_j(π_j(p))|²
    
    其中：
    - d_i(x) 是视角i下像素x的深度
    - π_j(p) 是将3D点p投影到视角j的像素坐标
    - w_{i,j} 是权重（基于投影有效性）
    
    Args:
        gaussians: 高斯模型
        render_func: 渲染函数 (viewpoint, gaussians, pipe, background) -> render_pkg
        viewpoint_stack: 视角列表
        pipe: 渲染管道参数
        background: 背景颜色张量
        lambda_multiview: 损失权重
        n_views: 采样的视角数量（建议2-3个）
        sample_random: 是否随机采样视角
        return_details: 是否返回详细信息（用于调试）
    
    Returns:
        loss: 多视角深度一致性损失（标量）
        如果return_details=True，还返回详细信息字典
    """
    if len(viewpoint_stack) < 2:
        # 如果视角数量不足，返回零损失
        if return_details:
            return torch.tensor(0.0, device="cuda"), {}
        return torch.tensor(0.0, device="cuda")
    
    # 随机采样n_views个视角
    n_views = min(n_views, len(viewpoint_stack))
    if sample_random:
        selected_views = random.sample(viewpoint_stack, n_views)
    else:
        selected_views = viewpoint_stack[:n_views]
    
    # 渲染所有选中的视角
    render_pkgs = []
    depths = []
    for view in selected_views:
        # 渲染
        render_pkg = render_func(view, gaussians, pipe, background)
        render_pkgs.append(render_pkg)
        depths.append(render_pkg['surf_depth'])  # (1, H, W) 或 (H, W)
    
    # 计算所有视角对之间的深度一致性
    total_loss = 0.0
    n_pairs = 0
    loss_details = {
        'n_views': n_views,
        'n_pairs': 0,
        'pair_losses': []
    }
    
    for i in range(n_views):
        for j in range(i + 1, n_views):
            depth_i = depths[i]
            depth_j = depths[j]
            view_i = selected_views[i]
            view_j = selected_views[j]
            
            # 确保深度图格式正确
            if depth_i.dim() == 3:
                depth_i = depth_i.squeeze(0)  # (H_i, W_i)
            if depth_j.dim() == 3:
                depth_j = depth_j.squeeze(0)  # (H_j, W_j)
            
            H_i, W_i = depth_i.shape
            H_j, W_j = depth_j.shape
            
            # 将视角i的深度投影到视角j
            depth_i_proj_cam, pixel_coords_ij, valid_mask_ij = project_depth_to_view(depth_i, view_i, view_j)
            
            # 将视角j的深度投影到视角i（双向约束）
            depth_j_proj_cam, pixel_coords_ji, valid_mask_ji = project_depth_to_view(depth_j, view_j, view_i)
            
            # 计算深度差异（视角i投影到视角j）
            # 使用grid_sample将视角j的深度采样到视角i投影的像素位置
            depth_j_sampled = F.grid_sample(
                depth_j.unsqueeze(0).unsqueeze(0),  # (1, 1, H_j, W_j)
                pixel_coords_ij.unsqueeze(0),  # (1, H_i, W_i, 2) - 归一化坐标
                mode='bilinear',
                padding_mode='border',
                align_corners=False
            ).squeeze(0).squeeze(0)  # (H_i, W_i)
            
            # 计算深度差异（只在有效区域）
            valid_mask_ij_final = valid_mask_ij & (depth_i_proj_cam > 0) & (depth_j_sampled > 0)
            
            if valid_mask_ij_final.sum() > 0:
                depth_diff_ij = (depth_i_proj_cam - depth_j_sampled)[valid_mask_ij_final]
                loss_ij = (depth_diff_ij.pow(2)).mean()
                
                # 反向投影（视角j投影到视角i）
                depth_i_sampled = F.grid_sample(
                    depth_i.unsqueeze(0).unsqueeze(0),  # (1, 1, H_i, W_i)
                    pixel_coords_ji.unsqueeze(0),  # (1, H_j, W_j, 2)
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=False
                ).squeeze(0).squeeze(0)  # (H_j, W_j)
                
                valid_mask_ji_final = valid_mask_ji & (depth_j_proj_cam > 0) & (depth_i_sampled > 0)
                
                if valid_mask_ji_final.sum() > 0:
                    depth_diff_ji = (depth_j_proj_cam - depth_i_sampled)[valid_mask_ji_final]
                    loss_ji = (depth_diff_ji.pow(2)).mean()
                    
                    # 双向损失的平均
                    pair_loss = (loss_ij + loss_ji) / 2.0
                    total_loss += pair_loss
                    n_pairs += 1
                    
                    if return_details:
                        loss_details['pair_losses'].append({
                            'pair': (i, j),
                            'loss_ij': loss_ij.item(),
                            'loss_ji': loss_ji.item(),
                            'pair_loss': pair_loss.item(),
                            'valid_pixels_ij': valid_mask_ij_final.sum().item(),
                            'valid_pixels_ji': valid_mask_ji_final.sum().item()
                        })
    
    # 计算平均损失
    if n_pairs > 0:
        avg_loss = total_loss / n_pairs
        final_loss = lambda_multiview * avg_loss
    else:
        final_loss = torch.tensor(0.0, device="cuda")
    
    loss_details['n_pairs'] = n_pairs
    loss_details['avg_loss'] = avg_loss.item() if n_pairs > 0 else 0.0
    loss_details['final_loss'] = final_loss.item()
    
    if return_details:
        return final_loss, loss_details
    
    return final_loss

