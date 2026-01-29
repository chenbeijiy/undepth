#
# Multi-view depth consistency utilities for Improvement 2.4
#

import torch
import torch.nn.functional as F


def project_depth_to_view(depth_map, source_view, target_view):
    """
    Project depth map from source view to target view.
    
    Args:
        depth_map: Depth map from source view, shape (1, H, W)
        source_view: Source camera view
        target_view: Target camera view
    
    Returns:
        projected_depth: Projected depth in target camera space, shape (1, H, W)
        projected_pixels: Projected pixel coordinates in target view, shape (H*W, 2)
        valid_mask: Valid mask indicating which pixels are visible in target view, shape (1, H, W)
    """
    H, W = depth_map.shape[1], depth_map.shape[2]
    device = depth_map.device
    
    # Convert depth map to 3D points in world space using source view
    from utils.point_utils import depths_to_points
    points_3d_world = depths_to_points(source_view, depth_map)  # (H*W, 3) - points in world space
    
    # Transform points from world space to target camera space
    points_3d_homogeneous = torch.cat([
        points_3d_world, 
        torch.ones(points_3d_world.shape[0], 1, device=device)
    ], dim=1)  # (H*W, 4)
    
    # Transform to target camera space
    points_target_cam = points_3d_homogeneous @ target_view.world_view_transform.T  # (H*W, 4)
    points_target_cam_3d = points_target_cam[:, :3]  # (H*W, 3)
    
    # Get depth in target camera space (z coordinate)
    z_target = points_target_cam_3d[:, 2:3]  # (H*W, 1)
    valid_z = z_target > target_view.znear  # Near plane check
    
    # Project to target view image plane
    target_W, target_H = target_view.image_width, target_view.image_height
    
    # Use full projection transform to project to NDC, then to pixel coordinates
    points_ndc_homogeneous = points_target_cam @ target_view.full_proj_transform.T  # (H*W, 4)
    points_ndc = points_ndc_homogeneous[:, :3] / (points_ndc_homogeneous[:, 3:4] + 1e-8)  # (H*W, 3)
    
    # Convert NDC to pixel coordinates
    ndc2pix = torch.tensor([
        [target_W / 2, 0, 0, (target_W - 1) / 2],
        [0, target_H / 2, 0, (target_H - 1) / 2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device).T
    
    points_pix_homogeneous = points_ndc @ ndc2pix.T  # (H*W, 4)
    points_pix = points_pix_homogeneous[:, :2] / (points_pix_homogeneous[:, 2:3] + 1e-8)  # (H*W, 2)
    
    # Check if points are within image bounds
    valid_x = (points_pix[:, 0] >= 0) & (points_pix[:, 0] < target_W)
    valid_y = (points_pix[:, 1] >= 0) & (points_pix[:, 1] < target_H)
    valid_bounds = valid_x & valid_y & valid_z.squeeze(-1)
    
    # Reshape to image dimensions
    z_target_2d = z_target.reshape(H, W, 1).permute(2, 0, 1)  # (1, H, W)
    valid_mask = valid_bounds.reshape(H, W).unsqueeze(0)  # (1, H, W)
    
    return z_target_2d, points_pix, valid_mask


def multi_view_depth_consistency_loss(depth_map_1, view_1, depth_map_2, view_2):
    """
    Compute multi-view depth consistency loss between two views.
    
    Improvement 2.4: Cross-view depth consistency loss
    L_multi_view_depth = sum ||project(d_v1(x), v2) - d_v2(x')||^2
    
    Args:
        depth_map_1: Depth map from view 1, shape (1, H, W)
        view_1: Camera view 1
        depth_map_2: Depth map from view 2, shape (1, H, W)
        view_2: Camera view 2
    
    Returns:
        loss: Multi-view depth consistency loss (scalar tensor)
    """
    H1, W1 = depth_map_1.shape[1], depth_map_1.shape[2]
    H2, W2 = depth_map_2.shape[1], depth_map_2.shape[2]
    device = depth_map_1.device
    
    # Project depth from view 1 to view 2
    projected_depth_1_to_2, pixels_1_to_2, valid_mask_1_to_2 = project_depth_to_view(depth_map_1, view_1, view_2)
    
    # Project depth from view 2 to view 1
    projected_depth_2_to_1, pixels_2_to_1, valid_mask_2_to_1 = project_depth_to_view(depth_map_2, view_2, view_1)
    
    # Sample depth_map_2 at projected pixel locations from view 1
    # Normalize pixel coordinates to [-1, 1] for grid_sample
    pixels_1_to_2_normalized = torch.zeros_like(pixels_1_to_2)
    pixels_1_to_2_normalized[:, 0] = (pixels_1_to_2[:, 0] / (W2 - 1)) * 2.0 - 1.0
    pixels_1_to_2_normalized[:, 1] = (pixels_1_to_2[:, 1] / (H2 - 1)) * 2.0 - 1.0
    grid_1_to_2 = pixels_1_to_2_normalized.reshape(H1, W1, 2).unsqueeze(0)  # (1, H1, W1, 2)
    
    # Sample depth_map_2 at projected locations
    sampled_depth_2 = F.grid_sample(
        depth_map_2.unsqueeze(0),
        grid_1_to_2,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(0)  # (1, H1, W1)
    
    # Sample depth_map_1 at projected pixel locations from view 2
    pixels_2_to_1_normalized = torch.zeros_like(pixels_2_to_1)
    pixels_2_to_1_normalized[:, 0] = (pixels_2_to_1[:, 0] / (W1 - 1)) * 2.0 - 1.0
    pixels_2_to_1_normalized[:, 1] = (pixels_2_to_1[:, 1] / (H1 - 1)) * 2.0 - 1.0
    grid_2_to_1 = pixels_2_to_1_normalized.reshape(H2, W2, 2).unsqueeze(0)  # (1, H2, W2, 2)
    
    sampled_depth_1 = F.grid_sample(
        depth_map_1.unsqueeze(0),
        grid_2_to_1,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(0)  # (1, H2, W2)
    
    # Compute depth differences (projected depth vs sampled depth)
    depth_diff_1_to_2 = (projected_depth_1_to_2 - sampled_depth_2) * valid_mask_1_to_2.float()
    depth_diff_2_to_1 = (projected_depth_2_to_1 - sampled_depth_1) * valid_mask_2_to_1.float()
    
    # Compute loss (L2) - only on valid pixels
    valid_count_1_to_2 = valid_mask_1_to_2.sum()
    valid_count_2_to_1 = valid_mask_2_to_1.sum()
    
    if valid_count_1_to_2 > 0 and valid_count_2_to_1 > 0:
        loss_1_to_2 = (depth_diff_1_to_2 ** 2).sum() / valid_count_1_to_2
        loss_2_to_1 = (depth_diff_2_to_1 ** 2).sum() / valid_count_2_to_1
        return (loss_1_to_2 + loss_2_to_1) / 2.0
    elif valid_count_1_to_2 > 0:
        return (depth_diff_1_to_2 ** 2).sum() / valid_count_1_to_2
    elif valid_count_2_to_1 > 0:
        return (depth_diff_2_to_1 ** 2).sum() / valid_count_2_to_1
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)

