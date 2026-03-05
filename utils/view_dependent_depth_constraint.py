"""
视角依赖的深度约束损失（View-Dependent Depth Constraint Loss）

创新点3：根据视角-法线夹角自适应调整深度约束强度
- 正面视角（视角-法线夹角小）：强深度约束（因为正面视角深度更可靠）
- 侧面视角（视角-法线夹角大）：弱深度约束（因为侧面视角深度可能不准确）

数学公式：
    L_view = Σ w_view(x) · ||∇d(x)||²
    
其中：
    w_view(x) = exp(-λ_view_weight · (1 - cos(θ(x))))
    θ(x) 是视角-法线夹角
"""

import torch
import torch.nn.functional as F


def compute_view_direction(viewpoint_cam, H, W):
    """
    计算每个像素点的视角方向（从相机中心指向像素点的方向）
    
    使用相机内参和外参计算视角方向：
    1. 从像素坐标计算射线方向（相机坐标系）
    2. 转换到世界坐标系
    
    Args:
        viewpoint_cam: 相机对象，包含world_view_transform、full_proj_transform和camera_center
        H: 图像高度
        W: 图像宽度
    
    Returns:
        view_dirs: [H, W, 3] 视角方向（世界坐标系，已归一化）
    """
    device = viewpoint_cam.camera_center.device
    
    # 获取相机参数
    camera_center = viewpoint_cam.camera_center  # [3]
    
    # 创建像素坐标网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # 将像素坐标转换为NDC坐标（归一化设备坐标）
    # NDC坐标范围：[-1, 1]
    x_ndc = (x_coords / W) * 2.0 - 1.0  # [H, W]
    y_ndc = (y_coords / H) * 2.0 - 1.0  # [H, W]
    
    # 获取投影矩阵的逆矩阵（NDC到世界坐标）
    ndc2world = viewpoint_cam.ndc2world  # [4, 4]
    
    # 创建齐次坐标 [H, W, 4]
    # 使用深度=1.0（在NDC空间中）来计算方向向量
    # 注意：NDC坐标的z分量在[-1, 1]范围内，我们使用z=1.0（远平面）
    ones = torch.ones_like(x_ndc)
    ndc_coords = torch.stack([x_ndc, y_ndc, ones, ones], dim=-1)  # [H, W, 4]
    
    # 将NDC坐标转换为世界坐标
    # 使用齐次坐标变换
    world_coords = torch.matmul(ndc_coords, ndc2world.T)  # [H, W, 4]
    
    # 齐次坐标转换为3D坐标
    world_points = world_coords[..., :3] / (world_coords[..., 3:4] + 1e-8)  # [H, W, 3]
    
    # 计算视角方向（从相机中心指向像素点）
    view_dirs = world_points - camera_center.unsqueeze(0).unsqueeze(0)  # [H, W, 3]
    
    # 归一化视角方向（处理可能为零的向量）
    view_dirs_norm = torch.norm(view_dirs, p=2, dim=-1, keepdim=True)  # [H, W, 1]
    view_dirs = view_dirs / (view_dirs_norm + 1e-8)  # [H, W, 3]
    
    return view_dirs


def compute_view_normal_angle(view_dirs, normals):
    """
    计算视角-法线夹角
    
    Args:
        view_dirs: [H, W, 3] 视角方向（已归一化，从相机指向表面）
        normals: [3, H, W] 或 [H, W, 3] 表面法线（世界坐标系，从表面指向外）
    
    Returns:
        cos_theta: [H, W] 视角-法线夹角的余弦值
    """
    # 确保normals的形状是 [H, W, 3]
    if normals.dim() == 3 and normals.shape[0] == 3:
        # 从 [3, H, W] 转换为 [H, W, 3]
        normals = normals.permute(1, 2, 0)
    
    # 归一化法线（处理可能为零的法线）
    normals_norm = torch.norm(normals, p=2, dim=-1, keepdim=True)  # [H, W, 1]
    normals = normals / (normals_norm + 1e-8)  # [H, W, 3]
    
    # 计算点积（余弦值）
    # 注意：视角方向是从相机指向表面，法线是从表面指向外
    # 对于正面视角：view_dir ≈ -normal，所以 view_dir · normal ≈ -1
    # 对于侧面视角：view_dir ⊥ normal，所以 view_dir · normal ≈ 0
    # 我们想要cos(θ)，其中θ是视角和法线的夹角
    # cos(θ) = view_dir · normal（因为normal指向外，view_dir指向内，所以需要取负）
    # 但实际上，我们想要的是视角和法线的夹角，所以：
    # cos(θ) = -view_dir · normal（这样正面视角时cos(θ)接近1）
    cos_theta = -torch.sum(view_dirs * normals, dim=-1)  # [H, W]
    
    # 限制在合理范围内（避免边界值导致exp函数不稳定）
    # 使用[-0.99, 0.99]而不是[-1, 1]，避免exp函数在边界处不稳定
    cos_theta = torch.clamp(cos_theta, -0.99, 0.99)
    
    return cos_theta


def compute_depth_gradient(depth):
    """
    计算深度梯度 ||∇d||²（添加数值稳定性处理）
    
    Args:
        depth: [1, H, W] 或 [H, W] 深度图
    
    Returns:
        depth_grad_sq: [H, W] 深度梯度的平方和
    """
    # 确保深度图的形状是 [1, H, W]
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    elif depth.dim() == 3 and depth.shape[0] != 1:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    
    # 计算深度梯度
    grad_x = depth[:, :, 1:] - depth[:, :, :-1]  # [1, H, W-1]
    grad_y = depth[:, 1:, :] - depth[:, :-1, :]  # [1, H-1, W]
    
    # 添加梯度裁剪，避免异常值导致数值不稳定
    grad_x = torch.clamp(grad_x, -10.0, 10.0)
    grad_y = torch.clamp(grad_y, -10.0, 10.0)
    
    # 填充边界（使用零填充）
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='constant', value=0.0)  # [1, H, W]
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='constant', value=0.0)  # [1, H, W]
    
    # 计算梯度平方和（添加上限，避免异常值）
    depth_grad_sq = grad_x.squeeze(0) ** 2 + grad_y.squeeze(0) ** 2  # [H, W]
    depth_grad_sq = torch.clamp(depth_grad_sq, 0.0, 100.0)  # 限制最大值，避免异常值
    
    return depth_grad_sq


def view_dependent_depth_constraint_loss(
    render_pkg,
    viewpoint_cam,
    lambda_view_weight=2.0,
    mask_background=True
):
    """
    计算视角依赖的深度约束损失
    
    Args:
        render_pkg: 渲染结果字典，包含：
            - 'surf_depth': [1, H, W] 表面深度图
            - 'surf_normal': [3, H, W] 表面法线（世界坐标系）
            - 'rend_alpha': [1, H, W] 渲染的alpha值（用于mask背景）
        viewpoint_cam: 相机对象
        lambda_view_weight: 视角权重参数（默认2.0）
        mask_background: 是否mask背景区域（默认True）
    
    Returns:
        loss: 标量tensor，视角依赖的深度约束损失
    """
    # 获取渲染结果
    surf_depth = render_pkg['surf_depth']  # [1, H, W]
    surf_normal = render_pkg['surf_normal']  # [3, H, W]
    rend_alpha = render_pkg.get('rend_alpha', None)  # [1, H, W] 或 None
    
    H, W = surf_depth.shape[1], surf_depth.shape[2]
    
    # 1. 计算视角方向
    view_dirs = compute_view_direction(viewpoint_cam, H, W)  # [H, W, 3]
    
    # 2. 计算视角-法线夹角
    cos_theta = compute_view_normal_angle(view_dirs, surf_normal)  # [H, W]
    
    # 3. 计算视角依赖权重（改进版：更稳定的权重计算）
    # w_view(x) = exp(-λ_view_weight · (1 - cos(θ(x))))
    # 当cos(θ)接近1时（正面视角），权重接近1.0（强约束）
    # 当cos(θ)接近-1时（侧面视角），权重接近exp(-2λ)（弱约束）
    
    # Improved: Use linear interpolation instead of exp for better stability
    # Linear weight: w = 0.1 + 0.9 * (cos_theta + 1) / 2
    # This gives: cos_theta = 1 -> w = 1.0, cos_theta = -1 -> w = 0.1
    # More stable than exp function
    view_weight_linear = 0.1 + 0.9 * (cos_theta + 1.0) / 2.0  # [H, W]
    
    # Also compute exp-based weight but with reduced lambda
    lambda_view_weight_reduced = lambda_view_weight * 0.5  # Reduce exponential strength
    view_weight_exp = torch.exp(-lambda_view_weight_reduced * (1.0 - cos_theta))  # [H, W]
    
    # Blend both weights: 70% linear + 30% exp (linear is more stable)
    view_weight = 0.7 * view_weight_linear + 0.3 * view_weight_exp
    
    # 添加数值稳定性处理：限制权重范围，避免异常值
    view_weight = torch.clamp(view_weight, 0.05, 5.0)  # 更合理的权重范围
    
    # 4. 计算深度梯度（改进版：添加阈值保护）
    depth_grad_sq = compute_depth_gradient(surf_depth)  # [H, W]
    
    # 添加阈值保护：只约束超出阈值的梯度（避免过度约束已平滑的区域）
    grad_threshold = 0.001  # Only penalize gradients above this threshold
    excess_grad = torch.clamp(depth_grad_sq - grad_threshold, min=0.0)  # [H, W]
    
    # 5. 应用视角依赖权重
    weighted_depth_grad = view_weight * excess_grad  # [H, W]
    
    # 6. Mask背景区域（可选）
    if mask_background and rend_alpha is not None:
        # 使用alpha值作为mask（alpha < 0.5的区域视为背景）
        alpha_mask = (rend_alpha.squeeze(0) > 0.5).float()  # [H, W]
        weighted_depth_grad = weighted_depth_grad * alpha_mask
        
        # 计算有效像素数量（避免除零）
        valid_pixels = alpha_mask.sum()
        if valid_pixels > 0:
            loss = weighted_depth_grad.sum() / valid_pixels
        else:
            loss = torch.tensor(0.0, device=weighted_depth_grad.device, requires_grad=True)
    else:
        # 7. 计算平均损失
        loss = weighted_depth_grad.mean()
    
    return loss


def view_dependent_depth_constraint_loss_detailed(
    render_pkg,
    viewpoint_cam,
    lambda_view_weight=2.0,
    mask_background=True,
    return_components=False
):
    """
    计算视角依赖的深度约束损失（详细版本，返回中间结果用于调试）
    
    Args:
        render_pkg: 渲染结果字典
        viewpoint_cam: 相机对象
        lambda_view_weight: 视角权重参数
        mask_background: 是否mask背景区域
        return_components: 是否返回中间结果
    
    Returns:
        loss: 标量tensor，视角依赖的深度约束损失
        components (可选): 字典，包含中间结果
    """
    surf_depth = render_pkg['surf_depth']
    surf_normal = render_pkg['surf_normal']
    rend_alpha = render_pkg.get('rend_alpha', None)
    
    H, W = surf_depth.shape[1], surf_depth.shape[2]
    
    # 计算视角方向和夹角
    view_dirs = compute_view_direction(viewpoint_cam, H, W)
    cos_theta = compute_view_normal_angle(view_dirs, surf_normal)
    
    # 计算权重和梯度
    view_weight = torch.exp(-lambda_view_weight * (1.0 - cos_theta))
    depth_grad_sq = compute_depth_gradient(surf_depth)
    
    # 应用权重
    weighted_depth_grad = view_weight * depth_grad_sq
    
    # Mask背景
    if mask_background and rend_alpha is not None:
        alpha_mask = (rend_alpha.squeeze(0) > 0.5).float()
        weighted_depth_grad = weighted_depth_grad * alpha_mask
    
    loss = weighted_depth_grad.mean()
    
    if return_components:
        components = {
            'view_dirs': view_dirs,
            'cos_theta': cos_theta,
            'view_weight': view_weight,
            'depth_grad_sq': depth_grad_sq,
            'weighted_depth_grad': weighted_depth_grad,
            'alpha_mask': alpha_mask if mask_background and rend_alpha is not None else None
        }
        return loss, components
    
    return loss
