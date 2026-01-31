"""
自适应Densification工具：用于在坑洞区域自动增加高斯
创新6：Adaptive Densification for Holes
"""

import torch
import torch.nn.functional as F
import numpy as np
from utils.general_utils import inverse_sigmoid, build_rotation
from utils.sh_utils import RGB2SH


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


def detect_hole_regions(surf_depth, rend_alpha, depth_var_threshold=0.01, alpha_threshold=0.5):
    """
    检测坑洞风险区域
    
    Args:
        surf_depth: 表面深度图 (1, H, W) 或 (H, W)
        rend_alpha: 渲染alpha图 (1, H, W) 或 (H, W)
        depth_var_threshold: 深度方差阈值
        alpha_threshold: alpha阈值
    
    Returns:
        hole_mask: 坑洞掩码 (H, W), True表示坑洞区域
        hole_risk: 坑洞风险图 (H, W)
    """
    # 确保是2D
    if surf_depth.dim() == 3:
        surf_depth = surf_depth.squeeze(0)
    if rend_alpha.dim() == 3:
        rend_alpha = rend_alpha.squeeze(0)
    
    # 计算深度方差
    depth_var = compute_spatial_variance(surf_depth, kernel_size=5)
    
    # 计算坑洞风险：深度方差大 + alpha低
    hole_risk = depth_var * (1.0 - rend_alpha)
    
    # 生成坑洞掩码
    hole_mask = (depth_var > depth_var_threshold) & (rend_alpha < alpha_threshold)
    
    return hole_mask, hole_risk


def pixel_depth_to_world_point(viewpoint, pixel_coords, depth):
    """
    将像素坐标和深度转换为3D世界坐标
    
    Args:
        viewpoint: 相机视角对象
        pixel_coords: 像素坐标 (x, y) 元组或 (N, 2) 张量
        depth: 深度值（标量或张量）
    
    Returns:
        world_points: 3D世界坐标 (N, 3)
    """
    device = depth.device if isinstance(depth, torch.Tensor) else torch.device("cuda")
    
    # 处理输入格式
    if isinstance(pixel_coords, tuple):
        # 单个像素 (x, y)
        x, y = pixel_coords
        pixel_coords_tensor = torch.tensor([[x, y]], device=device, dtype=torch.float32)
        if not isinstance(depth, torch.Tensor):
            depth = torch.tensor(depth, device=device, dtype=torch.float32)
        depth = depth.unsqueeze(0) if depth.dim() == 0 else depth
    else:
        pixel_coords_tensor = pixel_coords
        if not isinstance(depth, torch.Tensor):
            depth = torch.tensor(depth, device=device, dtype=torch.float32)
    
    if pixel_coords_tensor.dim() == 1:
        pixel_coords_tensor = pixel_coords_tensor.unsqueeze(0)
    if depth.dim() == 0:
        depth = depth.unsqueeze(0)
    
    N = pixel_coords_tensor.shape[0]
    H, W = viewpoint.image_height, viewpoint.image_width
    
    # 计算相机到世界的变换矩阵
    c2w = (viewpoint.world_view_transform.T).inverse()
    
    # 构建像素到NDC的变换（参考utils/point_utils.py）
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W - 1) / 2],
        [0, H / 2, 0, (H - 1) / 2],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device).T
    
    # 计算内参矩阵
    projection_matrix = c2w.T @ viewpoint.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T
    
    # 构建像素点（齐次坐标）
    pixels = torch.ones(N, 3, device=device)
    pixels[:, 0] = pixel_coords_tensor[:, 0]  # x
    pixels[:, 1] = pixel_coords_tensor[:, 1]  # y
    
    # 计算射线方向
    rays_d = pixels @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    
    # 计算3D点
    world_points = depth.unsqueeze(-1) * rays_d + rays_o
    
    return world_points


def create_gaussian_at_depth(viewpoint, pixel_coord, depth, normal, rgb=None, 
                            scene_extent=1.0, sh_degree=0):
    """
    在指定深度位置创建新高斯
    
    Args:
        viewpoint: 相机视角对象
        pixel_coord: 像素坐标 (y, x) 或 (x, y)
        depth: 深度值（标量）
        normal: 表面法线 (3,) 或 (3, 1)
        rgb: RGB颜色 (3,) 或 None（从图像采样）
        scene_extent: 场景范围，用于初始化scaling
        sh_degree: 球谐函数度数
    
    Returns:
        gaussian_params: 字典，包含高斯的所有参数
    """
    device = depth.device if isinstance(depth, torch.Tensor) else torch.device("cuda")
    
    # 转换像素坐标格式
    if isinstance(pixel_coord, tuple):
        y, x = pixel_coord
    else:
        x, y = pixel_coord[0], pixel_coord[1]
    
    # 转换为3D世界坐标
    world_point = pixel_depth_to_world_point(viewpoint, (y, x), depth)
    xyz = world_point.squeeze(0)  # (3,)
    
    # 处理法线
    if normal.dim() > 1:
        normal = normal.squeeze()
    normal = normal / (normal.norm() + 1e-8)  # 归一化
    
    # 从图像采样RGB（如果未提供）
    if rgb is None:
        y_int = int(y.clamp(0, viewpoint.image_height - 1))
        x_int = int(x.clamp(0, viewpoint.image_width - 1))
        rgb = viewpoint.original_image[:, y_int, x_int]  # (3,)
    else:
        if isinstance(rgb, torch.Tensor) and rgb.dim() > 1:
            rgb = rgb.squeeze()
    
    # 转换RGB到SH系数
    rgb_tensor = rgb.unsqueeze(0) if rgb.dim() == 0 else rgb
    if rgb_tensor.dim() == 1:
        rgb_tensor = rgb_tensor.unsqueeze(0)
    
    sh_dc = RGB2SH(rgb_tensor).squeeze(0)  # (3, 1)
    
    # 初始化SH rest（全零）
    # features_rest的形状应该是 (3, n_sh_rest)，需要transpose后变成 (n_sh_rest, 3)
    if sh_degree > 0:
        n_sh_rest = (sh_degree + 1) ** 2 - 1
        sh_rest = torch.zeros(3, n_sh_rest, device=device)  # (3, n_sh_rest)
    else:
        sh_rest = torch.zeros(3, 0, device=device)  # (3, 0)
    
    # 初始化opacity（中等值，让训练过程调整）
    opacity = inverse_sigmoid(torch.tensor(0.1, device=device))
    
    # 初始化scaling（基于场景范围）
    init_scale = scene_extent * 0.01
    scaling = torch.log(torch.tensor([init_scale, init_scale, init_scale], device=device))
    
    # 初始化rotation（基于法线方向）
    # 构建一个与法线对齐的旋转
    # 简化：使用法线作为z轴，构建旋转四元数
    z_axis = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
    if torch.abs(normal.dot(z_axis)) > 0.99:
        # 法线接近z轴，使用单位旋转
        rotation = torch.tensor([1, 0, 0, 0], device=device, dtype=torch.float32)
    else:
        # 计算旋转轴和角度
        rot_axis = torch.cross(z_axis, normal)
        rot_axis = rot_axis / (rot_axis.norm() + 1e-8)
        rot_angle = torch.acos(torch.clamp(normal.dot(z_axis), -1, 1))
        
        # 转换为四元数 (w, x, y, z)
        w = torch.cos(rot_angle / 2)
        xyz_quat = rot_axis * torch.sin(rot_angle / 2)
        rotation = torch.cat([w.unsqueeze(0), xyz_quat])
    
    return {
        'xyz': xyz,
        'features_dc': sh_dc,
        'features_rest': sh_rest,
        'opacity': opacity.unsqueeze(0),
        'scaling': scaling,
        'rotation': rotation
    }


def adaptive_densification_for_holes(gaussians, render_pkg, viewpoint, 
                                    depth_var_threshold=0.01, 
                                    alpha_threshold=0.5,
                                    max_new_gaussians=1000,
                                    scene_extent=1.0):
    """
    自适应Densification：在坑洞区域增加高斯
    
    Args:
        gaussians: GaussianModel对象
        render_pkg: 渲染结果字典，包含'surf_depth'和'rend_alpha'
        viewpoint: 相机视角对象
        depth_var_threshold: 深度方差阈值
        alpha_threshold: alpha阈值
        max_new_gaussians: 每次最多添加的高斯数量
        scene_extent: 场景范围
    
    Returns:
        num_added: 添加的高斯数量
    """
    surf_depth = render_pkg['surf_depth']  # (1, H, W)
    rend_alpha = render_pkg['rend_alpha']  # (1, H, W)
    
    # 检测坑洞区域
    hole_mask, hole_risk = detect_hole_regions(
        surf_depth, rend_alpha, 
        depth_var_threshold=depth_var_threshold,
        alpha_threshold=alpha_threshold
    )
    
    if not hole_mask.any():
        return 0
    
    # 获取坑洞像素坐标
    hole_pixels = torch.nonzero(hole_mask, as_tuple=False)  # (N, 2) [y, x]
    
    if hole_pixels.shape[0] == 0:
        return 0
    
    # 限制数量
    if hole_pixels.shape[0] > max_new_gaussians:
        # 按风险值排序，选择风险最高的
        risk_values = hole_risk[hole_pixels[:, 0], hole_pixels[:, 1]]
        _, indices = torch.sort(risk_values, descending=True)
        hole_pixels = hole_pixels[indices[:max_new_gaussians]]
    
    # 准备新高斯的参数
    new_xyz_list = []
    new_features_dc_list = []
    new_features_rest_list = []
    new_opacities_list = []
    new_scaling_list = []
    new_rotation_list = []
    
    # 获取法线（如果有）
    surf_normal = render_pkg.get('surf_normal', None)  # (3, H, W)
    
    # 为每个坑洞像素创建高斯
    for i in range(hole_pixels.shape[0]):
        y, x = hole_pixels[i, 0].item(), hole_pixels[i, 1].item()
        depth = surf_depth[0, y, x]
        
        # 获取法线
        if surf_normal is not None:
            normal = surf_normal[:, y, x]
        else:
            # 如果没有法线，使用默认值
            normal = torch.tensor([0, 0, 1], device=depth.device, dtype=torch.float32)
        
        # 创建高斯参数
        gaussian_params = create_gaussian_at_depth(
            viewpoint, (x, y), depth, normal,
            scene_extent=scene_extent,
            sh_degree=gaussians.max_sh_degree
        )
        
        new_xyz_list.append(gaussian_params['xyz'])
        new_features_dc_list.append(gaussian_params['features_dc'])
        new_features_rest_list.append(gaussian_params['features_rest'])
        new_opacities_list.append(gaussian_params['opacity'])
        new_scaling_list.append(gaussian_params['scaling'])
        new_rotation_list.append(gaussian_params['rotation'])
    
    if len(new_xyz_list) == 0:
        return 0
    
    # 堆叠所有参数
    new_xyz = torch.stack(new_xyz_list, dim=0)  # (N, 3)
    new_features_dc = torch.stack(new_features_dc_list, dim=0)  # (N, 3, 1)
    new_features_rest = torch.stack(new_features_rest_list, dim=0)  # (N, 3, M)
    new_opacities = torch.stack(new_opacities_list, dim=0)  # (N, 1)
    new_scaling = torch.stack(new_scaling_list, dim=0)  # (N, 3)
    new_rotation = torch.stack(new_rotation_list, dim=0)  # (N, 4)
    
    # 确保features_dc和features_rest的形状正确（需要transpose）
    # features_dc: (N, 3, 1) -> 正确
    # features_rest: (N, 3, M) -> 正确
    
    # 添加到高斯模型
    gaussians.densification_postfix(
        new_xyz, new_features_dc, new_features_rest,
        new_opacities, new_scaling, new_rotation
    )
    
    return len(new_xyz_list)

