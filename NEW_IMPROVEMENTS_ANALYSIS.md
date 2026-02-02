# Unbiased Surfel 几何重建质量改进方案

## 一、问题深度分析

### 1.1 核心问题识别

经过深入分析代码和TSDF融合流程，发现以下**根本问题**：

#### **问题1：Median Depth选择的不稳定性**

当前实现（Eq. 9）：
```cpp
cum_opacity += (alpha + 0.1 * G);
if (cum_opacity < 0.6f) {
    median_depth = last_depth > 0 ? (last_depth + depth) * 0.5 : depth;
}
```

**关键问题**：
1. **`0.1 * G`项的不稳定性**：G是高斯值，随距离快速衰减，导致cum_opacity增长不稳定
2. **固定阈值0.6**：不考虑深度收敛程度，可能在深度未收敛时就选择深度
3. **深度平滑掩盖问题**：`(last_depth + depth) * 0.5`的平滑可能掩盖深度跳跃的真实问题
4. **没有考虑深度收敛信息**：选择深度时没有考虑该深度的收敛程度

#### **问题2：深度收敛损失的局限性**

当前实现：
```cpp
if(abs(depth - last_depth) > ConvergeThreshold)  // ConvergeThreshold = 1.0
    0  // 不惩罚
else
    min(G, last_G) * (depth - last_depth)^2
```

**关键问题**：
1. **阈值过大**：ConvergeThreshold = 1.0意味着深度差>1.0就不惩罚，这太宽松了
2. **只惩罚相邻高斯**：如果中间高斯被剪枝，深度差大的高斯对不会被直接惩罚
3. **没有考虑alpha权重**：应该更强烈地惩罚alpha大的高斯的深度差异

#### **问题3：TSDF融合时的深度不一致**

TSDF融合公式：`sdf = sampled_depth - z`

**关键问题**：
1. **不同视角的sampled_depth不一致**：即使单视角深度收敛，多视角融合时仍可能不一致
2. **深度图的平滑性不足**：相邻像素的深度可能跳跃，导致TSDF融合产生噪声
3. **没有考虑深度置信度**：所有深度的权重相同，应该给收敛好的深度更高权重

### 1.2 现有改进方案的问题

#### **全局深度收敛损失的问题**：
- **过度平滑**：强制所有高斯向平均深度收敛，可能丢失细节
- **计算开销**：需要计算加权平均深度，增加计算量
- **可能破坏局部结构**：如果某些区域确实需要深度变化，强制收敛可能不合适

#### **多视角一致性损失的问题**：
- **计算开销大**：需要渲染多个视角，显著增加训练时间
- **不稳定**：随机采样视角可能导致训练不稳定
- **可能冲突**：多视角一致性可能与单视角最优解冲突

#### **Alpha完整性损失的问题**：
- **过于激进**：强制alpha饱和可能导致过度填充
- **阈值选择困难**：valid_surface_mask的阈值选择困难
- **可能破坏细节**：在细节区域强制alpha饱和可能不合适

## 二、新的改进方案

### 2.1 改进Median Depth选择策略（核心改进）

#### **改进1：基于深度收敛度的自适应阈值**

**核心思想**：深度收敛越好，阈值越高（选择更靠前的深度）

```cpp
// 计算深度收敛度
float convergence_degree = 1.0f / (1.0f + converge_ray * 10.0f);  // 归一化到[0,1]
float adaptive_threshold = 0.5f + 0.2f * convergence_degree;  // 范围[0.5, 0.7]

// 使用自适应阈值
if (cum_opacity < adaptive_threshold) {
    // 选择深度时考虑收敛度
    if (convergence_degree > 0.7f) {
        // 深度收敛好，直接使用当前深度
        median_depth = depth;
    } else {
        // 深度未收敛，使用平滑深度
        median_depth = last_depth > 0 ? (last_depth + depth) * 0.5 : depth;
    }
}
```

#### **改进2：改进cum_opacity计算**

**核心思想**：移除不稳定的`0.1 * G`项，使用更稳定的alpha累积

```cpp
// 原公式：cum_opacity += (alpha + 0.1 * G);
// 改进：只使用alpha，但考虑alpha的权重
cum_opacity += alpha * (1.0f + 0.1f * G);  // G作为权重因子，而不是直接相加
// 或者更简单：cum_opacity += alpha;  // 直接使用alpha
```


### 2.2 改进深度收敛损失（关键改进）

#### **改进1：降低ConvergeThreshold**

**核心思想**：更严格地惩罚深度差异

```cpp
const float ConvergeThreshold = 0.5f;  // 从1.0降低到0.5
```

#### **改进2：加权深度收敛损失**

**核心思想**：更强烈地惩罚alpha大的高斯的深度差异

```cpp
// 原公式：min(G, last_G) * (depth - last_depth)^2
// 改进：使用alpha权重
float alpha_weight = (alpha + last_alpha) * 0.5f;
Converge += alpha_weight * min(G, last_G) * (depth - last_depth)^2;
```

#### **改进3：多尺度深度收敛**

**核心思想**：在不同深度尺度下惩罚深度差异

```cpp
// 计算深度差异的相对大小
float depth_diff_relative = abs(depth - last_depth) / (min(depth, last_depth) + 1e-6f);
// 在不同尺度下惩罚
if (depth_diff_relative > 0.1f) {  // 10%差异
    Converge += alpha_weight * min(G, last_G) * (depth - last_depth)^2;
}
```

### 2.3 空间深度平滑损失（新改进）

#### **核心思想**：在空间邻域内强制深度平滑

```python
def spatial_depth_smoothness_loss(surf_depth, lambda_smooth=0.1):
    """
    空间深度平滑损失：惩罚相邻像素的深度差异
    
    Args:
        surf_depth: 表面深度图 (1, H, W)
        lambda_smooth: 平滑权重
    """
    # 计算水平和垂直方向的深度梯度
    depth_grad_x = torch.abs(surf_depth[:, :, 1:] - surf_depth[:, :, :-1])
    depth_grad_y = torch.abs(surf_depth[:, 1:, :] - surf_depth[:, :-1, :])
    
    # 只在有效表面区域计算（alpha > 0.5）
    # valid_mask = (rend_alpha > 0.5).float()
    # depth_grad_x = depth_grad_x * valid_mask[:, :, 1:]
    # depth_grad_y = depth_grad_y * valid_mask[:, 1:, :]
    
    smooth_loss = lambda_smooth * (depth_grad_x.mean() + depth_grad_y.mean())
    return smooth_loss
```

**优势**：
- **直接改善TSDF融合**：平滑的深度图直接改善TSDF融合质量
- **计算开销小**：只需要计算梯度，开销很小
- **不破坏细节**：可以通过mask只在有效表面区域计算


### 2.5 渐进式深度收敛（新改进）

#### **核心思想**：在训练过程中逐步加强深度收敛约束

```python
# 训练早期：宽松的深度收敛
if iteration < 5000:
    lambda_converge = 0.0  # 不惩罚深度差异
elif iteration < 10000:
    lambda_converge = opt.lambda_converge * 0.5  # 轻度惩罚
else:
    lambda_converge = opt.lambda_converge  # 完全惩罚
```

**优势**：
- **训练稳定性**：早期不强制深度收敛，让模型先学习基本几何
- **逐步优化**：后期加强深度收敛，逐步优化几何质量

## 三、推荐实施方案

### 3.1 优先级排序

1. **高优先级（立即实施）**：
   - **改进2.1.2**：改进cum_opacity计算（移除0.1*G项）
   - **改进2.2.1**：降低ConvergeThreshold到0.5
   - **改进2.2.2**：加权深度收敛损失（使用alpha权重）

2. **中优先级（测试后实施）**：
   - **改进2.1.1**：基于深度收敛度的自适应阈值
   - **改进2.3**：空间深度平滑损失

3. **低优先级（可选）**：
   - **改进2.5**：渐进式深度收敛

### 3.2 实施建议

1. **先实施高优先级改进**：这些改进简单直接，风险低
2. **逐步添加中优先级改进**：测试每个改进的效果
3. **根据效果调整参数**：不同场景可能需要不同参数

### 3.3 预期效果

1. **深度选择更稳定**：移除0.1*G项后，cum_opacity增长更稳定
2. **深度收敛更好**：降低阈值和加权损失后，深度差异更小
3. **TSDF融合质量提升**：空间平滑损失直接改善TSDF融合
4. **几何重建质量提升**：整体几何重建质量应该显著提升

## 四、与现有方案的区别

### 4.1 关键区别

1. **不强制全局收敛**：不强制所有高斯向平均深度收敛，避免过度平滑
2. **不依赖多视角一致性**：避免计算开销和不稳定性
3. **直接改善深度选择**：从根源改善median depth选择策略
4. **空间平滑而非全局平滑**：只在空间邻域内平滑，保持细节


## 五、实施细节

### 5.1 CUDA代码修改

主要修改`forward.cu`：
1. 修改cum_opacity计算
2. 修改ConvergeThreshold
3. 修改深度收敛损失计算
4. 添加自适应阈值逻辑

### 5.2 Python代码修改

主要修改`train.py`：
1. 添加空间深度平滑损失
2. 添加渐进式深度收敛逻辑

### 5.3 参数配置

在`arguments/__init__.py`中添加：
- `lambda_depth_smooth`: 空间深度平滑损失权重（默认0.1）
- `converge_threshold`: 深度收敛阈值（默认0.5）

