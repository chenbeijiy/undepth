# 改进总结：与2DGS的区别

## 一、核心改动

### 1. **改进cum_opacity计算**（改进2.1.2）
- **原实现（论文Eq. 9）**：`cum_opacity += (alpha + 0.1) * G`
- **改进后**：`cum_opacity += alpha`
- **原因**：移除G项（G随距离快速衰减），使深度选择更稳定

### 2. **降低深度收敛阈值**（改进2.2.1）
- **原实现**：`ConvergeThreshold = 1.0`
- **改进后**：`ConvergeThreshold = 0.5`
- **原因**：更严格地惩罚深度差异，更早发现深度不一致

### 3. **加权深度收敛损失**（改进2.2.2）
- **原实现**：`Converge += min(G, last_G) * (depth - last_depth)^2`
- **改进后**：`Converge += alpha_weight * min(G, last_G) * (depth - last_depth)^2`
- **原因**：高alpha的高斯对渲染贡献更大，其深度一致性更重要

### 4. **自适应阈值选择深度**（改进2.1.1）
- **原实现**：固定阈值0.6
- **改进后**：自适应阈值0.5-0.7，根据深度收敛度动态调整
- **原因**：深度收敛好的区域可以更早选择深度，未收敛区域使用平滑深度

## 二、与2DGS的关键区别

### 2DGS的Median Depth选择
```cpp
// 2DGS: 当T > 0.5时选择深度
if (T > 0.5) {
    median_depth = depth;
}
```

### Unbiased Surfel的改进（原论文）
```cpp
// 原论文 Eq. 9: 使用累积不透明度
cum_opacity += (alpha + 0.1) * G;  // 注意：是(alpha+0.1)*G，不是alpha+0.1*G
if (cum_opacity < 0.6) {
    median_depth = (last_depth + depth) * 0.5;  // 平滑深度
}
```

### 我们的改进
```cpp
// 改进1: 移除G项（G随距离快速衰减导致不稳定），更稳定
cum_opacity += alpha;  // 原论文是 (alpha + 0.1) * G

// 改进2: 自适应阈值
float convergence_degree = compute_convergence_degree();
float adaptive_threshold = 0.5 + 0.2 * convergence_degree;  // [0.5, 0.7]

// 改进3: 根据收敛度选择深度
if (cum_opacity < adaptive_threshold) {
    if (convergence_degree > 0.7) {
        median_depth = depth;  // 收敛好，直接使用
    } else {
        median_depth = (last_depth + depth) * 0.5;  // 未收敛，平滑
    }
}
```

## 三、关键差异对比表

| 特性 | 2DGS | Unbiased Surfel (原) | 我们的改进 |
|------|------|---------------------|-----------|
| **深度选择时机** | T > 0.5 | cum_opacity < 0.6 | 自适应阈值 [0.5, 0.7] |
| **累积方式** | 使用T | alpha + 0.1*G | 仅alpha（更稳定） |
| **深度平滑** | 无 | 总是平滑 | 收敛好时不平滑 |
| **收敛阈值** | 无 | 1.0 | 0.5（更严格） |
| **损失权重** | 无 | 无 | alpha加权 |

## 四、改进效果

1. **深度选择更稳定**：移除0.1*G项，cum_opacity增长更可预测
2. **深度收敛更好**：降低阈值和加权损失，深度差异更小
3. **自适应优化**：根据收敛程度智能选择深度策略
4. **几何质量提升**：预期显著改善TSDF融合和mesh重建质量

## 五、修改的文件

1. `submodules/diff_surfel_rasterization/cuda_rasterizer/auxiliary.h`
   - ConvergeThreshold: 1.0 → 0.5

2. `submodules/diff_surfel_rasterization/cuda_rasterizer/forward.cu`
   - 改进cum_opacity计算
   - 实现自适应阈值
   - 实现加权深度收敛损失
   - 添加last_alpha跟踪

## 六、无需修改的文件

- **train.py**：无需修改，自动使用改进后的值
- **其他Python文件**：无需修改

---

**核心思想**：通过更稳定的深度选择策略和更严格的深度收敛约束，提升几何重建质量。

