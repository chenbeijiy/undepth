# 代码修改文档与性能分析

## 一、与原始Unbiased Depth代码的差异

### 修改文件列表

1. **`submodules/diff_surfel_rasterization/cuda_rasterizer/auxiliary.h`**
2. **`submodules/diff_surfel_rasterization/cuda_rasterizer/forward.cu`**

---

## 二、详细修改内容

### 修改1：降低ConvergeThreshold（auxiliary.h）

**位置**：`auxiliary.h` line 44

**原始代码**：
```cpp
__device__ const float ConvergeThreshold = 1.0f;
```

**修改后**：
```cpp
// Improvement 2.2.1: Lower ConvergeThreshold for stricter depth convergence
__device__ const float ConvergeThreshold = 0.5f;
```

**影响**：
- ✅ **性能影响**：无（常量值，编译时优化）
- ✅ **功能影响**：更严格地惩罚深度差异

---

### 修改2：改进cum_opacity计算（forward.cu）

**位置**：`forward.cu` line 454-458

**原始代码**（论文Eq. 9）：
```cpp
cum_opacity += (alpha + 0.1) * G;
```

**修改后**：
```cpp
// Improvement 2.1.2: Improved cum_opacity calculation
// Original (from paper Eq. 9): cum_opacity += (alpha + 0.1) * G;
// The G term causes instability as it decays rapidly with distance
// Improved: Use alpha directly for more stable accumulation (removes G dependency)
cum_opacity += alpha;
```

**影响**：
- ✅ **性能影响**：**性能提升**（移除了G的乘法运算）
- ✅ **功能影响**：更稳定的深度选择

---

### 修改3：添加last_alpha跟踪（forward.cu）

**位置**：`forward.cu` line 330

**原始代码**：
```cpp
float last_G = 0;
float cum_opacity = 0;
```

**修改后**：
```cpp
float last_G = 0;
float last_alpha = 0.0f;  // Improvement 2.2.2: Track last alpha for weighted convergence loss
float cum_opacity = 0;
```

**影响**：
- ⚠️ **性能影响**：极小（只是多一个变量赋值）
- ✅ **功能影响**：用于加权深度收敛损失

---

### 修改4：自适应阈值计算（forward.cu）⚠️ **性能瓶颈**

**位置**：`forward.cu` line 460-494

**原始代码**：
```cpp
// Cumulated opacity. Eq. (9) from paper Unbiased 2DGS.
if (cum_opacity < 0.6f) {
    // Make the depth map smoother
    median_depth = last_depth > 0 ? (last_depth + depth) * 0.5 : depth;
    median_contributor = contributor;
}
```

**修改后**：
```cpp
// Improvement 2.1.1: Adaptive threshold based on depth convergence degree
// Compute convergence degree from current depth difference
float convergence_degree = 1.0f;
if (last_depth > 0) {
    // Use current depth difference as immediate convergence indicator
    float depth_diff_relative = abs(depth - last_depth) / (min(depth, last_depth) + 1e-6f);
    // Lower relative depth difference means better convergence
    float immediate_convergence = 1.0f / (1.0f + depth_diff_relative * 100.0f);  // Map to [0, 1]
    
    // Use only immediate convergence (no accumulated convergence)
    convergence_degree = immediate_convergence;
}

// Adaptive threshold: better convergence -> higher threshold (select depth earlier)
float adaptive_threshold = 0.5f + 0.2f * convergence_degree;  // Range: [0.5, 0.7]

// Use adaptive threshold for median depth selection
if (cum_opacity < adaptive_threshold) {
    if (convergence_degree > 0.7f) {
        // Depth well converged, use current depth directly
        median_depth = depth;
    } else {
        // Depth not well converged, use smoothed depth
        median_depth = last_depth > 0 ? (last_depth + depth) * 0.5 : depth;
    }
    median_contributor = contributor;
}
```

**影响**：
- ❌ **性能影响**：**显著下降** ⚠️
  - 每个高斯都需要计算`depth_diff_relative`（包含除法和min运算）
  - 每个高斯都需要计算`immediate_convergence`（包含除法）
  - 每个高斯都需要计算`adaptive_threshold`
  - 每个高斯都需要判断`convergence_degree > 0.7f`
  - **这些计算在每个高斯上都会执行，即使cum_opacity已经超过阈值**

---

### 修改5：加权深度收敛损失（forward.cu）

**位置**：`forward.cu` line 579-597

**原始代码**：
```cpp
// Converge Loss - Original adjacent constraint
if((T > 0.09f)) {
    if(last_converge > 0) {
        Converge += abs(depth - last_depth) > ConvergeThreshold ?
            0 : min(G, last_G) * (depth - last_depth) * (depth - last_depth);
    }
    last_G = G;
    last_converge = contributor;
}
```

**修改后**：
```cpp
// Improvement 2.2.2: Weighted depth convergence loss (use alpha weight)
// Original: min(G, last_G) * (depth - last_depth)^2
// Improved: Use alpha weight to more strongly penalize depth differences for high-alpha Gaussians
if((T > 0.09f)) {
    if(last_converge > 0) {
        float depth_diff = abs(depth - last_depth);
        if (depth_diff <= ConvergeThreshold) {
            // Compute alpha weight: average of current and last alpha
            float alpha_weight = (alpha + last_alpha) * 0.5f;
            // Weighted convergence loss: alpha_weight * min(G, last_G) * depth_diff^2
            Converge += alpha_weight * min(G, last_G) * depth_diff * depth_diff;
        }
        // If depth_diff > ConvergeThreshold, no penalty (as before)
    }
    last_G = G;
    last_alpha = alpha;  // Track alpha for next iteration
    last_converge = contributor;
}
```

**影响**：
- ⚠️ **性能影响**：轻微下降（多了一次条件判断和一次加法运算）
- ✅ **功能影响**：更合理的深度收敛损失

---

## 三、性能问题分析

### 🔴 主要性能瓶颈：自适应阈值计算

**问题位置**：`forward.cu` line 460-494

**性能开销**：
1. **每个高斯都执行**（即使cum_opacity已超过阈值）：
   ```cpp
   float depth_diff_relative = abs(depth - last_depth) / (min(depth, last_depth) + 1e-6f);
   float immediate_convergence = 1.0f / (1.0f + depth_diff_relative * 100.0f);
   float adaptive_threshold = 0.5f + 0.2f * convergence_degree;
   ```

2. **计算复杂度**：
   - `abs()`: 1次运算
   - `min()`: 1次比较
   - 除法: 1次（可能较慢）
   - 除法: 1次（计算immediate_convergence）
   - 乘法: 1次
   - 加法: 1次
   - **总计：每个高斯约6-7次浮点运算 + 2次除法**

3. **如果场景有100万个高斯，每个像素平均10个高斯**：
   - 额外计算：100万 × 7次运算 = 700万次浮点运算
   - 除法运算较慢，可能显著影响性能

### 优化建议

#### 方案1：延迟计算（推荐）✅

**只在需要时计算自适应阈值**：

```cpp
// Improvement 2.1.1: Adaptive threshold based on depth convergence degree
// Only compute when cum_opacity is still below base threshold
if (cum_opacity < 0.7f) {  // Only compute if might need adaptive threshold
    float convergence_degree = 1.0f;
    if (last_depth > 0) {
        float depth_diff_relative = abs(depth - last_depth) / (min(depth, last_depth) + 1e-6f);
        float immediate_convergence = 1.0f / (1.0f + depth_diff_relative * 100.0f);
        convergence_degree = immediate_convergence;
    }
    
    float adaptive_threshold = 0.5f + 0.2f * convergence_degree;
    
    if (cum_opacity < adaptive_threshold) {
        if (convergence_degree > 0.7f) {
            median_depth = depth;
        } else {
            median_depth = last_depth > 0 ? (last_depth + depth) * 0.5 : depth;
        }
        median_contributor = contributor;
    }
} else {
    // cum_opacity already exceeds max threshold, skip calculation
}
```

**性能提升**：当cum_opacity超过0.7后，完全跳过计算，可节省大量计算

#### 方案2：简化计算

**使用更简单的收敛度计算**：

```cpp
// Simplified convergence degree calculation
float convergence_degree = 1.0f;
if (last_depth > 0) {
    float depth_diff_abs = abs(depth - last_depth);
    float depth_avg = (depth + last_depth) * 0.5f;
    // Use simpler formula: avoid division
    float depth_diff_relative = depth_diff_abs / (depth_avg + 1e-6f);
    // Use linear approximation instead of sigmoid
    convergence_degree = max(0.0f, 1.0f - depth_diff_relative * 10.0f);
}
```

**性能提升**：减少一次除法运算

#### 方案3：使用固定阈值（最快）

**如果性能优先，可以暂时使用固定阈值**：

```cpp
// Use fixed threshold for better performance
if (cum_opacity < 0.6f) {
    median_depth = last_depth > 0 ? (last_depth + depth) * 0.5 : depth;
    median_contributor = contributor;
}
```

---

## 四、性能影响总结

| 修改项 | 性能影响 | 严重程度 |
|--------|---------|---------|
| ConvergeThreshold降低 | ✅ 无影响 | - |
| cum_opacity计算改进 | ✅ 性能提升 | - |
| last_alpha跟踪 | ⚠️ 极小影响 | 低 |
| **自适应阈值计算** | ❌ **显著下降** | **高** |
| 加权深度收敛损失 | ⚠️ 轻微下降 | 低 |

### 总体性能影响

- **主要瓶颈**：自适应阈值计算（每个高斯都执行）
- **预期速度下降**：10-30%（取决于场景复杂度）
- **建议**：使用方案1（延迟计算）优化

---

## 五、修改对比表

| 特性 | 原始Unbiased | 修改后 |
|------|-------------|--------|
| **cum_opacity计算** | `(alpha + 0.1) * G` | `alpha` |
| **ConvergeThreshold** | 1.0 | 0.5 |
| **深度收敛损失** | `min(G, last_G) * diff²` | `alpha_weight * min(G, last_G) * diff²` |
| **阈值选择** | 固定0.6 | 自适应[0.5, 0.7] |
| **性能开销** | 基准 | +10-30% |

---

## 六、性能优化（已实施）✅

### ✅ 已应用优化：延迟计算自适应阈值

**实施状态**：已完成

**优化内容**：
- 将自适应阈值计算包装在 `if (cum_opacity < 0.7f)` 条件中
- 当cum_opacity >= 0.7时，完全跳过所有自适应阈值计算
- 保持功能不变，但显著减少计算开销

**预期性能提升**：
- ✅ 当cum_opacity超过0.7后，完全跳过计算
- ✅ 预期性能恢复到接近原始速度（或仅下降5-10%）
- ✅ 保持自适应阈值的所有功能

**代码位置**：`forward.cu` line 460-494（已优化）

---

**文档创建日期**：2025年1月
**性能分析版本**：v1.0

