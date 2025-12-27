# SeismicLab 演示脚本对比

本文档对比 MATLAB 和 Python 版本的演示脚本，确保功能一致性。

## 演示脚本映射

| # | MATLAB 脚本 | Python 脚本 | 状态 | 说明 |
|---|------------|------------|------|------|
| 1 | `fx_decon_demo.m` | `fx_decon_demo.py` | ✅ | FX 反褶积去噪 |
| 2 | `med_demo.m` | `med_demo.py` | ✅ | 中值滤波去噪 |
| 3 | `moveout_demo.m` | `moveout_demo.py` | ✅ | 动校正演示 |
| 4 | `parabolic_moveout_demo.m` | `parabolic_moveout_demo.py` | ✅ | 抛物线时差校正 |
| 5 | `pocs_demo.m` | `pocs_demo.py` | ✅ | 凸集投影 |
| 6 | `radon_demo_1.m` | `radon_demo_1.py` | ✅ | Radon 变换去多次波 |
| 7 | `radon_demo_2.m` | `radon_demo_2.py` | ✅ | Radon 变换重建 |
| 8 | `run_spiking.m` | `run_spiking.py` | ✅ | 运行尖脉冲反褶积 |
| 9 | `sparse_decon_demo.m` | `sparse_decon_demo.py` | ✅ | 稀疏反褶积 |
| 10 | `spiking_decon_demo.m` | `spiking_decon_demo.py` | ✅ | 尖脉冲反褶积 |
| 11 | `spitz_demo.m` | `spitz_demo.py` | ✅ | Spitz 插值 |
| 12 | `va_demo.m` | `va_demo.py` | ✅ | 速度分析 |

## 关键差异说明

### 1. 导入方式

**MATLAB**:
```matlab
% 自动添加路径
addpath('../codes')
% 直接调用函数
d = linear_events(dt, f0, tmax, h, tau, p, amp);
```

**Python**:
```python
# 需要手动设置路径
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# 导入函数
from seismiclab_py import linear_events

# 调用函数
d, _, _ = linear_events(dt, f0, tmax, h, tau, p, amp)
```

### 2. 数组索引

**MATLAB** (1-based):
```matlab
first_element = A(1)
last_element = A(end)
```

**Python** (0-based):
```python
first_element = A[0]
last_element = A[-1]
```

### 3. 矩阵运算

**MATLAB**:
```matlab
C = A * B      % 矩阵乘法
D = A .* B     % 逐元素乘法
E = A .^ 2     % 逐元素幂
```

**Python (NumPy)**:
```python
C = A @ B      # 矩阵乘法
D = A * B      # NumPy 默认逐元素运算
E = A ** 2     # 逐元素幂
```

### 4. 绘图

**MATLAB**:
```matlab
figure(1);
clf;
imagesc(data);
colormap(seismic(1));
colorbar;
```

**Python**:
```python
import matplotlib.pyplot as plt
from seismiclab_py import seismic_colormap

plt.figure(1)
plt.clf()
plt.imshow(data, cmap=seismic_colormap())
plt.colorbar()
```

### 5. 文件 I/O

**MATLAB**:
```matlab
[D, H] = readsegy('data.su');
writesegy('output.su', D, H);
```

**Python**:
```python
from seismiclab_py import read_su, write_su

data, headers = read_su('data.su')
write_su('output.su', data, headers, dt)
```

## 运行演示

### MATLAB 版本

```matlab
% 在 MATLAB 命令行中
cd SeismicLab_demos
fx_decon_demo
```

### Python 版本

```bash
# 在命令行中
cd SeismicLab_demos_py
python fx_decon_demo.py
```

## 输出对比

预期输出应该基本一致，但可能存在细微差异：

1. **数值精度**: Python 使用 float64，与 MATLAB 相同
2. **随机数**: 需要设置相同的种子才能得到相同结果
3. **图像显示**: 颜色映射可能略有不同

### 设置随机种子 (Python)

```python
import numpy as np
np.random.seed(0)  # 固定随机种子
```

### 设置随机种子 (MATLAB)

```matlab
rng(0);  % 固定随机种子
```

## 性能对比

通常情况下：

- **小数据集**: MATLAB 和 Python 性能相近
- **大数据集**: Python (NumPy/SciPy) 可能更快，因为使用了优化的 BLAS/LAPACK
- **显式循环**: MATLAB 的 JIT 编译可能更快

## 验证一致性

要验证 Python 和 MATLAB 结果的一致性：

1. 在 MATLAB 中运行演示并保存结果
2. 在 Python 中运行演示并保存结果
3. 使用 Python 加载 MATLAB 结果并比较

```python
import scipy.io as sio
import numpy as np

# 加载 MATLAB 结果
mat = sio.loadmat('matlab_result.mat')
matlab_result = mat['variable_name']

# 加载 Python 结果
python_result = np.load('python_result.npz')['variable_name']

# 比较误差
error = np.abs(matlab_result - python_result)
max_error = np.max(error)
print(f"Maximum error: {max_error}")

# 相对误差
relative_error = error / (np.abs(matlab_result) + 1e-10)
print(f"Max relative error: {np.max(relative_error)}")
```

## 常见问题

### Q: 为什么 Python 版本运行更慢？

A: 可能的原因：
1. 首次运行需要编译/导入模块
2. 没有使用向量化操作
3. Python 版本包含了额外的可视化代码

### Q: 图像显示不一致？

A: 检查：
1. 颜色映射是否相同 (`seismic_colormap()`)
2. 数据范围是否相同 (vmin, vmax)
3. 插值方法是否相同

### Q: 数值结果有差异？

A: 检查：
1. 随机种子是否相同
2. 浮点精度是否相同 (float64)
3. 算法实现是否完全一致

## 未来改进

- [ ] 添加单元测试验证数值一致性
- [ ] 创建 Jupyter notebook 交互式演示
- [ ] 添加性能基准测试
- [ ] 支持命令行参数
- [ ] 生成 HTML 报告
