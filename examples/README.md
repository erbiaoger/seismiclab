# SeismicLab Python 演示脚本

这个目录包含从 MATLAB 转换的 Python 演示脚本，展示 SeismicLab 的各种功能。

## 演示脚本列表

| 演示脚本 | 描述 | 功能 |
|---------|------|-----|
| `fx_decon_demo.py` | FX 反褶积去噪 | 使用 FX 域反褶积压制随机噪声 |
| `med_demo.py` | 中值滤波去噪 | 使用中值滤波去除异常值 |
| `moveout_demo.py` | 动校正演示 | NMO 和抛物线时差校正 |
| `parabolic_moveout_demo.py` | 抛物线时差校正 | 抛物线时差校正演示 |
| `pocs_demo.py` | 凸集投影 | POCS 算法数据重建 |
| `radon_demo_1.py` | Radon 变换去多次波 | 使用抛物线 Radon 变换去除多次波 |
| `radon_demo_2.py` | Radon 变换重建 | Radon 变换数据重建 |
| `sparse_decon_demo.py` | 稀疏反褶积 | 稀疏反褶积处理 |
| `spiking_decon_demo.py` | 尖脉冲反褶积 | 尖脉冲反褶积恢复反射系数 |
| `spitz_demo.py` | Spitz 插值 | FX 域地震道插值 |
| `va_demo.py` | 速度分析 | 速度分析演示 |
| `run_spiking.py` | 运行尖脉冲反褶积 | 运行并保存尖脉冲反褶积结果 |

## 如何运行

### 运行单个演示

```bash
cd SeismicLab_demos_py
python fx_decon_demo.py
```

### 运行所有演示

```bash
cd SeismicLab_demos_py
python run_all_demos.py
```

## 依赖项

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- seismiclab_py (SeismicLab Python 库)

## 安装依赖

```bash
cd ..  # 回到项目根目录
pip install -r requirements.txt
```

## 演示脚本说明

### 1. FX 反褶积去噪 (`fx_decon_demo.py`)

展示如何使用 FX 域反褶积来压制地震数据中的随机噪声。

**主要步骤**:
1. 生成包含线性事件的合成地震数据
2. 添加随机噪声
3. 应用 FX 反褶积
4. 比较处理前后的信噪比

### 2. 中值滤波去噪 (`med_demo.py`)

展示中值滤波在地震数据去噪中的应用。

**主要步骤**:
1. 生成包含异常值的噪声数据
2. 应用中值滤波
3. 显示噪声压制效果

### 3. 动校正演示 (`moveout_demo.py`)

展示正常时差 (NMO) 校正和反 NMO 校正。

**主要步骤**:
1. 读取 CMP 道集
2. 应用 NMO 校正
3. 应用反 NMO 校正
4. 显示校正效果

### 4. 抛物线时差校正 (`parabolic_moveout_demo.py`)

展示抛物线时差校正。

### 5. 凸集投影 (`pocs_demo.py`)

展示使用凸集投影 (POCS) 进行数据重建。

### 6. Radon 变换去多次波 (`radon_demo_1.py`)

展示使用抛物线 Radon 变换去除多次波。

**主要步骤**:
1. 读取包含多次波的 CMP 道集
2. 应用 NMO 校正
3. 使用 Radon 变换分离一次波和多次波
4. 反 Radon 变换得到一次波
5. 应用反 NMO 校正

### 7. Radon 变换重建 (`radon_demo_2.py`)

展示 Radon 变换在数据重建中的应用。

### 8. 稀疏反褶积 (`sparse_decon_demo.py`)

展示稀疏反褶积算法。

**主要步骤**:
1. 生成合成地震记录
2. 应用稀疏反褶积
3. 显示反射系数恢复效果

### 9. 尖脉冲反褶积 (`spiking_decon_demo.py`)

展示尖脉冲反褶积。

**主要步骤**:
1. 生成或加载测试数据
2. 应用尖脉冲反褶积
3. 显示处理前后波形和频谱

### 10. Spitz 插值 (`spitz_demo.py`)

展示 FX 域地震道插值方法。

### 11. 速度分析 (`va_demo.py`)

展示速度分析流程。

## 与 MATLAB 版本的对比

所有 Python 演示脚本都对应 MATLAB 版本 (`SeismicLab_demos/` 目录下的 `.m` 文件)，使用相同的算法和参数设置。

主要差异:
- 文件扩展名: `.m` → `.py`
- 导入方式: MATLAB 自动路径 vs Python `sys.path`
- 数组索引: MATLAB 1-based vs Python 0-based
- 矩阵运算: MATLAB `.*` vs Python `*`

## 结果比较

可以运行以下命令来对比 MATLAB 和 Python 的输出结果:

```bash
# 运行 MATLAB 版本 (需要 MATLAB)
# cd SeismicLab_demos
# matlab -batch "run('fx_decon_demo.m')"

# 运行 Python 版本
cd SeismicLab_demos_py
python fx_decon_demo.py
```

## 故障排除

### 导入错误

如果遇到 `ModuleNotFoundError: No module named 'seismiclab_py'`，确保：

1. 你在项目根目录下运行脚本
2. 或者在脚本中设置正确的 `sys.path`

```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
```

### 数据文件缺失

某些演示需要读取数据文件。确保数据目录存在：

```bash
cd examples/data
ls *.su
```

如果缺失数据文件，大多数演示会自动生成合成数据。

## 贡献

欢迎贡献新的演示脚本或改进现有脚本！

## 许可证

MIT License (与原始 SeismicLab 项目相同)
