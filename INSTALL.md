# SeismicLab Python - 安装指南

## 📦 安装 seismiclab-py

本项目提供了完整的 Python 版本的 SeismicLab 地震数据处理库。

### 方法 1: 使用 pip 从本地安装（推荐）

```bash
# 在项目根目录下
pip install -e .
```

### 方法 2: 使用 pyproject.toml 安装

```bash
# 在项目根目录下
pip install -e .
```

### 方法 3: 安装到虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装包
pip install -e .
```

## 📋 系统要求

- Python 3.8 或更高版本
- NumPy >= 1.20.0
- SciPy >= 1.8.0
- Matplotlib >= 3.3.0

## 🔧 安装可选依赖

### 开发依赖

```bash
pip install -e ".[dev]"
```

包含：
- pytest (测试)
- black (代码格式化)
- flake8 (代码检查)
- mypy (类型检查)
- isort (导入排序)

### 文档依赖

```bash
pip install -e ".[docs]"
```

包含：
- Sphinx (文档生成)
- sphinx-rtd-theme (ReadTheDocs 主题)

### 示例依赖

```bash
pip install -e ".[examples]"
```

包含：
- Jupyter Notebook
- IPython
- ipywidgets

### 安装所有依赖

```bash
pip install -e ".[all]"
```

## ✅ 验证安装

运行以下命令验证安装是否成功：

```python
# 测试导入
python -c "import seismiclab_py; print('✅ seismiclab_py 安装成功!')"

# 测试功能
python -c "from seismiclab_py import nmo, velan, fx_decon; print('✅ 核心功能正常!')"

# 运行演示脚本
cd SeismicLab_demos_py
python fx_decon_demo.py
```

## 🚀 快速开始

安装后，您可以使用以下方式导入和使用 seismiclab_py：

```python
import numpy as np
from seismiclab_py import (
    read_su, write_su,
    nmo, inmo, velan,
    fx_decon, med,
    pradon_demultiple,
    sparse_decon, spiking
)

# 读取 SU 数据
data, headers = read_su('data.su')

# 处理数据
dt = 0.004
h = headers[0]['offset']  # 偏移距
tnmo = [1.0, 2.0]
vnmo = [1500, 2000]

# NMO 校正
nmo_data = nmo(data, dt, h, tnmo, vnmo)

# FX 反褶积
denoised = fx_decon(nmo_data, dt, lf=5, mu=0.01, flow=1, fhigh=100)

# 保存结果
write_su('output.su', denoised, headers, dt)
```

## 📚 更多示例

查看 `SeismicLab_demos_py/` 目录中的演示脚本：

```bash
cd SeismicLab_demos_py

# 运行单个演示
python fx_decon_demo.py

# 运行所有演示
python run_all_demos.py
```

## 🐛 故障排除

### 问题 1: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'seismiclab_py'
```

**解决方案**:
- 确保在正确的目录下运行
- 或使用绝对路径导入：
  ```python
  import sys
  sys.path.append('/path/to/SeismicLab')
  from seismiclab_py import nmo
  ```

### 问题 2: SciPy 导入错误

```
ImportError: cannot import name 'svds' from 'scipy.linalg'
```

**解决方案**:
```bash
pip install --upgrade scipy
```

最低需要 SciPy 1.8.0，推荐 1.11.0+。

### 问题 3: Matplotlib 后端错误

```
UserWarning: Matplotlib is currently using agg
```

**解决方案**:
```python
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
```

## 📖 文档

完整文档请参阅：
- [README.md](README.md) - 项目概述
- [BUGFIXES.md](BUGFIXES.md) - 已知问题和修复
- [SeismicLab_demos_py/README.md](SeismicLab_demos_py/README.md) - 演示脚本说明
- [DEMO_COMPARISON.md](SeismicLab_demos_py/DEMO_COMPARISON.md) - MATLAB vs Python 对比

## 🤝 贡献

欢迎贡献！请：
1. Fork 本项目
2. 创建特性分支
3. 提交 Pull Request

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 🙏 致谢

- 原始 MATLAB 版本由 Mauricio D. Sacchi 开发
- Signal Analysis and Imaging Group (SAIG)
- University of Alberta

## 📮 联系方式

- 主页: http://seismic-lab.physics.ualberta.ca/
- 问题反馈: GitHub Issues
