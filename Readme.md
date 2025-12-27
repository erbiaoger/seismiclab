# PyProject.toml 配置说明

本项目使用现代的 Python 打包标准 (PEP 517/518) 通过 `pyproject.toml` 进行配置。

## 📦 项目结构

```
SeismicLab/
├── pyproject.toml          # 项目配置文件
├── MANIFEST.in             # 打包清单
├── LICENSE                 # MIT 许可证
├── INSTALL.md              # 安装指南
├── test_installation.py    # 安装测试脚本
├── seismiclab_py/          # Python 包
│   ├── __init__.py
│   ├── io.py
│   ├── velan_nmo.py
│   └── ...
└── SeismicLab_demos_py/    # 演示脚本
    └── ...
```

## 🔧 pyproject.toml 配置说明

### 构建系统

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
```

使用 setuptools 作为构建后端，支持现代 Python 打包。

### 项目信息

```toml
[project]
name = "seismiclab-py"
version = "1.0.0"
```

- **name**: PyPI 包名
- **version**: 遵循语义化版本 (Semantic Versioning)

### 依赖管理

#### 核心依赖

```toml
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.8.0",
    "matplotlib>=3.3.0",
]
```

#### 可选依赖

```toml
[project.optional-dependencies]
dev = ["pytest", "black", "flake8", ...]
docs = ["sphinx", "sphinx-rtd-theme", ...]
examples = ["jupyter", "ipython", ...]
all = ["seismiclab-py[dev,docs,examples]"]
```

安装方式：
```bash
pip install -e ".[dev]"      # 开发依赖
pip install -e ".[docs]"     # 文档依赖
pip install -e ".[examples]" # 示例依赖
pip install -e ".[all]"      # 所有依赖
```

### 工具配置

#### Black (代码格式化)

```toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']
```

#### isort (导入排序)

```toml
[tool.isort]
profile = "black"
line_length = 100
```

#### mypy (类型检查)

```toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
```

#### pytest (测试)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

#### Coverage (代码覆盖率)

```toml
[tool.coverage.run]
source = ["seismiclab_py"]
```

## 📝 版本更新

更新版本号时，修改 `pyproject.toml` 中的：

```toml
version = "1.0.1"  # 或 1.1.0, 2.0.0 等
```

## 🚀 发布流程

### 1. 更新版本号

```toml
version = "1.0.1"
```

### 2. 构建包

```bash
pip install build
python -m build
```

这将创建：
- `dist/seismiclab_py-1.0.1.tar.gz` (源码包)
- `dist/seismiclab_py-1.0.1-py3-none-any.whl` (wheel包)

### 3. 检查包

```bash
pip install twine
twine check dist/*
```

### 4. 上传到 PyPI (测试)

```bash
twine upload --repository testpypi dist/*
```

### 5. 上传到 PyPI (生产)

```bash
twine upload dist/*
```

## 🧪 本地测试

### 安装到虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### 运行测试

```bash
python test_installation.py
```

### 运行演示

```bash
cd SeismicLab_demos_py
python fx_decon_demo.py
```

## 📚 相关资源

- [PEP 517](https://peps.python.org/pep-0517/) - 声明构建依赖
- [PEP 518](https://peps.python.org/pep-0518/) - 项目元数据
- [PEP 621](https://peps.python.org/pep-0621/) - pyproject.toml
- [Setuptools 文档](https://setuptools.pypa.io/)
- [Python 打包指南](https://packaging.python.org/)

## 🔍 故障排除

### 问题: 构建失败

```bash
# 清理构建文件
rm -rf build dist *.egg-info

# 重新构建
pip install --upgrade build
python -m build
```

### 问题: 导入错误

```bash
# 重新安装
pip uninstall seismiclab-py
pip install -e .
```

### 问题: 找不到包

```bash
# 检查安装路径
pip show seismiclab-py

# 确认 Python 路径
python -c "import sys; print(sys.path)"
```

## ✅ 检查清单

发布前检查：
- [ ] 版本号已更新
- [ ] `pyproject.toml` 配置正确
- [ ] 所有依赖已列出
- [ ] LICENSE 文件存在
- [ ] README.md 更新
- [ ] 测试通过
- [ ] 文档完整
- [ ] CHANGELOG.md 更新

## 📞 支持

如有问题，请：
1. 查看 [INSTALL.md](INSTALL.md)
2. 运行 `python test_installation.py`
3. 提交 GitHub Issue
