# SeismicLab 文档

欢迎来到 SeismicLab 文档中心!

## 📚 文档目录

### [示例程序文档](demos.md)
详细介绍 SeismicLab 中所有示例程序的功能、方法、参数和可视化结果。包含16张生成的图片,涵盖以下主题:

- **去噪方法**: FX反褶积、中值滤波、POCS插值
- **动校正**: NMO/INMO校正、抛物线时差
- **Radon变换**: 多次波压制、高分辨率Radon
- **反褶积**: 稀疏反褶积、尖脉冲反褶积
- **其他技术**: Spitz插值、速度分析

---

## 📁 图片资源

所有示例程序生成的可视化图片都存储在 [figs/](figs/) 目录中,共16张PNG格式图片,分辨率150 DPI。

### 快速浏览

| 功能 | 图片文件 |
|------|----------|
| FX反褶积去噪 | [fx_decon_demo.png](figs/fx_decon_demo.png) |
| MED滤波 | [med_demo.png](figs/med_demo.png) |
| MED功率谱 | [med_demo_spectrum.png](figs/med_demo_spectrum.png) |
| NMO校正 | [moveout_demo.png](figs/moveout_demo.png) |
| 抛物线时差 | [parabolic_moveout_demo.png](figs/parabolic_moveout_demo.png) |
| POCS插值 | [pocs_demo.png](figs/pocs_demo.png) |
| POCS Wiggle | [pocs_demo_wiggle.png](figs/pocs_demo_wiggle.png) |
| Radon去多次(合成) | [radon_demo_1.png](figs/radon_demo_1.png) |
| Radon对比(合成) | [radon_demo_1_comparison.png](figs/radon_demo_1_comparison.png) |
| Radon去多次(实测) | [radon_demo_2.png](figs/radon_demo_2.png) |
| Radon时差谱 | [radon_demo_2_spectrum.png](figs/radon_demo_2_spectrum.png) |
| 稀疏反褶积 | [sparse_decon_demo.png](figs/sparse_decon_demo.png) |
| 尖脉冲反褶积 | [spiking_decon_demo.png](figs/spiking_decon_demo.png) |
| Spitz插值 | [spitz_demo.png](figs/spitz_demo.png) |
| Spitz Wiggle | [spitz_demo_wiggle.png](figs/spitz_demo_wiggle.png) |
| 速度分析 | [va_demo.png](figs/va_demo.png) |

---

## 🚀 快速开始

### 安装 SeismicLab

```bash
# 克隆仓库
git clone https://github.com/yourusername/seismiclab.git
cd seismiclab

# 使用conda创建环境
conda create -n dasQt python=3.12
conda activate dasQt

# 安装依赖
pip install numpy matplotlib scipy
pip install -e .
```

### 运行示例程序

```bash
# 运行单个示例
cd examples
python fx_decon_demo.py

# 运行所有示例
python run_all_demos.py

# 生成所有图片到docs/figs/
python generate_figs.py
```

---

## 📖 示例程序列表

SeismicLab 包含以下11个完整的示例程序(已生成16张可视化图片):

1. **fx_decon_demo.py** - FX预测反褶积去噪
2. **med_demo.py** - 最小熵反褶积滤波
3. **moveout_demo.py** - NMO/INMO动校正
4. **parabolic_moveout_demo.py** - 抛物线残余时差谱
5. **pocs_demo.py** - 凸集投影插值/去噪
6. **radon_demo_1.py** - 高分辨率Radon去多次(合成数据)
7. **radon_demo_2.py** - Radon去多次(实测数据)
8. **sparse_decon_demo.py** - 稀疏反褶积
9. **spiking_decon_demo.py** - 尖脉冲反褶积
10. **spitz_demo.py** - Spitz FX插值
11. **va_demo.py** - 速度分析

详细信息请查看 [示例程序文档](demos.md)。

---

## 🛠️ 技术支持

### 环境要求
- Python 3.8+
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- scipy >= 1.5.0

### 推荐环境
- conda环境: `dasQt`
- Python版本: 3.12

### 数据格式
- 输入: SU格式地震数据
- 输出: PNG格式图片(150 DPI)

---

## 📝 相关链接

- [主项目README](../README.md)
- [示例程序目录](../examples/)
- [API文档](../docs/api.md) (待添加)
- [贡献指南](../CONTRIBUTING.md) (待添加)

---

## 📧 联系方式

如有问题或建议,请提交 Issue 或 Pull Request。

---

**最后更新**: 2025-12-27

**SeismicLab 版本**: 0.1.0
