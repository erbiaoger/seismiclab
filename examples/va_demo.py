"""
速度分析示例：对合成CMP道集扫描速度谱，生成tau-v能量图并与原始道集并排展示以辅助选取速度。
"""

from __future__ import annotations  # 支持前向注解

import sys  # 系统路径
from pathlib import Path  # 路径工具

import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图

from seismiclab import read_su, seismic_colormap, velan  # 导入函数


DATA_DIR = Path("data")  # 数据目录


def main():
    d, headers = read_su(DATA_DIR / "syn_cmp.su")  # 读取合成CMP
    h = np.array([hdr["offset"] for hdr in headers], dtype=float)  # 偏移距
    dtsec = float(headers[0]["dt"]) / 1_000_000.0  # 采样间隔秒

    vmin, vmax, nv = 1000.0, 4000.0, 150  # 速度扫描范围与步数
    R, L = 1, 15  # 正则参数与窗长
    S, tau, v = velan(d, dtsec, h, vmin, vmax, nv, R, L)  # 速度分析

    nt, _ = d.shape  # 样本数
    t_axis = np.arange(nt) * dtsec  # 时间轴
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=False)  # 双图并排
    ax[0].imshow(d, aspect="auto", extent=[h[0], h[-1], t_axis[-1], t_axis[0]], cmap=seismic_colormap())  # CMP剖面
    ax[0].set_xlabel("Offset [m]")  # x轴
    ax[0].set_ylabel("Time [s]")  # y轴
    ax[0].set_title("CMP gather")  # 标题
    im = ax[1].imshow(
        S, aspect="auto", extent=[v[0], v[-1], tau[-1], tau[0]], cmap=seismic_colormap(), vmin=-0.1, vmax=0.9
    )  # tau-v能量
    ax[1].set_xlabel("Velocity [m/s]")  # x轴
    ax[1].set_title("Velocity analysis")  # 标题
    fig.colorbar(im, ax=ax[1])  # 色条
    fig.tight_layout()  # 紧凑布局
    plt.show()  # 显示


if __name__ == "__main__":  # 入口
    main()  # 运行示例
