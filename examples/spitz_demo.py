"""
Spitz FX插值示例：对NMO校正后的稀疏道集在频率-偏移域插值，生成稠密道集并以wiggle和幅度图对比。
"""

from __future__ import annotations  # 支持前向注解

import sys  # 系统路径
from pathlib import Path  # 路径工具

import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图

from seismiclab import clip, read_su, seismic_colormap, spitz_fx_interpolation, wigb  # 导入函数


DATA_DIR = Path("data")  # 数据目录


def main():
    d, headers = read_su(DATA_DIR / "gom_cdp_nmo.su")  # 读取NMO后数据
    h = np.array([hdr["offset"] for hdr in headers], dtype=float)  # 偏移距
    dt = float(headers[0]["dt"]) / 1_000_000.0  # 采样间隔秒

    n0 = 400  # 起始样点
    d = d[n0 - 1 :, :]  # 截取窗口
    nt, nh = d.shape  # 维度
    t0 = (n0 - 1) * dt  # 时间偏移
    taxis = t0 + np.arange(nt) * dt  # 时间轴
    h_interp = np.linspace(np.max(h), np.min(h), 2 * nh - 1)  # 插值偏移距

    npf = 25  # 预测步长
    pre1 = 1.0  # 预白化1
    pre2 = 1.0  # 预白化2
    flow = 0.1  # 低频
    fhigh = 90.0  # 高频

    di = spitz_fx_interpolation(d, dt, npf, pre1, pre2, flow, fhigh)  # FX插值

    fig, ax = plt.subplots(1, 2, figsize=(12, 10), sharey=True)  # wiggle对比
    plt.sca(ax[0]); wigb(d, 2, h, taxis); ax[0].set_title("Original")  # 原始wiggle
    plt.sca(ax[1]); wigb(di, 2, h_interp, taxis); ax[1].set_title("Spitz FX interpolation")  # 插值wiggle
    for a in ax:  # 坐标标签
        a.set_xlabel("Offset [m]")
        a.set_ylabel("Time [s]")
    fig.tight_layout()  # 紧凑布局

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 10), sharey=True)  # 幅度图对比
    ax2[0].imshow(clip(d, 60, 60), aspect="auto", extent=[h[0], h[-1], taxis[-1], taxis[0]], cmap=seismic_colormap())  # 原始幅度
    ax2[0].set_title("Original")  # 标题
    ax2[1].imshow(
        clip(di, 60, 60), aspect="auto", extent=[h_interp[0], h_interp[-1], taxis[-1], taxis[0]], cmap=seismic_colormap()
    )  # 插值幅度
    ax2[1].set_title("After interpolation")  # 标题
    for a in ax2:  # 坐标标签
        a.set_xlabel("Offset [m]")
        a.set_ylabel("Time [s]")
    fig2.tight_layout()  # 紧凑布局
    plt.show()  # 显示


if __name__ == "__main__":  # 入口
    main()  # 运行示例
