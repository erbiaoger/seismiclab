"""
最小熵反褶积（MED）示例：对近偏移现场数据进行迭代滤波，观察子波压缩、功率谱变化并计算逆滤波器长度。
"""

from __future__ import annotations  # 支持前向注解

import sys  # 系统路径
from pathlib import Path  # 路径处理

import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图

from seismiclab import ls_inv_filter, med, read_su, seismic_colormap, smooth_spectrum  # 处理函数
from seismiclab.med_alg import funalpha, dfunalpha  # MED非线性函数


DATA_DIR = Path("data")  # 数据目录


def main():
    s, headers = read_su(DATA_DIR / "gom_near_offsets.su")  # 读取近偏移SU数据
    s = s[399:800, :]  # MATLAB 400:800 子集

    wbp = np.array([1.0])  # 后置滤波
    dt = 0.004  # 采样间隔秒
    Nf = 21  # 滤波器长度
    mu = 0.01  # 正则
    Updates = 24  # 迭代次数

    filt, tf, x, tx, med_norm = med(wbp, s, dt, Nf, mu, Updates, funalpha, dfunalpha, 3)  # 运行MED

    Ps, f = smooth_spectrum(s, dt, 5, "li")[0:2]  # 原始功率谱
    Px, _ = smooth_spectrum(x, dt, 5, "li")[0:2]  # MED后功率谱

    nx = x.shape[1]  # 道数
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # 对比图
    clipd = 0.2 * np.max(np.abs(s))  # 色阶
    ax[0].imshow(
        s,
        aspect="auto",
        extent=[1, nx, tx[-1], tx[0]],
        cmap=seismic_colormap(),
        vmin=-clipd,
        vmax=clipd,
    )  # 输入
    ax[0].set_xlabel("Offset [m]")  # x轴
    ax[0].set_ylabel("Time [s]")  # y轴
    ax[0].set_title("Input")  # 标题
    ax[1].imshow(
        x,
        aspect="auto",
        extent=[1, nx, tx[-1], tx[0]],
        cmap=seismic_colormap(),
        vmin=-clipd,
        vmax=clipd,
    )  # 输出
    ax[1].set_xlabel("Offset [m]")  # x轴
    ax[1].set_title("After MED")  # 标题
    fig.tight_layout()  # 紧凑布局

    fig2, ax2 = plt.subplots(figsize=(6, 5))  # 频谱对比
    ax2.plot(f, Ps, label="Input")  # 原始谱
    ax2.plot(f, Px, label="After MED")  # 处理后谱
    ax2.set_xlabel("Frequency (Hz)")  # x轴
    ax2.legend()  # 图例
    ax2.set_title("Power spectra")  # 标题

    ls_filter, o = ls_inv_filter(filt.flatten(), 40, 20, 0.001)  # 最小二乘逆滤波器
    print("Least-squares inverse filter length:", len(ls_filter))  # 输出长度

    fig.savefig(Path(__file__).parent / "figs" / "med_demo.png", dpi=150, bbox_inches="tight")
    fig2.savefig(Path(__file__).parent / "figs" / "med_demo_spectrum.png", dpi=150, bbox_inches="tight")
    plt.show()  # 显示图形


if __name__ == "__main__":  # 入口
    main()  # 执行示例
