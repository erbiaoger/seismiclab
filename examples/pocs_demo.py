"""
POCS插值/去噪示例：在三维合成数据上随机抽零并加噪，用POCS迭代恢复缺失样点，展示干净、破坏和恢复切片的对比。
"""

from __future__ import annotations  # 支持前向注解

import sys  # 系统路径
from pathlib import Path  # 路径工具

import numpy as np  # 数值库
import matplotlib.pyplot as plt  # 绘图库

from seismiclab import add_noise, data_cube, pocs, seismic_colormap, wigb  # 处理函数


def main():
    dt = 0.004  # 采样间隔
    f0 = 30.0  # 主频
    nt = 140  # 时间采样数
    n1 = 80  # 空间维度1
    n2 = 60  # 空间维度2
    dx = [2.0, 2.0]  # 采样间距
    p = np.array([[0.04, -0.02], [0.03, 0.04]])  # 倾角
    t0 = np.array([0.1, 0.2])  # 起始时间
    A = np.array([-1.0, 1.0])  # 振幅
    snr = 9999  # 加噪信噪比
    L = 3  # 滤波长度

    d0 = data_cube([n1, n2], dt, f0, nt, dx, t0, p, A, "parabolic")  # 生成抛物线事件数据
    dn, _ = add_noise(d0, snr, L)  # 加噪声

    T = np.ones((n1, n2))  # 观测掩膜
    for k1 in range(n1):  # 遍历空间1
        for k2 in range(n2):  # 遍历空间2
            if np.random.randn() < 0.8:  # 随机抽零
                dn[:, k1, k2] = 0  # 清零数据
                T[k1, k2] = 0  # 掩膜置零

    f_low = 0.1  # 低频
    f_high = 90.0  # 高频
    option = 3  # POCS选项
    perc_i = 99.0  # 初始阈值百分比
    perc_f = 0.1  # 结束阈值百分比
    N = 100  # 最大迭代
    tol = 0.0001  # 收敛阈值
    a = 0.6  # 松弛因子
    dr, e1, e2, freq = pocs(dn, d0, T, dt, f_low, f_high, option, perc_i, perc_f, N, a, tol)  # 执行POCS

    c = np.max(np.abs(d0))  # 色阶
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))  # 三幅对比图
    ax[0].imshow(d0[:, :, 3], aspect="auto", vmin=-c, vmax=c, cmap=seismic_colormap())  # 干净数据
    ax[0].set_title("Clean signal")  # 标题
    ax[1].imshow(dn[:, :, 3], aspect="auto", vmin=-c, vmax=c, cmap=seismic_colormap())  # 抽零后
    ax[1].set_title("Noisy/decimated")  # 标题
    ax[2].imshow(dr[:, :, 4], aspect="auto", vmin=-c, vmax=c, cmap=seismic_colormap())  # 恢复后
    ax[2].set_title("Restored")  # 标题
    for a_ax in ax:  # 设置轴标签
        a_ax.set_xlabel("X2")
        a_ax.set_ylabel("Time sample")
    fig.tight_layout()  # 紧凑布局

    fig2 = plt.figure(figsize=(10, 4))  # wiggle展示
    plt.sca(plt.gca())  # 选择当前坐标轴
    wigb(np.hstack([d0[:, 4, :], dn[:, 4, :], dr[:, 4, :]]))  # 并排wiggle
    plt.title("Clean - Decimated - Restored (inline 5)")  # 标题
    plt.show()  # 显示


if __name__ == "__main__":  # 入口
    main()  # 运行示例
