"""
实测数据Radon去多次示例：对NMO后的海上CDP剖面做高分辨率Radon反褶积，抑制多次并比较残余moveout谱与主反射结果。
"""

from __future__ import annotations  # 支持前向引用注解

import sys  # 系统路径
from pathlib import Path  # 路径工具

import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图

from seismiclab import clip, parabolic_moveout, pradon_demultiple, read_su, seismic_colormap  # 处理函数

DATA_DIR = Path("data")  # 数据目录


def main():
    d, headers = read_su(DATA_DIR / "gom_cdp_nmo.su")  # 读取NMO后海上CDP数据
    d = d[599:1200, :]  # MATLAB 600:1200 选取窗口
    mutes = np.ones_like(d)  # 静默矩阵
    mutes[d == 0] = 0  # 将零值设为静默

    h = np.array([hdr["offset"] for hdr in headers], dtype=float)  # 偏移距
    dt = float(headers[0]["dt"]) / 1_000_000.0  # 采样间隔秒

    qmin, qmax, nq = -0.9, 1.2, 180  # 残余时差范围
    flow, fhigh, mu, q_cut = 0.1, 90.0, 10.2, 0.05  # 频带与正则参数

    prim, M, tau, q = pradon_demultiple(d, dt, h, qmin, qmax, nq, flow, fhigh, mu, q_cut, "ls")  # Radon去多次
    prim = prim * mutes  # 应用静默

    Sd, tau_p, q_p = parabolic_moveout(d, dt, h, qmin, qmax, nq, 2, 15)  # 原数据moveout谱
    Sp, _, _ = parabolic_moveout(prim, dt, h, qmin, qmax, nq, 2, 15)  # 去多次后谱

    t0 = (600 - 1) * dt  # 起始时间偏移
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))  # 三幅图
    # MATLAB imagesc shows leftmost trace as smallest offset; reverse x-lims to match
    extent_data = [h[-1], h[0], t0 + tau[-1], t0 + tau[0]]  # 数据坐标范围
    extent_q = [q[0], q[-1], t0 + tau[-1], t0 + tau[0]]  # Radon坐标范围
    ax[0].imshow(clip(d, 50), aspect="auto", extent=extent_data, cmap=seismic_colormap())  # 原始数据
    ax[0].set_xlabel("Offset [ft]")  # x轴标签
    ax[0].set_ylabel("Time [s]")  # y轴标签
    ax[0].set_title("Input gather")  # 标题
    ax[1].imshow(clip(M, 50), aspect="auto", extent=extent_q, cmap=seismic_colormap())  # Radon面板
    ax[1].set_xlabel("Residual moveout [s]")  # q轴标签
    ax[1].set_title("Radon panel")  # 标题
    ax[2].imshow(clip(prim, 50), aspect="auto", extent=extent_data, cmap=seismic_colormap())  # 去多次结果
    ax[2].set_xlabel("Offset [ft]")  # x轴标签
    ax[2].set_title("Primaries")  # 标题
    fig.tight_layout()  # 紧凑布局

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))  # moveout谱图
    ax2[0].imshow(Sd, aspect="auto", extent=extent_q, cmap=seismic_colormap())  # 输入谱
    ax2[0].set_ylabel("Time [s]")  # y轴
    ax2[0].set_xlabel("Residual moveout [s]")  # x轴
    ax2[0].set_title("Moveout spectrum - input")  # 标题
    ax2[1].imshow(Sp, aspect="auto", extent=extent_q, cmap=seismic_colormap())  # Primaries谱
    ax2[1].set_ylabel("Time [s]")  # y轴
    ax2[1].set_xlabel("Residual moveout [s]")  # x轴
    ax2[1].set_title("Moveout spectrum - primaries")  # 标题
    fig2.tight_layout()  # 紧凑布局
    plt.show()  # 展示图形


if __name__ == "__main__":  # 入口
    main()  # 运行
