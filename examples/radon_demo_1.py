"""
高分辨率Radon去多次示例：合成CMP道集经NMO校正后做抛物线Radon变换分离多次，再反变换得到初至并查看Radon面板。
"""

from __future__ import annotations  # 允许前向类型声明

import sys  # 访问系统路径
from pathlib import Path  # 处理跨平台路径

import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图

from seismiclab import clip, inmo, nmo, pradon_demultiple, read_su, seismic_colormap  # 导入处理函数


DATA_DIR = Path("data")  # 数据目录


def main():
    d, headers = read_su(DATA_DIR / "syn_cmp_mult.su")  # 读取合成多次波CMP数据
    h = np.array([hdr["offset"] for hdr in headers], dtype=float)  # 提取偏移距
    dtsec = float(headers[0]["dt"]) / 1_000_000.0  # 采样间隔秒

    tnmo = np.array([1.0, 2.0])  # NMO时深点
    vnmo = np.array([1500.0, 2000.0])  # 对应速度
    max_stretch = 50.0  # 最大拉伸百分比
    d_nmo, _, tau_axis, _ = nmo(d, dtsec, h, tnmo, vnmo, max_stretch)  # 执行NMO校正

    qmin, qmax, nq = -0.3, 0.8, 60  # 残余时差参数范围
    flow, fhigh, mu, q_cut = 1.0, 60.0, 1.0, 0.01  # 频带、正则与截断
    prim_est, M, tau, q = pradon_demultiple(d_nmo, dtsec, h, qmin, qmax, nq, flow, fhigh, mu, q_cut, "hr")  # Radon去多次

    prim_time, _, _, _ = inmo(prim_est, dtsec, h, tnmo, vnmo, max_stretch)  # 反NMO回到原时间

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))  # 三幅并排图
    clim_d = 0.9 * np.max(np.abs(d_nmo))  # 数据色阶
    clim_m = 0.9 * np.max(np.abs(M))  # Radon面板色阶
    im0 = ax[0].imshow(
        clip(d_nmo, 90),  # 限幅后的NMO数据
        aspect="auto",  # 自适应纵横比
        extent=[h[0], h[-1], tau_axis[-1], tau_axis[0]],  # 坐标范围
        cmap=seismic_colormap(),  # 地震灰度
        vmin=-clim_d,  # 色阶下限
        vmax=clim_d,  # 色阶上限
    )
    im1 = ax[1].imshow(
        M,  # Radon面板
        aspect="auto",
        extent=[q[0], q[-1], tau[-1], tau[0]],  # q-时间轴
        cmap=seismic_colormap(),
        vmin=-clim_m,
        vmax=clim_m,
    )
    im2 = ax[2].imshow(
        prim_est,  # 初至估计
        aspect="auto",
        extent=[h[0], h[-1], tau[-1], tau[0]],  # 偏移距-时间
        cmap=seismic_colormap(),
        vmin=-clim_d,
        vmax=clim_d,
    )
    ax[0].set_xlabel("Offset [m]"); ax[0].set_ylabel("Time [s]"); ax[0].set_title("NMO corrected")  # 标注左图
    ax[1].set_xlabel("q [s]"); ax[1].set_title("Radon panel")  # 标注中图
    ax[2].set_xlabel("Offset [m]"); ax[2].set_title("Primaries after Radon")  # 标注右图
    fig.colorbar(im0, ax=ax[0], shrink=0.7)  # 左图色条
    fig.colorbar(im1, ax=ax[1], shrink=0.7)  # 中图色条
    fig.colorbar(im2, ax=ax[2], shrink=0.7)  # 右图色条
    fig.tight_layout()  # 紧凑布局

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))  # 第二组图
    full_time = np.arange(d.shape[0]) * dtsec  # 原始时间轴
    clip_val = 0.9 * np.max(np.abs(d))  # 原数据色阶
    ax2[0].imshow(
        d, aspect="auto", extent=[h[0], h[-1], full_time[-1], full_time[0]], cmap=seismic_colormap(), vmin=-clip_val, vmax=clip_val
    )  # 原始数据
    ax2[1].imshow(
        prim_time,
        aspect="auto",
        extent=[h[0], h[-1], full_time[-1], full_time[0]],
        cmap=seismic_colormap(),
        vmin=-clip_val,
        vmax=clip_val,
    )  # 反NMO后的初至
    ax2[0].set_xlabel("Offset [m]"); ax2[0].set_ylabel("Time [s]"); ax2[0].set_title("Original")  # 左图标注
    ax2[1].set_xlabel("Offset [m]"); ax2[1].set_title("Primaries after INMO")  # 右图标注
    fig2.tight_layout()  # 紧凑布局
    fig.savefig(Path(__file__).parent / "figs" / "radon_demo_1.png", dpi=150, bbox_inches="tight")
    fig2.savefig(Path(__file__).parent / "figs" / "radon_demo_1_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()  # 显示全部图形


if __name__ == "__main__":  # 脚本入口
    main()  # 运行示例
