"""
FX预测反褶积去噪示例：生成线性事件合成道集，加噪声，然后用fx_decon压制随机噪声并对比信噪比。
"""

from __future__ import annotations  # 支持前向注解

import sys  # 系统路径
from pathlib import Path  # 路径工具

import numpy as np  # 数值运算
import matplotlib.pyplot as plt  # 绘图

from seismiclab import add_noise, fx_decon, linear_events, quality, seismic_colormap  # 导入处理函数


def main():
    dt   = 0.004                                  # 采样间隔
    f0   = 30.0                                   # 主频
    tmax = 1.0                                    # 最长时间
    h    = np.arange(0, 80)                       # 偏移距
    tau  = [0.4, 0.12, 0.22, 0.9, 1.3]            # 事件到时
    p    = [0.0, 0.001, 0.0012, -0.0011, -0.001]  # 倾角
    amp  = [-1, 1, 2, 1, -1]                      # 振幅
    snr  = 2                                      # 信噪比
    L    = 5                                      # 噪声滤波长度

    d0, _, _ = linear_events(dt, f0, tmax, h, tau, p, amp)  # 生成线性事件
    dn, _ = add_noise(d0, snr, L)                           # 加噪声

    lf    = 5                                      # 滤波器长度
    mu    = 0.01                                   # 正则
    flow  = 1.0                                    # 低频
    fhigh = 100.0                                  # 高频
    df    = fx_decon(dn, dt, lf, mu, flow, fhigh)  # FX反褶积

    c = 0.95 * np.max(np.abs(d0))  # 色阶
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    ax[0].imshow(d0, aspect="auto", vmin=-c, vmax=c, cmap=seismic_colormap())  # 干净数据
    ax[0].set_title("Clean signal")
    ax[1].imshow(dn, aspect="auto", vmin=-c, vmax=c, cmap=seismic_colormap())  # 含噪数据
    ax[1].set_title("Noisy signal")
    ax[2].imshow(df, aspect="auto", vmin=-c, vmax=c, cmap=seismic_colormap())  # 去噪结果
    ax[2].set_title("Denoised signal")
    fig.tight_layout()

    Q1 = quality(dn, d0)  # 原始SNR
    Q2 = quality(df, d0)  # 处理后SNR
    print("Demo FX decon")
    print(f"SNR of input   {Q1:6.2f} db")
    print(f"SNR of output  {Q2:6.2f} db")
    plt.show()


if __name__ == "__main__":
    main()
