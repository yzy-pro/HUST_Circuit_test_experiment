#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制RLC串联电路三种阻尼情况的相位图
用于实验十：二阶电路的响应
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import odeint

# 设置字体
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 11

# 创建图形 - 1行3列
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 电路参数
L = 0.1          # 电感 (H)
C = 100e-6       # 电容 (F) - 增大电容使轨迹更明显
omega_0 = 1 / np.sqrt(L * C)

# 初始条件 - 电容初始电压和电感初始电流
u_c0 = 10        # 初始电容电压 (V)
i_L0 = 0         # 初始电感电流 (A) - 从零开始

# 时间范围
t = np.linspace(0, 0.08, 3000)

# ========== (a) 过阻尼 ==========
R_over = 200  # 大电阻
alpha_over = R_over / (2 * L)

# 过阻尼: s1, s2 都是负实数
beta = np.sqrt(alpha_over**2 - omega_0**2)
s1 = -alpha_over - beta  # 更负的根
s2 = -alpha_over + beta  # 较小的负根

# 根据初始条件求系数
A2 = (i_L0 / C - u_c0 * s1) / (s2 - s1)
A1 = u_c0 - A2

u_c_over = A1 * np.exp(s1 * t) + A2 * np.exp(s2 * t)
i_L_over = C * (A1 * s1 * np.exp(s1 * t) + A2 * s2 * np.exp(s2 * t))
u_L_over = L * C * (A1 * s1**2 * np.exp(s1 * t) + A2 * s2**2 * np.exp(s2 * t))

axes[0].plot(u_c_over, u_L_over, 'b-', linewidth=2)
axes[0].plot(u_c_over[0], u_L_over[0], 'ro', markersize=8, label='Start')
axes[0].plot(0, 0, 'go', markersize=8, label='End')
axes[0].set_xlabel('$u_c$', fontsize=13, fontweight='bold')
axes[0].set_ylabel('$u_L$', fontsize=13, fontweight='bold')
axes[0].set_title('(a) Over-damped', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].axhline(y=0, color='k', linewidth=0.8)
axes[0].axvline(x=0, color='k', linewidth=0.8)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# ========== (b) 欠阻尼 ==========
R_under = 30  # 小电阻
alpha_under = R_under / (2 * L)
omega_d_under = np.sqrt(omega_0**2 - alpha_under**2)

# 欠阻尼解
u_c_under = np.exp(-alpha_under * t) * (u_c0 * np.cos(omega_d_under * t) + 
                                         (i_L0 / (C * omega_d_under) + u_c0 * alpha_under / omega_d_under) * np.sin(omega_d_under * t))

# 对u_c求导得到i_L
du_c_dt = np.exp(-alpha_under * t) * (
    -u_c0 * omega_d_under * np.sin(omega_d_under * t) +
    (i_L0 / (C * omega_d_under) + u_c0 * alpha_under / omega_d_under) * omega_d_under * np.cos(omega_d_under * t)
) - alpha_under * u_c_under

i_L_under = C * du_c_dt

# u_L = L * di_L/dt
di_L_dt = C * (
    -alpha_under * du_c_dt + np.exp(-alpha_under * t) * (
        -u_c0 * omega_d_under**2 * np.cos(omega_d_under * t) -
        (i_L0 / (C * omega_d_under) + u_c0 * alpha_under / omega_d_under) * omega_d_under**2 * np.sin(omega_d_under * t)
    ) - alpha_under * du_c_dt
)
u_L_under = L * di_L_dt / C

axes[1].plot(u_c_under, u_L_under, 'b-', linewidth=2)
axes[1].plot(u_c_under[0], u_L_under[0], 'ro', markersize=8, label='Start')
axes[1].plot(0, 0, 'go', markersize=8, label='End')
axes[1].set_xlabel('$u_c$', fontsize=13, fontweight='bold')
axes[1].set_ylabel('$u_L$', fontsize=13, fontweight='bold')
axes[1].set_title('(b) Under-damped', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].axhline(y=0, color='k', linewidth=0.8)
axes[1].axvline(x=0, color='k', linewidth=0.8)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# ========== (c) 无阻尼 ==========
R_none = 0  # 无电阻
alpha_none = 0
omega_d_none = omega_0

# 无阻尼解（简谐振荡）
u_c_none = u_c0 * np.cos(omega_d_none * t) + (i_L0 / (C * omega_d_none)) * np.sin(omega_d_none * t)
i_L_none = i_L0 * np.cos(omega_d_none * t) - u_c0 * C * omega_d_none * np.sin(omega_d_none * t)
u_L_none = -L * i_L0 * omega_d_none * np.sin(omega_d_none * t) - L * u_c0 * C * omega_d_none**2 * np.cos(omega_d_none * t)

axes[2].plot(u_c_none, u_L_none, 'b-', linewidth=2)
axes[2].plot(u_c_none[0], u_L_none[0], 'ro', markersize=8, label='Start')
axes[2].set_xlabel('$u_c$', fontsize=13, fontweight='bold')
axes[2].set_ylabel('$u_L$', fontsize=13, fontweight='bold')
axes[2].set_title('(c) Undamped', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, linestyle='--')
axes[2].axhline(y=0, color='k', linewidth=0.8)
axes[2].axvline(x=0, color='k', linewidth=0.8)
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

# 调整布局
plt.tight_layout()

# 保存图片
output_file = 'E10_preview/相位图.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"相位图已保存到: {output_file}")

# 显示图形
plt.show()
