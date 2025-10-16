#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制RLC串联电路欠阻尼响应波形图
用于实验十：二阶电路的响应
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 设置图形参数
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# 电路参数设置
R = 30           # 电阻 (Ω) - 减小电阻以降低阻尼
L = 0.1          # 电感 (H)
C = 10e-6        # 电容 (F)

# 计算阻尼系数和谐振角频率
alpha = R / (2 * L)
omega_0 = 1 / np.sqrt(L * C)
omega_d = np.sqrt(omega_0**2 - alpha**2)  # 阻尼振荡角频率

print(f"Damping coefficient α = {alpha:.2f} rad/s")
print(f"Natural frequency ω₀ = {omega_0:.2f} rad/s")
print(f"Damped oscillation frequency ωd = {omega_d:.2f} rad/s")
print(f"Damping ratio ζ = {alpha/omega_0:.3f}")

# 时间范围
t = np.linspace(0, 0.02, 1000)

# 初始条件
A = 10  # 幅值 (V)
phi = 0  # 初相位 (从零开始的正弦波)

# 欠阻尼响应: u_c(t) = A * e^(-αt) * sin(ω_d*t)
u_c = A * np.exp(-alpha * t) * np.sin(omega_d * t + phi)

# 包络线
envelope_upper = A * np.exp(-alpha * t)
envelope_lower = -A * np.exp(-alpha * t)

# 找到前两个峰值点用于标注
peaks_indices = []
for i in range(1, len(u_c) - 1):
    if u_c[i] > u_c[i-1] and u_c[i] > u_c[i+1] and u_c[i] > 0:
        peaks_indices.append(i)
        if len(peaks_indices) == 2:
            break

# 创建图形
fig, ax = plt.subplots()

# 绘制波形
ax.plot(t * 1000, u_c, 'b-', linewidth=2, label='$u_c(t)$')
ax.plot(t * 1000, envelope_upper, 'r--', linewidth=1.5, alpha=0.7, label='Envelope')
ax.plot(t * 1000, envelope_lower, 'r--', linewidth=1.5, alpha=0.7)

# 标注峰值点
if len(peaks_indices) >= 2:
    t1 = t[peaks_indices[0]]
    t2 = t[peaks_indices[1]]
    u_cm1 = u_c[peaks_indices[0]]
    u_cm2 = u_c[peaks_indices[1]]
    
    # 绘制峰值点
    ax.plot(t1 * 1000, u_cm1, 'ro', markersize=8, zorder=5)
    ax.plot(t2 * 1000, u_cm2, 'ro', markersize=8, zorder=5)
    
    # 绘制垂直虚线到x轴
    ax.plot([t1 * 1000, t1 * 1000], [0, u_cm1], 'gray', linestyle='--', alpha=0.5)
    ax.plot([t2 * 1000, t2 * 1000], [0, u_cm2], 'gray', linestyle='--', alpha=0.5)
    
    # 绘制水平虚线到y轴
    ax.plot([0, t1 * 1000], [u_cm1, u_cm1], 'gray', linestyle='--', alpha=0.5)
    ax.plot([0, t2 * 1000], [u_cm2, u_cm2], 'gray', linestyle='--', alpha=0.5)
    
    # 标注第一个峰值
    ax.text(t1 * 1000, -1.5, f'$t_1$', ha='center', fontsize=11, color='blue', fontweight='bold')
    ax.text(-0, u_cm1, f'$u_{{cm1}}$', ha='right', va='center', fontsize=11, color='blue', fontweight='bold')
    
    # 标注第二个峰值
    ax.text(t2 * 1000, -1.5, f'$t_2$', ha='center', fontsize=11, color='blue', fontweight='bold')
    ax.text(-0, u_cm2, f'$u_{{cm2}}$', ha='right', va='center', fontsize=11, color='blue', fontweight='bold')
    
    # 标注时间差 t2-t1
    mid_t = (t1 + t2) / 2 * 1000
    ax.annotate('', xy=(t2 * 1000, -6), xytext=(t1 * 1000, -6),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(mid_t, -7, f'$t_2 - t_1$', ha='center', fontsize=11, 
            color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.7))

# 设置坐标轴
ax.set_xlabel('Time $t$ (ms)', fontsize=12)
ax.set_ylabel('Voltage $u_c$ (V)', fontsize=12)
ax.set_title('RLC Series Circuit Underdamped Response', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=10)
ax.axhline(y=0, color='k', linewidth=0.5)

# 删除坐标轴刻度
ax.set_xticks([])
ax.set_yticks([])

# 移除边框，保留坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 调整布局
plt.tight_layout()

# 保存图片
output_file = 'E10_preview/欠阻尼波形图.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nWaveform saved to: {output_file}")

# 显示图形
plt.show()
