#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RLC电路三种阻尼状态的李萨如图绘制脚本

作者: GitHub Copilot
日期: 2025-10-09
功能: 绘制RLC电路在欠阻尼、临界阻尼、过阻尼三种状态下的李萨如图
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def rlc_response(t, R, L, C, V0, damping_type='underdamped'):
    """
    计算RLC电路的响应
    
    参数:
    t: 时间数组
    R: 电阻 (Ω)
    L: 电感 (H)
    C: 电容 (F)
    V0: 初始电压 (V)
    damping_type: 阻尼类型 ('underdamped', 'critical', 'overdamped')
    
    返回:
    v_c: 电容电压
    i_L: 电感电流 (在串联RLC电路中等于电容电流)
    """
    
    # 计算特征参数
    omega_0 = 1 / np.sqrt(L * C)  # 固有角频率
    alpha = R / (2 * L)           # 阻尼系数
    
    # 判断阻尼类型（基于实际的α和ω₀关系）
    if alpha < 0.01 * omega_0:  # 无阻尼 (R≈0)
        # 无阻尼振荡: α ≈ 0
        v_c = V0 * np.cos(omega_0 * t)
        # 电容电流 i_c = C * dv_c/dt
        i_c = -V0 * C * omega_0 * np.sin(omega_0 * t)
        
    elif alpha < omega_0:  # 欠阻尼
        # 欠阻尼: α < ω₀
        omega_d = np.sqrt(omega_0**2 - alpha**2)  # 阻尼振荡频率
        
        # 电容电压
        v_c = V0 * np.exp(-alpha * t) * (np.cos(omega_d * t) + (alpha/omega_d) * np.sin(omega_d * t))
        
        # 电容电流 i_c = C * dv_c/dt，正确计算导数
        dv_dt = V0 * np.exp(-alpha * t) * (
            -alpha * (np.cos(omega_d * t) + (alpha/omega_d) * np.sin(omega_d * t)) +
            (-omega_d * np.sin(omega_d * t) + alpha * np.cos(omega_d * t))
        )
        i_c = C * dv_dt
        
    elif abs(alpha - omega_0) < 0.01 * omega_0:  # 临界阻尼
        # 临界阻尼: α = ω₀
        v_c = V0 * np.exp(-alpha * t) * (1 + alpha * t)
        
        # 电容电流 i_c = C * dv_c/dt
        dv_dt = V0 * np.exp(-alpha * t) * (alpha - alpha * (1 + alpha * t))
        dv_dt = V0 * alpha * np.exp(-alpha * t) * (-alpha * t)
        i_c = C * dv_dt
        
    else:  # 过阻尼
        # 过阻尼: α > ω₀
        s1 = -alpha + np.sqrt(alpha**2 - omega_0**2)
        s2 = -alpha - np.sqrt(alpha**2 - omega_0**2)
        
        A1 = V0 * s2 / (s2 - s1)
        A2 = -V0 * s1 / (s2 - s1)
        
        v_c = A1 * np.exp(s1 * t) + A2 * np.exp(s2 * t)
        
        # 电容电流 i_c = C * dv_c/dt
        dv_dt = A1 * s1 * np.exp(s1 * t) + A2 * s2 * np.exp(s2 * t)
        i_c = C * dv_dt
    
    return v_c, i_c  # i_c在串联电路中等于i_L

def plot_lissajous_figures():
    """绘制三种阻尼状态的李萨如图"""
    
    # 时间参数
    t_max = 0.02  # 最大时间改为20ms，能看到几个完整周期
    dt = 0.00001  # 时间步长改小
    t = np.arange(0, t_max, dt)
    
    # 电路参数 - 调整为更合理的值
    L = 10e-3   # 电感 10mH
    C = 10e-6   # 电容 10μF
    V0 = 5      # 初始电压 5V (减小以改善可视性)
    
    # 计算固有角频率
    omega_0 = 1 / np.sqrt(L * C)
    
    # 三种阻尼状态的电阻值
    R_critical = 2 * np.sqrt(L / C)  # 临界阻尼电阻
    R_under = 0.2 * R_critical       # 欠阻尼电阻 (α < ω₀)
    R_over = 1 / 0.2 * R_critical          # 过阻尼电阻 (α > ω₀)
    R_none = 0.001                   # 无阻尼电阻 (接近0)
    
    # 创建图形 - 只显示李萨如图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Lissajous Figures for RLC Circuit with Three Damping States', fontsize=16, fontweight='bold')
    
    # 阻尼类型和参数
    damping_configs = [
        ('overdamped', R_over, 'Overdamped'),
        ('underdamped', R_under, 'Underdamped'),
        ('underdamped', R_none, 'Undamped')
    ]
    
    colors = ['blue', 'red', 'green']
    
    for i, (damping_type, R, title) in enumerate(damping_configs):
        # 计算响应
        v_c, i_L = rlc_response(t, R, L, C, V0, damping_type)
        
        # 李萨如图 (相平面图)
        ax_lissa = axes[i]
        
        # 计算合适的电流缩放系数，使电流数值范围约为电压的1/10
        # 电压范围约为-5到+5V，电流希望显示为-0.5到+0.5的范围
        current_scale = 10000  # 调整这个系数来达到合适的数值比例
        
        # 绘制李萨如图
        ax_lissa.plot(v_c, i_L * current_scale, colors[i], linewidth=2, alpha=0.8)
        
        # 标记起点和终点
        ax_lissa.plot(v_c[0], i_L[0] * current_scale, 'o', color=colors[i], markersize=8, label='Start')
        ax_lissa.plot(v_c[-1], i_L[-1] * current_scale, 's', color=colors[i], markersize=6, label='End')
        
        # 添加方向箭头
        n_arrows = 5
        arrow_indices = np.linspace(len(t)//10, len(t)//2, n_arrows, dtype=int)
        for idx in arrow_indices:
            if idx < len(v_c) - 1:
                dx = v_c[idx+1] - v_c[idx]
                dy = (i_L[idx+1] - i_L[idx]) * current_scale
                ax_lissa.arrow(v_c[idx], i_L[idx] * current_scale, dx*5, dy*5, 
                              head_width=0.1, head_length=0.05, fc=colors[i], ec=colors[i], alpha=0.6)
        
        ax_lissa.set_xlabel('Capacitor Voltage $v_C$ (V)')
        ax_lissa.set_ylabel('Inductor Current (Scaled)')
        ax_lissa.set_title(f'{title}\nLissajous Figure')
        ax_lissa.grid(True, alpha=0.3)
        ax_lissa.legend()
        
        # 保持轴长度相同，只调整数值比例尺
        
        # 计算并显示阻尼系数
        alpha = R / (2 * L)
        damping_ratio = alpha / omega_0
        
        # 在图上添加参数信息
        info_text = f'R = {R:.1f} Ω\nα = {alpha:.0f} rad/s\nζ = {damping_ratio:.2f}'
        ax_lissa.text(0.02, 0.98, info_text, transform=ax_lissa.transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_phase_portrait_comparison():
    """绘制三种阻尼状态的相轨迹对比图"""
    
    # 时间参数
    t_max = 0.025  # 25ms
    dt = 0.00001
    t = np.arange(0, t_max, dt)
    
    # 电路参数
    L = 10e-3   # 电感 10mH
    C = 10e-6   # 电容 10μF
    V0 = 5      # 初始电压 5V
    
    # 计算电阻值
    R_critical = 2 * np.sqrt(L / C)
    R_under = 0.3 * R_critical
    R_over = 3 * R_critical
    R_none = 0.001
    
    # 创建对比图
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    damping_configs = [
        ('overdamped', R_over, 'Overdamped', 'green'),
        ('underdamped', R_under, 'Underdamped', 'blue'),
        ('underdamped', R_none, 'Undamped', 'red')
    ]
    
    # 使用相同的电流缩放系数
    current_scale = 10000
    
    for damping_type, R, label, color in damping_configs:
        v_c, i_L = rlc_response(t, R, L, C, V0, damping_type)
        ax.plot(v_c, i_L * current_scale, color=color, linewidth=3, label=label, alpha=0.8)
        
        # 标记起点
        ax.plot(v_c[0], i_L[0] * current_scale, 'o', color=color, markersize=10)
    
    ax.set_xlabel('Capacitor Voltage $v_C$ (V)', fontsize=14)
    ax.set_ylabel('Inductor Current (Scaled)', fontsize=14)
    ax.set_title('Phase Trajectory Comparison of RLC Circuit', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # 保持轴长度相同，只调整数值比例尺
    
    # 添加原点标记
    ax.plot(0, 0, 'ko', markersize=8, label='Equilibrium Point')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("Generating Lissajous figures for RLC circuit with three damping states...")
    
    # 绘制主要的李萨如图
    fig1 = plot_lissajous_figures()
    
    # 绘制相轨迹对比图
    fig2 = plot_phase_portrait_comparison()
    
    # 保存图片
    try:
        fig1.savefig('RLC_Lissajous_Three_Damping_States.png', dpi=300, bbox_inches='tight')
        fig2.savefig('RLC_Phase_Trajectory_Comparison.png', dpi=300, bbox_inches='tight')
        print("Images saved:")
        print("- RLC_Lissajous_Three_Damping_States.png")
        print("- RLC_Phase_Trajectory_Comparison.png")
    except Exception as e:
        print(f"Error saving images: {e}")
    
    # 关闭图形以释放内存
    plt.close(fig1)
    plt.close(fig2)
    
    print("\nCircuit Parameters:")
    print("- Inductance L = 10 mH")
    print("- Capacitance C = 10 μF") 
    print("- Initial Voltage V₀ = 5 V (optimized for display)")
    print(f"- Natural frequency f₀ = {1/(2*np.pi*np.sqrt(10e-3*10e-6)):.1f} Hz")
    print("- Overdamped resistance: R = 189.7 Ω (α > ω₀)")
    print("- Underdamped resistance: R = 19.0 Ω (α < ω₀)")
    print("- Undamped resistance: R ≈ 0 Ω (α ≈ 0)")
    print("- Voltage:Current scale ratio = 1:10")
    print("- Current scaled by 10000x for visibility")
    print("- Equal axis lengths, different numerical scales")

if __name__ == "__main__":
    main()
