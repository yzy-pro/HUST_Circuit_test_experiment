#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制CH2通道的欠阻尼波形
从t=0开始显示完整波形
对原始数据进行滤波处理以减少噪声
"""

import numpy as np
import json
import struct
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import savgol_filter, find_peaks
import csv

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def read_dat_file(filename):
    """
    读取示波器保存的dat文件
    
    参数:
    filename: dat文件路径
    
    返回:
    params: 通道参数字典
    channels_data: 各通道的数据字典
    """
    with open(filename, 'rb') as f:
        # 读取第一行JSON参数
        first_line = f.readline().decode('utf-8')
        params = json.loads(first_line)
        
        # 获取通道参数
        channel_params = params['CHANNEL_PARAM']
        
        # 读取剩余的二进制数据
        binary_data = f.read()
    
    # 解析各通道数据
    channels_data = {}
    
    for ch_param in channel_params:
        channel = ch_param['Channel']
        depth = int(ch_param['Depth'])
        sample_rate = float(ch_param['SampleRate'])
        y_zero = float(ch_param['yZero'])
        y_scale = float(ch_param['yScale'])
        x_scale = float(ch_param['xScale'])
        probe = float(ch_param['Probe'].rstrip('X'))
        
        # 解析二进制数据 (假设是16位有符号整数)
        offset = (int(channel) - 1) * depth * 2  # 每个通道depth个点，每个点2字节
        
        if offset + depth * 2 <= len(binary_data):
            # 提取当前通道的数据
            channel_binary = binary_data[offset:offset + depth * 2]
            
            # 解析为16位有符号整数
            raw_values = struct.unpack(f'<{depth}h', channel_binary)
            
            # 转换为实际电压值
            # 公式: voltage = (raw_value - y_zero) / y_scale * probe
            voltages = np.array([(val - y_zero) / y_scale * probe for val in raw_values])
            
            # 生成时间轴 (从0开始)
            time_step = 1.0 / sample_rate
            times = np.arange(depth) * time_step
            
            channels_data[channel] = {
                'time': times,
                'voltage': voltages,
                'sample_rate': sample_rate,
                'depth': depth,
                'probe': probe
            }
        else:
            print(f"警告: 通道{channel}的数据不足")
    
    return params, channels_data

def save_filtered_data_to_csv(time, voltage_raw, voltage_filtered, filename='CH2_filtered_data.csv'):
    """
    将滤波前后的数据保存到CSV文件
    
    参数:
    time: 时间数组
    voltage_raw: 原始电压数组
    voltage_filtered: 滤波后的电压数组
    filename: 输出的CSV文件名
    """
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        writer.writerow(['Time_s', 'Time_ms', 'Voltage_Raw_V', 'Voltage_Filtered_V'])
        
        # 写入数据
        for i in range(len(time)):
            writer.writerow([
                f'{time[i]:.9f}',           # 时间(秒)
                f'{time[i]*1000:.6f}',      # 时间(毫秒)
                f'{voltage_raw[i]:.6f}',    # 原始电压
                f'{voltage_filtered[i]:.6f}' # 滤波后电压
            ])
    
    print(f"  数据已保存到: {filename}")
    print(f"  总共保存了 {len(time)} 个数据点")

def find_peaks_improved(time, voltage, num_peaks=2, debug=True):
    """
    使用改进的算法找到电压波形的前几个峰值
    专门在0.08ms和0.23ms附近寻找峰值
    同时找出t1之后到0.1ms之前的所有极大值
    
    参数:
    time: 时间数组
    voltage: 电压数组（已滤波）
    num_peaks: 需要找到的峰值数量
    debug: 是否输出调试信息
    
    返回:
    peaks_info: 包含峰值时间和电压的列表
    all_peaks_between: t1和0.1ms之间的所有极大值
    """
    # 计算合适的distance参数
    sample_rate = 1.0 / (time[1] - time[0])
    min_distance = int(0.02e-3 * sample_rate)  # 0.02ms对应的样本点数，更小的间隔
    
    if debug:
        print(f"  采样率: {sample_rate} Hz")
        print(f"  时间步长: {(time[1] - time[0])*1000:.6f} ms")
        print(f"  最小峰间距: {min_distance} 个样本点")
    
    # 找到所有正峰值，使用非常宽松的参数
    peak_indices, properties = find_peaks(
        voltage,
        prominence=None,  # 不限制显著性
        distance=min_distance,
        height=None,  # 不限制高度
        width=1  # 峰的最小宽度
    )
    
    if debug:
        print(f"  找到 {len(peak_indices)} 个候选峰值")
    
    # 在指定的时间范围内寻找峰值
    # 范围1: 0.05-0.15ms (寻找0.08ms附近的峰)
    # 范围2: 0.15-0.35ms (寻找0.23ms附近的峰)
    
    range1_peaks = []  # 0.08ms附近
    range2_peaks = []  # 0.23ms附近
    
    for idx in peak_indices:
        t_ms = time[idx] * 1000
        if 0.05 <= t_ms <= 0.15:
            range1_peaks.append(idx)
        elif 0.15 <= t_ms <= 0.35:
            range2_peaks.append(idx)
    
    if debug:
        print(f"\n  在0.05-0.15ms范围内找到 {len(range1_peaks)} 个峰值 (期望0.08ms附近):")
        for idx in range1_peaks[:5]:
            print(f"    时间={time[idx]*1000:.6f}ms, 电压={voltage[idx]:.6f}V")
        
        print(f"\n  在0.15-0.35ms范围内找到 {len(range2_peaks)} 个峰值 (期望0.23ms附近):")
        for idx in range2_peaks[:5]:
            print(f"    时间={time[idx]*1000:.6f}ms, 电压={voltage[idx]:.6f}V")
    
    peaks_info = []
    all_peaks_between = []
    
    # 选择t1: 在0.08ms附近找最大峰值
    if len(range1_peaks) > 0:
        # 选择电压最大的作为t1
        range1_voltages = [voltage[idx] for idx in range1_peaks]
        max_idx = np.argmax(range1_voltages)
        t1_index = range1_peaks[max_idx]
        t1_time = time[t1_index]
        
        if debug:
            print(f"\n  ✓ 选择t1: 索引={t1_index}, 时间={t1_time*1000:.6f}ms, 电压={voltage[t1_index]:.6f}V")
        
        peaks_info.append({
            'index': t1_index,
            'time': t1_time,
            'voltage': voltage[t1_index]
        })
        
        # 找出t1之后到0.1ms之前的所有极大值
        for idx in peak_indices:
            t_ms = time[idx] * 1000
            if time[idx] > t1_time and t_ms < 0.1:
                all_peaks_between.append({
                    'index': idx,
                    'time': time[idx],
                    'voltage': voltage[idx]
                })
        
        if debug:
            print(f"\n  在t1({t1_time*1000:.6f}ms)之后到0.1ms之前找到 {len(all_peaks_between)} 个极大值:")
            for i, peak in enumerate(all_peaks_between[:10]):
                print(f"    #{i+1}: 时间={peak['time']*1000:.6f}ms, 电压={peak['voltage']:.6f}V")
    else:
        if debug:
            print("\n  ✗ 警告: 在0.05-0.15ms范围内未找到峰值!")
    
    # 选择t2: 在0.23ms附近找最大峰值
    if num_peaks >= 2 and len(range2_peaks) > 0:
        # 选择电压最大的作为t2
        range2_voltages = [voltage[idx] for idx in range2_peaks]
        max_idx = np.argmax(range2_voltages)
        t2_index = range2_peaks[max_idx]
        
        if debug:
            print(f"\n  ✓ 选择t2: 索引={t2_index}, 时间={time[t2_index]*1000:.6f}ms, 电压={voltage[t2_index]:.6f}V")
        
        peaks_info.append({
            'index': t2_index,
            'time': time[t2_index],
            'voltage': voltage[t2_index]
        })
    elif num_peaks >= 2:
        if debug:
            print("\n  ✗ 警告: 在0.15-0.35ms范围内未找到峰值!")
    
    return peaks_info, all_peaks_between

def plot_ch2_waveform(time, voltage_raw, voltage_filtered, peaks, all_peaks_between):
    """
    绘制CH2通道波形并标注峰值
    
    参数:
    time: 时间数组
    voltage_raw: 原始电压数组
    voltage_filtered: 滤波后的电压数组
    peaks: 主要峰值信息列表（t1和t2）
    all_peaks_between: t1到0.1ms之间的所有极大值
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 上图：原始波形
    ax1.plot(time * 1000, voltage_raw, 'gray', linewidth=1, alpha=0.5, label='原始数据')
    ax1.plot(time * 1000, voltage_filtered, 'b-', linewidth=1.5, label='滤波后数据')
    ax1.set_xlabel('时间 t (ms)', fontsize=12)
    ax1.set_ylabel('电压 (V)', fontsize=12)
    ax1.set_title('CH2 欠阻尼电容电压波形 - 原始数据 vs 滤波数据', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    
    # 下图：滤波后的波形并标注峰值
    ax2.plot(time * 1000, voltage_filtered, 'b-', linewidth=2, label='CH2 电容电压（滤波后）')
    
    # 标注主要峰值点（t1和t2）
    colors = ['red', 'orange']
    for i, peak in enumerate(peaks):
        peak_time_ms = peak['time'] * 1000
        peak_voltage = peak['voltage']
        color = colors[i] if i < len(colors) else 'red'
        
        # 绘制峰值点
        ax2.plot(peak_time_ms, peak_voltage, 'o', color=color, markersize=12, zorder=5, 
                markeredgecolor='black', markeredgewidth=2)
        
        # 绘制垂直虚线到x轴
        ax2.plot([peak_time_ms, peak_time_ms], [0, peak_voltage], 
                color=color, linestyle='--', linewidth=1.5, alpha=0.6)
        
        # 标注峰值
        ax2.annotate(f't{i+1}\n{peak_time_ms:.4f}ms\n{peak_voltage:.4f}V',
                   xy=(peak_time_ms, peak_voltage),
                   xytext=(15, 25), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', 
                                 color=color, lw=2),
                   fontsize=10, fontweight='bold')
    
    # 标注t1之后到0.1ms之前的所有极大值
    for i, peak in enumerate(all_peaks_between):
        peak_time_ms = peak['time'] * 1000
        peak_voltage = peak['voltage']
        
        # 绘制小峰值点
        ax2.plot(peak_time_ms, peak_voltage, 's', color='green', markersize=6, 
                zorder=4, alpha=0.7, markeredgecolor='darkgreen', markeredgewidth=1)
        
        # 绘制垂直细虚线
        ax2.plot([peak_time_ms, peak_time_ms], [0, peak_voltage], 
                color='green', linestyle=':', linewidth=1, alpha=0.4)
        
        # 标注小峰值（文字较小，避免重叠）
        ax2.text(peak_time_ms, peak_voltage, f'  {peak_time_ms:.3f}',
                fontsize=8, color='darkgreen', rotation=45, 
                verticalalignment='bottom')
    
    # 添加图例说明
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=2, label='CH2 电容电压'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, markeredgecolor='black', markeredgewidth=2, label='主峰值 (t1, t2)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
               markersize=8, markeredgecolor='darkgreen', label=f't1后的极大值 (共{len(all_peaks_between)}个)')
    ]
    
    # 如果有两个峰值，标注时间差
    if len(peaks) >= 2:
        t1 = peaks[0]['time'] * 1000
        t2 = peaks[1]['time'] * 1000
        time_diff = t2 - t1
        
        # 绘制时间差箭头
        y_pos = voltage_filtered.min() * 0.7
        ax2.annotate('', xy=(t2, y_pos), xytext=(t1, y_pos),
                   arrowprops=dict(arrowstyle='<->', color='purple', lw=2.5))
        ax2.text((t1 + t2) / 2, y_pos * 1.2, 
               f'Δt = {time_diff:.4f}ms\n= {time_diff*1e-3:.7f}s',
               ha='center', fontsize=11, color='purple', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.8))
        
        legend_elements.append(Line2D([0], [0], color='purple', linewidth=2.5, label='时间差 Δt'))
    
    # 设置坐标轴
    ax2.set_xlabel('时间 t (ms)', fontsize=12)
    ax2.set_ylabel('电压 (V)', fontsize=12)
    ax2.set_title('CH2 欠阻尼电容电压波形 - 峰值标注（包含t1后所有极大值）', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    
    # 添加0.1ms的垂直标记线
    ax2.axvline(x=0.1, color='gray', linestyle='-.', linewidth=1.5, alpha=0.5, label='0.1ms边界')
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    # 读取dat文件
    filename = 'underdamping_wave.dat'
    print(f"正在读取文件: {filename}")
    
    try:
        params, channels_data = read_dat_file(filename)
        print("文件读取成功！")
        
        # 获取CH2数据
        if '2' in channels_data:
            ch2_data = channels_data['2']
            time = ch2_data['time']
            voltage_raw = ch2_data['voltage']
            
            print(f"\nCH2通道信息:")
            print(f"采样率: {ch2_data['sample_rate']} Hz")
            print(f"数据点数: {ch2_data['depth']}")
            print(f"探头倍数: {ch2_data['probe']}X")
            print(f"时间范围: 0 ~ {time[-1]*1000:.3f} ms")
            print(f"电压范围: {voltage_raw.min():.4f} ~ {voltage_raw.max():.4f} V")
            
            # 对数据进行滤波处理
            print("\n正在对数据进行滤波处理...")
            # 使用Savitzky-Golay滤波器，保持波形形状的同时减少噪声
            # window_length必须是奇数，polyorder是多项式阶数
            window_length = 51  # 窗口长度，可以调整
            polyorder = 3       # 多项式阶数
            
            if len(voltage_raw) >= window_length:
                voltage_filtered = savgol_filter(voltage_raw, window_length, polyorder)
                print(f"滤波参数: 窗口长度={window_length}, 多项式阶数={polyorder}")
            else:
                voltage_filtered = voltage_raw
                print("警告: 数据点数太少，跳过滤波")
            
            # 寻找前两个峰值（使用滤波后的数据）
            print("\n正在寻找峰值...")
            print("策略: 在0.08ms和0.23ms附近分别寻找峰值")
            print("\n峰值检测调试信息:")
            peaks, all_peaks_between = find_peaks_improved(time, voltage_filtered, num_peaks=2, debug=True)
            
            print(f"\n找到 {len(peaks)} 个主峰值:")
            for i, peak in enumerate(peaks, 1):
                print(f"\n第{i}个峰值 (t{i}):")
                print(f"  时间: {peak['time']*1000:.6f} ms ({peak['time']:.9f} s)")
                print(f"  电压: {peak['voltage']:.6f} V")
                
                # 验证峰值位置
                if i == 1:
                    expected = 0.08
                    diff = abs(peak['time']*1000 - expected)
                    print(f"  理论位置: ~{expected} ms, 偏差: {diff:.4f} ms")
                elif i == 2:
                    expected = 0.23
                    diff = abs(peak['time']*1000 - expected)
                    print(f"  理论位置: ~{expected} ms, 偏差: {diff:.4f} ms")
            
            if len(all_peaks_between) > 0:
                print(f"\n在t1之后到0.1ms之前找到 {len(all_peaks_between)} 个额外的极大值")
            
            # 保存滤波后的数据到CSV
            print("\n正在保存滤波数据到CSV文件...")
            csv_filename = 'CH2_filtered_data.csv'
            save_filtered_data_to_csv(time, voltage_raw, voltage_filtered, csv_filename)
            
            # 计算时间差和频率
            if len(peaks) >= 2:
                time_diff = peaks[1]['time'] - peaks[0]['time']
                print(f"\n两个峰值之间的时间差:")
                print(f"  Δt = {time_diff*1000:.6f} ms ({time_diff:.9f} s)")
                
                freq = 1.0 / time_diff
                omega_d = 2 * np.pi * freq
                print(f"\n阻尼振荡参数:")
                print(f"  频率 f_d = {freq:.2f} Hz")
                print(f"  角频率 ω_d = {omega_d:.2f} rad/s")
                print(f"  周期 T_d = {1/freq*1000:.4f} ms")
            
            # 绘制波形
            print("\n正在绘制波形...")
            fig = plot_ch2_waveform(time, voltage_raw, voltage_filtered, peaks, all_peaks_between)
            
            # 保存图片
            output_file = 'CH2_underdamping_waveform_filtered.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n波形图已保存: {output_file}")
            
            # 显示图形
            plt.show()
            
        else:
            print("错误: 未找到CH2通道数据")
            
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 不存在")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
