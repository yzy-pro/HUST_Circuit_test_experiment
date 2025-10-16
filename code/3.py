#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析CH2_filtered_data.csv中的极大值
找出滤波后电压的所有极大值点
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import find_peaks

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def load_csv_data(filename='CH2_filtered_data.csv'):
    """
    从CSV文件加载数据
    
    参数:
    filename: CSV文件名
    
    返回:
    df: pandas DataFrame对象
    """
    try:
        df = pd.read_csv(filename)
        print(f"成功加载数据文件: {filename}")
        print(f"数据点数: {len(df)}")
        print(f"列名: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 不存在")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def find_all_peaks(time, voltage, min_distance_ms=0.01):
    """
    找出所有极大值点（使用非常宽松的条件）
    
    参数:
    time: 时间数组（秒）
    voltage: 电压数组
    min_distance_ms: 峰之间的最小距离（毫秒）
    
    返回:
    peaks_info: 包含所有极大值信息的列表
    """
    # 计算最小峰间距对应的样本点数
    sample_rate = 1.0 / (time[1] - time[0])
    min_distance = max(1, int(min_distance_ms * 1e-3 * sample_rate))  # 至少1个样本点
    
    print(f"\n极大值检测参数 (宽松模式):")
    print(f"  采样率: {sample_rate:.0f} Hz")
    print(f"  时间步长: {(time[1] - time[0])*1000:.6f} ms")
    print(f"  最小峰间距: {min_distance_ms} ms = {min_distance} 个样本点")
    
    # 找出所有极大值（使用非常宽松的条件）
    peak_indices, properties = find_peaks(
        voltage,
        prominence=None,    # 不限制显著性
        distance=min_distance,  # 最小峰间距
        height=None,        # 不限制高度
        width=None,         # 不限制宽度
        threshold=None,     # 不限制阈值
        plateau_size=None   # 不限制平台大小
    )
    
    print(f"\n找到 {len(peak_indices)} 个极大值点")
    
    # 整理峰值信息
    peaks_info = []
    for i, idx in enumerate(peak_indices):
        peaks_info.append({
            'index': idx,
            'number': i + 1,
            'time_s': time[idx],
            'time_ms': time[idx] * 1000,
            'voltage': voltage[idx]
        })
    
    return peaks_info, peak_indices

def analyze_peaks(peaks_info):
    """
    分析极大值的统计信息
    
    参数:
    peaks_info: 极大值信息列表
    """
    if len(peaks_info) == 0:
        print("没有找到极大值")
        return
    
    voltages = [p['voltage'] for p in peaks_info]
    times_ms = [p['time_ms'] for p in peaks_info]
    
    print(f"\n极大值统计分析:")
    print(f"  总数: {len(peaks_info)}")
    print(f"  电压范围: {min(voltages):.6f} ~ {max(voltages):.6f} V")
    print(f"  时间范围: {min(times_ms):.6f} ~ {max(times_ms):.6f} ms")
    print(f"  平均电压: {np.mean(voltages):.6f} V")
    print(f"  电压标准差: {np.std(voltages):.6f} V")
    
    # 找出电压最大的前10个峰值
    sorted_peaks = sorted(peaks_info, key=lambda x: x['voltage'], reverse=True)
    
    print(f"\n电压最大的前10个极大值:")
    print(f"{'序号':<6} {'时间(ms)':<12} {'电压(V)':<12} {'索引':<8}")
    print("-" * 50)
    for i, peak in enumerate(sorted_peaks[:10], 1):
        print(f"{i:<6} {peak['time_ms']:<12.6f} {peak['voltage']:<12.6f} {peak['index']:<8}")
    
    # 计算相邻峰值之间的时间间隔
    if len(peaks_info) >= 2:
        intervals = []
        for i in range(len(peaks_info) - 1):
            dt = peaks_info[i+1]['time_ms'] - peaks_info[i]['time_ms']
            intervals.append(dt)
        
        print(f"\n相邻极大值的时间间隔:")
        print(f"  平均间隔: {np.mean(intervals):.6f} ms")
        print(f"  最小间隔: {min(intervals):.6f} ms")
        print(f"  最大间隔: {max(intervals):.6f} ms")
        print(f"  标准差: {np.std(intervals):.6f} ms")

def print_all_peaks(peaks_info, max_display=100):
    """
    打印所有极大值的详细信息
    
    参数:
    peaks_info: 极大值信息列表
    max_display: 最多显示的数量
    """
    print(f"\n所有极大值详细列表 (显示前{min(len(peaks_info), max_display)}个):")
    print(f"{'编号':<6} {'时间(s)':<14} {'时间(ms)':<12} {'电压(V)':<12} {'索引':<8}")
    print("-" * 60)
    
    for peak in peaks_info[:max_display]:
        print(f"{peak['number']:<6} {peak['time_s']:<14.9f} {peak['time_ms']:<12.6f} "
              f"{peak['voltage']:<12.6f} {peak['index']:<8}")
    
    if len(peaks_info) > max_display:
        print(f"\n... 还有 {len(peaks_info) - max_display} 个极大值未显示")

def plot_peaks(time_ms, voltage, peak_indices, peaks_info):
    """
    绘制电压波形和所有极大值点
    
    参数:
    time_ms: 时间数组（毫秒）
    voltage: 电压数组
    peak_indices: 极大值的索引数组
    peaks_info: 极大值信息列表
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # 上图：完整波形
    ax1.plot(time_ms, voltage, 'b-', linewidth=1.5, label='滤波后电压')
    ax1.plot(time_ms[peak_indices], voltage[peak_indices], 'ro', 
             markersize=5, label=f'极大值 (共{len(peak_indices)}个)', zorder=5)
    
    # 标注前5个最大的峰值
    sorted_peaks = sorted(peaks_info, key=lambda x: x['voltage'], reverse=True)
    for i, peak in enumerate(sorted_peaks[:5], 1):
        ax1.annotate(f'#{i}\n{peak["time_ms"]:.3f}ms',
                    xy=(peak['time_ms'], peak['voltage']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    fontsize=8)
    
    ax1.set_xlabel('时间 (ms)', fontsize=12)
    ax1.set_ylabel('电压 (V)', fontsize=12)
    ax1.set_title('CH2 滤波后电压波形 - 所有极大值', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    
    # 下图：放大显示前0.5ms的波形
    mask = time_ms <= 0.5
    time_zoom = time_ms[mask]
    voltage_zoom = voltage[mask]
    peaks_in_range = [i for i in peak_indices if time_ms[i] <= 0.5]
    
    ax2.plot(time_zoom, voltage_zoom, 'b-', linewidth=2, label='滤波后电压')
    ax2.plot(time_ms[peaks_in_range], voltage[peaks_in_range], 'ro', 
             markersize=8, label=f'极大值 (共{len(peaks_in_range)}个)', zorder=5)
    
    # 标注每个峰值
    for idx in peaks_in_range[:15]:  # 只标注前15个避免过于拥挤
        peak_num = next((p['number'] for p in peaks_info if p['index'] == idx), None)
        if peak_num:
            ax2.text(time_ms[idx], voltage[idx], f' {peak_num}', 
                    fontsize=9, color='red', fontweight='bold',
                    verticalalignment='bottom')
    
    ax2.set_xlabel('时间 (ms)', fontsize=12)
    ax2.set_ylabel('电压 (V)', fontsize=12)
    ax2.set_title('CH2 滤波后电压波形 - 前0.5ms放大图', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def save_peaks_to_csv(peaks_info, filename='CH2_peaks_analysis.csv'):
    """
    将极大值信息保存到CSV文件
    
    参数:
    peaks_info: 极大值信息列表
    filename: 输出文件名
    """
    df = pd.DataFrame(peaks_info)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"\n极大值数据已保存到: {filename}")

def main():
    """主函数"""
    print("=" * 60)
    print("CH2 滤波数据极大值分析")
    print("=" * 60)
    
    # 加载CSV数据
    df = load_csv_data('CH2_filtered_data.csv')
    
    if df is None:
        return
    
    # 提取数据
    time = df['Time_s'].values
    time_ms = df['Time_ms'].values
    voltage_raw = df['Voltage_Raw_V'].values
    voltage_filtered = df['Voltage_Filtered_V'].values
    
    print(f"\n数据信息:")
    print(f"  时间范围: {time[0]:.6f} ~ {time[-1]:.6f} s")
    print(f"  时间范围: {time_ms[0]:.6f} ~ {time_ms[-1]:.6f} ms")
    print(f"  原始电压范围: {voltage_raw.min():.6f} ~ {voltage_raw.max():.6f} V")
    print(f"  滤波电压范围: {voltage_filtered.min():.6f} ~ {voltage_filtered.max():.6f} V")
    
    # 找出所有极大值
    peaks_info, peak_indices = find_all_peaks(time, voltage_filtered, min_distance_ms=0.01)
    
    # 分析极大值
    analyze_peaks(peaks_info)
    
    # 打印所有极大值
    print_all_peaks(peaks_info, max_display=100)
    
    # 输出第7和第8个极大值点的详细信息
    print("\n" + "=" * 60)
    print("重点关注：第7和第8个极大值点")
    print("=" * 60)
    
    if len(peaks_info) >= 7:
        peak7 = peaks_info[6]  # 索引从0开始，第7个是索引6
        print(f"\n第7个极大值点:")
        print(f"  横坐标(时间):")
        print(f"    - 秒(s):     {peak7['time_s']:.9f}")
        print(f"    - 毫秒(ms):  {peak7['time_ms']:.6f}")
        print(f"  纵坐标(电压): {peak7['voltage']:.6f} V")
        print(f"  数据索引:     {peak7['index']}")
    else:
        print(f"\n警告: 只找到 {len(peaks_info)} 个极大值，无法显示第7个点")
    
    if len(peaks_info) >= 8:
        peak8 = peaks_info[7]  # 索引从0开始，第8个是索引7
        print(f"\n第8个极大值点:")
        print(f"  横坐标(时间):")
        print(f"    - 秒(s):     {peak8['time_s']:.9f}")
        print(f"    - 毫秒(ms):  {peak8['time_ms']:.6f}")
        print(f"  纵坐标(电压): {peak8['voltage']:.6f} V")
        print(f"  数据索引:     {peak8['index']}")
        
        # 计算第7和第8点之间的差值
        if len(peaks_info) >= 7:
            time_diff = peak8['time_ms'] - peak7['time_ms']
            voltage_diff = peak8['voltage'] - peak7['voltage']
            print(f"\n第7点和第8点之间的差值:")
            print(f"  时间差 Δt:   {time_diff:.6f} ms = {time_diff*1e-3:.9f} s")
            print(f"  电压差 ΔV:   {voltage_diff:.6f} V")
            print(f"  电压比 V8/V7: {peak8['voltage']/peak7['voltage']:.6f}")
    else:
        print(f"\n警告: 只找到 {len(peaks_info)} 个极大值，无法显示第8个点")
    
    print("=" * 60)
    
    # 保存极大值到CSV
    save_peaks_to_csv(peaks_info, 'CH2_peaks_analysis.csv')
    
    # 绘制图形
    print("\n正在绘制图形...")
    fig = plot_peaks(time_ms, voltage_filtered, peak_indices, peaks_info)
    
    # 保存图片
    output_file = 'CH2_peaks_analysis.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图形已保存到: {output_file}")
    
    # 显示图形
    plt.show()
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()
