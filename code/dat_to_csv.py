#!/usr/bin/env python3
"""
将示波器的DAT文件转换为CSV格式
支持包含JSON头部和二进制波形数据的DAT文件
支持绘制波形图并保存
"""

import json
import struct
import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt冲突
import matplotlib.pyplot as plt
import numpy as np


def parse_dat_file(dat_file_path):
    """
    解析DAT文件，提取JSON头部和波形数据
    
    Args:
        dat_file_path: DAT文件路径
        
    Returns:
        tuple: (channel_params, display_params, waveform_data)
    """
    with open(dat_file_path, 'rb') as f:
        content = f.read()
    
    # 查找JSON结束位置（通常以}}结束）
    json_end = content.find(b'}}') + 2
    
    if json_end == 1:  # 没找到
        raise ValueError("无法找到JSON头部")
    
    # 解析JSON头部
    json_str = content[:json_end].decode('utf-8', errors='ignore')
    try:
        params = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"JSON内容: {json_str[:200]}")
        raise
    
    # 提取二进制数据
    binary_data = content[json_end:]
    
    return params, binary_data


def convert_binary_to_values(binary_data, channel_param):
    """
    将二进制数据转换为实际电压值
    
    Args:
        binary_data: 二进制波形数据
        channel_param: 通道参数字典
        
    Returns:
        list: 电压值列表
    """
    depth = int(channel_param['Depth'])
    y_zero = float(channel_param['yZero'])
    y_scale = float(channel_param['yScale'])
    probe = float(channel_param['Probe'].replace('X', ''))
    
    # 解析二进制数据（每个采样点2字节，小端格式）
    values = []
    for i in range(0, min(depth * 2, len(binary_data)), 2):
        if i + 1 < len(binary_data):
            # 读取16位无符号整数（小端）
            raw_value = struct.unpack('<H', binary_data[i:i+2])[0]
            # 转换为电压值
            voltage = (raw_value - y_zero) / y_scale * probe
            values.append(voltage)
    
    return values


def plot_waveform(time_data, channels_data, channel_params, output_path, dat_file_path):
    """
    绘制波形图并保存
    
    Args:
        time_data: 时间数据列表
        channels_data: 各通道电压数据列表
        channel_params: 通道参数列表
        output_path: 输出图片路径
        dat_file_path: 原始DAT文件路径
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    num_channels = len(channels_data)
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 4 * num_channels), squeeze=False)
    
    # 转换为秒或毫秒或微秒
    time_array = np.array(time_data)
    if time_array[-1] < 1e-3:
        time_array = time_array * 1e6
        time_unit = 'μs'
    elif time_array[-1] < 1:
        time_array = time_array * 1e3
        time_unit = 'ms'
    else:
        time_unit = 's'
    
    # 绘制每个通道
    colors = ['#FFD700', '#00CED1', '#FF69B4', '#32CD32']  # 黄、青、粉、绿
    
    for idx, (data, param) in enumerate(zip(channels_data, channel_params)):
        ax = axes[idx, 0]
        channel = param['Channel']
        unit = param.get('Unit', 'V')
        sample_rate = float(param['SampleRate'])
        
        # 绘制波形
        color = colors[idx % len(colors)]
        ax.plot(time_array, data, color=color, linewidth=1.5, label=f'CH{channel}')
        
        # 设置标签和标题
        ax.set_xlabel(f'Time ({time_unit})', fontsize=12)
        ax.set_ylabel(f'Voltage ({unit})', fontsize=12)
        ax.set_title(f'Channel {channel} - Sample Rate: {sample_rate/1e6:.1f} MS/s', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=10)
        
        # 添加统计信息
        voltage_array = np.array(data)
        v_max = np.max(voltage_array)
        v_min = np.min(voltage_array)
        v_pp = v_max - v_min
        v_mean = np.mean(voltage_array)
        v_rms = np.sqrt(np.mean(voltage_array**2))
        
        # 在图上显示统计信息
        stats_text = f'Max: {v_max:.3f}{unit}\nMin: {v_min:.3f}{unit}\n'
        stats_text += f'Vpp: {v_pp:.3f}{unit}\nMean: {v_mean:.3f}{unit}\nRMS: {v_rms:.3f}{unit}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9, family='monospace')
    
    # 设置总标题
    fig.suptitle(f'Waveform: {Path(dat_file_path).name}', fontsize=16, fontweight='bold')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def dat_to_csv(dat_file_path, csv_file_path=None, plot=False, plot_file_path=None):
    """
    将DAT文件转换为CSV文件
    
    Args:
        dat_file_path: 输入的DAT文件路径
        csv_file_path: 输出的CSV文件路径（可选，默认为同名.csv文件）
        plot: 是否绘制波形图（默认False）
        plot_file_path: 波形图保存路径（可选，默认为同名.png文件）
    """
    # 解析DAT文件
    print(f"正在读取文件: {dat_file_path}")
    params, binary_data = parse_dat_file(dat_file_path)
    
    # 获取通道参数
    channel_params = params.get('CHANNEL_PARAM', [])
    if not channel_params:
        raise ValueError("未找到通道参数")
    
    # 确定输出文件名
    if csv_file_path is None:
        csv_file_path = Path(dat_file_path).with_suffix('.csv')
    
    # 转换数据并写入CSV
    with open(csv_file_path, 'w', encoding='utf-8') as f:
        # 写入头部信息
        f.write("# 示波器波形数据\n")
        f.write(f"# 原始文件: {Path(dat_file_path).name}\n")
        
        # 处理每个通道
        all_channels_data = []
        all_time_data = []
        headers = ["Time(s)"]
        
        for idx, channel_param in enumerate(channel_params):
            channel = channel_param['Channel']
            sample_rate = float(channel_param['SampleRate'])
            unit = channel_param.get('Unit', 'V')
            
            f.write(f"# 通道 {channel}: 采样率={sample_rate} Hz, 单位={unit}\n")
            
            # 转换波形数据
            voltages = convert_binary_to_values(binary_data, channel_param)
            all_channels_data.append(voltages)
            headers.append(f"CH{channel}({unit})")
            
            print(f"  通道 {channel}: {len(voltages)} 个采样点")
        
        # 写入CSV表头
        f.write(','.join(headers) + '\n')
        
        # 写入数据
        max_length = max(len(data) for data in all_channels_data) if all_channels_data else 0
        sample_rate = float(channel_params[0]['SampleRate'])
        
        for i in range(max_length):
            time = i / sample_rate
            all_time_data.append(time)
            row = [f"{time:.9e}"]
            
            for data in all_channels_data:
                if i < len(data):
                    row.append(f"{data[i]:.6e}")
                else:
                    row.append("")
            
            f.write(','.join(row) + '\n')
    
    print(f"转换完成: {csv_file_path}")
    print(f"共 {max_length} 个采样点")
    
    # 绘制波形图
    if plot:
        if plot_file_path is None:
            plot_file_path = Path(dat_file_path).with_suffix('.png')
        
        plot_waveform(all_time_data, all_channels_data, channel_params, plot_file_path, dat_file_path)
        print(f"波形图已保存: {plot_file_path}")
    
    return all_time_data, all_channels_data


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='将示波器DAT文件转换为CSV格式，并可选择绘制波形图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dat_to_csv.py data.dat                    # 只转换为CSV
  python dat_to_csv.py data.dat -p                 # 转换并绘制波形图
  python dat_to_csv.py data.dat -o output.csv      # 指定输出CSV文件名
  python dat_to_csv.py data.dat -p -i wave.png     # 指定波形图文件名
        """
    )
    
    parser.add_argument('dat_file', help='输入的DAT文件路径')
    parser.add_argument('-o', '--output', help='输出的CSV文件路径（可选）')
    parser.add_argument('-p', '--plot', action='store_true', help='绘制波形图')
    parser.add_argument('-i', '--image', help='波形图输出路径（可选，默认为同名.png）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dat_file):
        print(f"错误: 文件不存在: {args.dat_file}")
        sys.exit(1)
    
    try:
        dat_to_csv(args.dat_file, args.output, args.plot, args.image)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
