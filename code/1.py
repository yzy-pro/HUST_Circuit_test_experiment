#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据dat文件中记载的波形数据，其中CH2为欠阻尼状态下的电容电压波形，
根据CH2的波形数据，计算出电容电压的峰值以及各个峰值出现的时间点（只计算第1，2个即可）
"""

import numpy as np
import json
import struct

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
            
            # 生成时间轴
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

def find_peaks(time, voltage, num_peaks=2):
    """
    找到电压波形的前几个峰值
    
    参数:
    time: 时间数组
    voltage: 电压数组
    num_peaks: 需要找到的峰值数量
    
    返回:
    peaks_info: 包含峰值时间和电压的列表
    """
    peaks_info = []
    
    # 寻找局部最大值
    for i in range(1, len(voltage) - 1):
        if voltage[i] > voltage[i-1] and voltage[i] > voltage[i+1]:
            # 找到一个峰值
            if voltage[i] > 0:  # 只考虑正峰值
                peaks_info.append({
                    'index': i,
                    'time': time[i],
                    'voltage': voltage[i]
                })
                
                if len(peaks_info) >= num_peaks:
                    break
    
    return peaks_info

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
            voltage = ch2_data['voltage']
            
            print(f"\nCH2通道信息:")
            print(f"采样率: {ch2_data['sample_rate']} Hz")
            print(f"数据点数: {ch2_data['depth']}")
            print(f"探头倍数: {ch2_data['probe']}X")
            
            # 寻找前两个峰值
            peaks = find_peaks(time, voltage, num_peaks=2)
            
            print(f"\n找到 {len(peaks)} 个峰值:")
            for i, peak in enumerate(peaks, 1):
                print(f"\n第{i}个峰值:")
                print(f"  时间: {peak['time']*1000:.6f} ms ({peak['time']:.9f} s)")
                print(f"  电压: {peak['voltage']:.6f} V")
            
            # 计算时间差
            if len(peaks) >= 2:
                time_diff = peaks[1]['time'] - peaks[0]['time']
                print(f"\n两个峰值之间的时间差: {time_diff*1000:.6f} ms ({time_diff:.9f} s)")
                
                # 计算阻尼振荡频率
                freq = 1.0 / time_diff
                print(f"阻尼振荡频率 f_d = {freq:.2f} Hz")
                print(f"阻尼振荡角频率 ω_d = {2*np.pi*freq:.2f} rad/s")
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