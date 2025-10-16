#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取并显示CH2通道的横纵坐标刻度(div)信息
"""

import json

def read_channel_info(filename='underdamping_wave.dat'):
    """
    读取dat文件中的通道参数信息
    """
    with open(filename, 'rb') as f:
        # 读取第一行JSON参数
        first_line = f.readline().decode('utf-8')
        params = json.loads(first_line)
    
    return params

def main():
    print("=" * 70)
    print("CH2通道的横纵坐标刻度(div)信息")
    print("=" * 70)
    
    # 读取参数
    params = read_channel_info('underdamping_wave.dat')
    
    # 获取CH2的参数
    channel_params = params['CHANNEL_PARAM']
    
    ch2_param = None
    for ch_param in channel_params:
        if ch_param['Channel'] == '2':
            ch2_param = ch_param
            break
    
    if ch2_param is None:
        print("错误: 未找到CH2通道信息")
        return
    
    print("\nCH2通道完整参数:")
    print("-" * 70)
    for key, value in ch2_param.items():
        print(f"  {key:15s}: {value}")
    
    print("\n" + "=" * 70)
    print("重点信息：")
    print("=" * 70)
    
    # 横坐标（时间轴）信息
    x_scale = float(ch2_param['xScale'])  # 单位：ns/div
    x_start = float(ch2_param['xStart'])
    x_end = float(ch2_param['xEnd'])
    sample_rate = float(ch2_param['SampleRate'])
    
    print(f"\n横坐标（时间轴）刻度:")
    print(f"  xScale:      {x_scale} ns/div")
    print(f"               = {x_scale/1e6} ms/div")
    print(f"               = {x_scale/1e9} s/div")
    print(f"  采样率:      {sample_rate} Hz")
    print(f"  采样间隔:    {1/sample_rate*1e6:.6f} μs")
    print(f"  采样间隔:    {1/sample_rate*1e9:.6f} ns")
    
    # 纵坐标（电压）信息
    y_scale = float(ch2_param['yScale'])
    y_zero = float(ch2_param['yZero'])
    probe = float(ch2_param['Probe'].rstrip('X'))
    unit = ch2_param['Unit']
    
    print(f"\n纵坐标（电压）刻度:")
    print(f"  yScale:      {y_scale}")
    print(f"  探头倍数:    {probe}X")
    print(f"  实际电压/div: {1/y_scale * probe} {unit}/div")
    print(f"  yZero:       {y_zero} (零点偏移)")
    print(f"  单位:        {unit}")
    
    # 显示参数信息
    display_params = params['DISPLAY_PARAM']
    print(f"\n显示参数:")
    print(f"  网格数:")
    print(f"    横向(xGrid): {display_params['xGrid']} 格")
    print(f"    纵向(yGrid): {display_params['yGrid']} 格")
    
    # 计算总的时间和电压范围
    depth = int(ch2_param['Depth'])
    total_time = depth / sample_rate
    
    print(f"\n数据范围:")
    print(f"  数据点数:    {depth}")
    print(f"  总时间:      {total_time*1e6:.3f} μs")
    print(f"               = {total_time*1e3:.6f} ms")
    print(f"               = {total_time:.9f} s")
    
    # 计算实际的div值
    x_grid = int(display_params['xGrid'])
    total_time_ns = total_time * 1e9
    actual_time_per_div = total_time_ns / x_grid
    
    print(f"\n根据实际数据计算的刻度:")
    print(f"  总时间范围:  {total_time_ns:.3f} ns")
    print(f"  横向格数:    {x_grid}")
    print(f"  实际时间/div: {actual_time_per_div:.3f} ns/div")
    print(f"               = {actual_time_per_div/1e6:.6f} ms/div")
    
    print("\n" + "=" * 70)
    print("总结：")
    print("=" * 70)
    print(f"\nCH2 横坐标刻度: {x_scale} ns/div = {x_scale/1e6} ms/div")
    print(f"CH2 纵坐标刻度: {1/y_scale * probe} V/div")
    print("=" * 70)

if __name__ == "__main__":
    main()
