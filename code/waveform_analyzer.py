import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def extract_waveform_by_color(image_path, target_color='cyan', color_threshold=50):
    """
    根据颜色提取波形
    
    Parameters:
    - image_path: 图像路径
    - target_color: 目标颜色 ('yellow', 'cyan')
    - color_threshold: 颜色阈值
    """
    # 读取图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 定义颜色范围（RGB格式）
    if target_color == 'cyan':
        # 青色：低R、高G、高B
        lower_bound = np.array([0, 150, 150])
        upper_bound = np.array([100, 255, 255])
    elif target_color == 'yellow':
        # 黄色：高R、高G、低B
        lower_bound = np.array([150, 150, 0])
        upper_bound = np.array([255, 255, 100])
    else:
        raise ValueError("Unsupported color")
    
    # 创建颜色掩码
    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
    # 提取波形坐标点
    points = np.column_stack(np.where(mask > 0))
    
    if len(points) == 0:
        print(f"未找到{target_color}色波形")
        return None, None
    
    # 按x坐标排序并去重
    # 注意：points的格式是(y, x)
    y_coords = points[:, 0]
    x_coords = points[:, 1]
    
    # 按x坐标分组，每个x取平均y值
    unique_x = np.unique(x_coords)
    averaged_y = []
    
    for x in unique_x:
        y_values = y_coords[x_coords == x]
        averaged_y.append(np.mean(y_values))
    
    return unique_x, np.array(averaged_y)


def analyze_peaks(x_coords, y_coords, image_shape, prominence=20):
    """
    分析波形峰值
    
    Parameters:
    - x_coords: x坐标数组
    - y_coords: y坐标数组（注意：图像坐标系y轴向下）
    - image_shape: 图像尺寸
    - prominence: 峰值显著性阈值
    """
    # 因为图像坐标系y轴向下，我们需要反转y坐标来找峰值
    inverted_y = -y_coords
    
    # 平滑处理
    smoothed_y = gaussian_filter1d(inverted_y, sigma=2)
    
    # 寻找峰值
    peaks, properties = find_peaks(smoothed_y, prominence=prominence, distance=50)
    
    # 转换坐标到实际物理单位
    # 根据用户提供的信息：
    # X轴：全宽约10ms，从 -5ms 到 +5ms
    # Y轴：全高约2000mV (即±1000mV)
    
    img_width = image_shape[1]
    img_height = image_shape[0]
    
    # 时间范围和电压范围
    time_range = (-5e-3, 5e-3)  # -5ms to +5ms (全宽10ms)
    voltage_range = (1000e-3, -1000e-3)  # 1000mV to -1000mV (注意y轴反转)
    
    # 坐标转换
    def pixel_to_physical(x_pix, y_pix):
        time = time_range[0] + (x_pix / img_width) * (time_range[1] - time_range[0])
        voltage = voltage_range[0] + (y_pix / img_height) * (voltage_range[1] - voltage_range[0])
        return time, voltage
    
    # 分析结果
    peak_info = []
    for i, peak_idx in enumerate(peaks):
        x_pix = x_coords[peak_idx]
        y_pix = y_coords[peak_idx]
        time, voltage = pixel_to_physical(x_pix, y_pix)
        
        peak_info.append({
            'peak_number': i + 1,
            'pixel_x': x_pix,
            'pixel_y': y_pix,
            'time_ms': time * 1000,  # 转换为ms
            'voltage_mv': voltage * 1000,  # 转换为mV
            'prominence': properties['prominences'][i]
        })
    
    return peak_info, peaks, smoothed_y


def analyze_valleys(x_coords, y_coords, image_shape, prominence=20):
    """
    分析波形谷值（波谷）
    
    Parameters:
    - x_coords: x坐标数组
    - y_coords: y坐标数组
    - image_shape: 图像尺寸
    - prominence: 谷值显著性阈值
    """
    # 对于谷值，直接使用y_coords（不反转）
    smoothed_y = gaussian_filter1d(y_coords, sigma=2)
    
    # 寻找谷值
    valleys, properties = find_peaks(smoothed_y, prominence=prominence, distance=50)
    
    img_width = image_shape[1]
    img_height = image_shape[0]
    
    time_range = (-5e-3, 5e-3)  # -5ms to +5ms
    voltage_range = (1000e-3, -1000e-3)
    
    def pixel_to_physical(x_pix, y_pix):
        time = time_range[0] + (x_pix / img_width) * (time_range[1] - time_range[0])
        voltage = voltage_range[0] + (y_pix / img_height) * (voltage_range[1] - voltage_range[0])
        return time, voltage
    
    valley_info = []
    for i, valley_idx in enumerate(valleys):
        x_pix = x_coords[valley_idx]
        y_pix = y_coords[valley_idx]
        time, voltage = pixel_to_physical(x_pix, y_pix)
        
        valley_info.append({
            'valley_number': i + 1,
            'pixel_x': x_pix,
            'pixel_y': y_pix,
            'time_ms': time * 1000,
            'voltage_mv': voltage * 1000,
            'prominence': properties['prominences'][i]
        })
    
    return valley_info, valleys


def visualize_results(x_coords, y_coords, peaks, valleys, smoothed_y, peak_info, valley_info, image_path):
    """
    可视化结果
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 显示原始图像和提取的波形
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.7, label='Extracted CH2 Waveform')
    axes[0].plot(x_coords[peaks], y_coords[peaks], 'yo', markersize=10, label='Detected Peaks')
    axes[0].plot(x_coords[valleys], y_coords[valleys], 'mo', markersize=10, label='Detected Valleys')
    axes[0].set_title('Original Image with Extracted CH2 Waveform (Cyan)')
    axes[0].legend()
    axes[0].axis('off')
    
    # 显示波形和峰值/谷值
    axes[1].plot(x_coords, -y_coords, 'cyan', linewidth=1.5, alpha=0.7, label='Original')
    axes[1].plot(x_coords, smoothed_y, 'b-', linewidth=2, label='Smoothed')
    axes[1].plot(x_coords[peaks], smoothed_y[peaks], 'ro', markersize=10, label='Peaks')
    axes[1].plot(x_coords[valleys], -y_coords[valleys], 'mo', markersize=10, label='Valleys')
    
    # 标注峰值
    for info in peak_info:
        axes[1].annotate(f"Peak {info['peak_number']}\n({info['time_ms']:.2f}ms, {info['voltage_mv']:.1f}mV)",
                        xy=(info['pixel_x'], -info['pixel_y']),
                        xytext=(10, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 标注谷值
    for info in valley_info:
        axes[1].annotate(f"Valley {info['valley_number']}\n({info['time_ms']:.2f}ms, {info['voltage_mv']:.1f}mV)",
                        xy=(info['pixel_x'], -info['pixel_y']),
                        xytext=(10, -30), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='magenta', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (inverted pixels)')
    axes[1].set_title('CH2 Waveform Analysis with Peak and Valley Detection')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ch2_waveform_analysis_result.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 图像路径（请修改为您的图像路径）
    image_path = 'image.png'
    
    # 提取CH2（青色）波形
    print("正在提取CH2通道（青色）波形...")
    x_coords, y_coords = extract_waveform_by_color(image_path, target_color='cyan')
    
    if x_coords is None:
        return
    
    print(f"提取到 {len(x_coords)} 个数据点")
    
    # 读取图像尺寸
    img = cv2.imread(image_path)
    print(f"图像尺寸: {img.shape[1]} x {img.shape[0]} 像素")
    print(f"时间刻度: -5ms 到 +5ms (全宽10ms)")
    print(f"电压刻度: -1000mV 到 +1000mV (全高2000mV)")
    
    # 分析峰值
    print("\n正在分析波形峰值...")
    peak_info, peaks, smoothed_y = analyze_peaks(x_coords, y_coords, img.shape, prominence=20)
    
    # 分析谷值
    print("正在分析波形谷值...")
    valley_info, valleys = analyze_valleys(x_coords, y_coords, img.shape, prominence=20)
    
    # 打印峰值信息
    print(f"\n检测到 {len(peak_info)} 个峰值：")
    print("-" * 80)
    print(f"{'峰值编号':<10} {'像素X':<12} {'像素Y':<12} {'时间(ms)':<15} {'电压(mV)':<15} {'显著性':<10}")
    print("-" * 80)
    
    for info in peak_info:
        print(f"{info['peak_number']:<10} {info['pixel_x']:<12.1f} {info['pixel_y']:<12.1f} "
              f"{info['time_ms']:<15.3f} {info['voltage_mv']:<15.2f} {info['prominence']:<10.2f}")
    
    # 打印谷值信息
    print(f"\n检测到 {len(valley_info)} 个谷值：")
    print("-" * 80)
    print(f"{'谷值编号':<10} {'像素X':<12} {'像素Y':<12} {'时间(ms)':<15} {'电压(mV)':<15} {'显著性':<10}")
    print("-" * 80)
    
    for info in valley_info:
        print(f"{info['valley_number']:<10} {info['pixel_x']:<12.1f} {info['pixel_y']:<12.1f} "
              f"{info['time_ms']:<15.3f} {info['voltage_mv']:<15.2f} {info['prominence']:<10.2f}")
    
    # 可视化结果
    print("\n正在生成可视化结果...")
    visualize_results(x_coords, y_coords, peaks, valleys, smoothed_y, peak_info, valley_info, image_path)
    
    # 保存数据到CSV
    import pandas as pd
    
    df_peaks = pd.DataFrame(peak_info)
    df_peaks.to_csv('ch2_peak_analysis.csv', index=False)
    print("\n峰值数据已保存到 ch2_peak_analysis.csv")
    
    df_valleys = pd.DataFrame(valley_info)
    df_valleys.to_csv('ch2_valley_analysis.csv', index=False)
    print("谷值数据已保存到 ch2_valley_analysis.csv")


if __name__ == "__main__":
    main()