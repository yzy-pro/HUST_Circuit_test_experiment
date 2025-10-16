import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import math

# 设置字体
plt.rcParams['axes.unicode_minus'] = False

def read_all_data():
    """读取CSV文件中的所有数据组"""
    # 直接手动解析CSV文件
    with open('data.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析表头
    header = lines[0].strip().split(',')
    
    # 解析所有数据行
    all_data = []
    for i in range(1, len(lines)):
        if lines[i].strip():  # 跳过空行
            data_line = lines[i].strip().rstrip(',').split(',')
            
            # 创建数据字典
            data_dict = {}
            for j, col in enumerate(header):
                if j < len(data_line):
                    try:
                        if data_line[j] and data_line[j] != '':
                            data_dict[col] = float(data_line[j])
                        else:
                            data_dict[col] = 0
                    except ValueError:
                        # 如果不能转换为数字，保持字符串
                        data_dict[col] = data_line[j]
                else:
                    data_dict[col] = 0
            
            all_data.append(pd.Series(data_dict))
    
    return all_data

def calculate_voltage_vectors(data):
    """计算电压矢量，正确构建电压矢量图，区分N点和n点"""
    
    # 首先根据线电压构建ABC三角形
    # 设A点在原点作为起始点
    A_point = complex(0, 0)
    
    # 根据线电压Uab确定B点位置
    Uab = complex(data['U_ab(V)'] * np.cos(np.radians(30)), 
                  data['U_ab(V)'] * np.sin(np.radians(30)))
    B_point = A_point + Uab
    
    # 根据线电压Uac确定C点位置（注意Uca = -Uac）
    Uca = complex(data['U_ca(V)'] * np.cos(np.radians(150)), 
                  data['U_ca(V)'] * np.sin(np.radians(150)))
    # 从A点出发，沿着-Uca方向到达C点
    C_point = A_point - Uca
    
    # 验证BC线段，计算Ubc
    Ubc_calculated = C_point - B_point
    
    # 计算三角形的重心作为中性点N的位置
    N_point = (A_point + B_point + C_point) / 3
    
    # 计算n点的位置，满足约束条件：
    # |n - A| = Uan, |n - B| = Ubn
    # 这是两个圆的交点问题
    Uan_magnitude = data['U_an(V)']
    Ubn_magnitude = data['U_bn(V)']
    
    # 使用几何方法求解两圆交点
    # 圆1: 中心A，半径Uan_magnitude
    # 圆2: 中心B，半径Ubn_magnitude
    
    # AB之间的距离
    d = abs(B_point - A_point)
    
    if d > 0 and d <= (Uan_magnitude + Ubn_magnitude) and abs(Uan_magnitude - Ubn_magnitude) <= d:
        # 两圆相交，计算交点
        # 使用余弦定理计算角度
        cos_angle = (Uan_magnitude**2 + d**2 - Ubn_magnitude**2) / (2 * Uan_magnitude * d)
        cos_angle = max(-1, min(1, cos_angle))  # 限制在[-1, 1]范围内
        
        # AB方向的单位向量
        AB_direction = (B_point - A_point) / d
        
        # 计算两个交点
        angle = np.arccos(cos_angle)
        # 第一个交点（逆时针旋转）
        rotated_direction1 = AB_direction * np.exp(1j * angle)
        n_point1 = A_point + Uan_magnitude * rotated_direction1
        
        # 第二个交点（顺时针旋转）
        rotated_direction2 = AB_direction * np.exp(-1j * angle)
        n_point2 = A_point + Uan_magnitude * rotated_direction2
        
        # 选择距离N点更近的交点
        dist1 = abs(n_point1 - N_point)
        dist2 = abs(n_point2 - N_point)
        
        if dist1 <= dist2:
            n_point = n_point1
            print(f"选择交点1，距离N点: {dist1:.2f}")
        else:
            n_point = n_point2
            print(f"选择交点2，距离N点: {dist2:.2f}")
            
    else:
        # 如果两圆不相交，使用近似位置
        print(f"警告：圆不相交，d={d:.2f}, Uan={Uan_magnitude}, Ubn={Ubn_magnitude}")
        n_point = (A_point + B_point) / 2  # 使用AB中点作为近似
    
    # 验证n点到C点的距离是否等于Ucn
    Ucn_magnitude = data['U_cn(V)']
    calculated_Ucn = abs(n_point - C_point)
    
    # 计算相电压（从各相点到n点的矢量）
    Uan = n_point - A_point
    Ubn = n_point - B_point  
    Ucn = n_point - C_point
    
    # 计算UNn矢量（从N点到n点）
    UNn = n_point - N_point
    
    # 使用测量的线电压值，但调整方向
    Uab = B_point - A_point  # A到B
    Ubc = C_point - B_point  # B到C
    Uca = A_point - C_point  # C到A
    
    return {
        'Uan': Uan, 'Ubn': Ubn, 'Ucn': Ucn,
        'Uab': Uab, 'Ubc': Ubc, 'Uca': Uca,
        'UNn': UNn,
        'N_point': N_point, 'A_point': A_point, 
        'B_point': B_point, 'C_point': C_point,
        'n_point': n_point,
        'Ubc_calculated': Ubc_calculated,
        'Ucn_verification': calculated_Ucn
    }

def calculate_current_vectors(data):
    """计算电流矢量，以Ia为参考"""
    # 以Ia为参考（0度）
    Ia = complex(data['I_a(mA)'], 0)
    
    # 根据三相对称性计算其他电流（相位差120度）
    Ib = complex(data['I_b(mA)'] * np.cos(np.radians(-120)), 
                 data['I_b(mA)'] * np.sin(np.radians(-120)))
    Ic = complex(data['I_c(mA)'] * np.cos(np.radians(120)), 
                 data['I_c(mA)'] * np.sin(np.radians(120)))
    
    # 计算三相电流的矢量和
    three_phase_sum = Ia + Ib + Ic
    
    # 零序电流：大小以测量值为准，方向是三相电流矢量和的反方向
    Io_magnitude = data['I_o(mA)']  # 测量值的大小
    
    if Io_magnitude > 0 and abs(three_phase_sum) > 0:
        # 如果有测量值且三相电流和不为零，使用测量值的大小，计算方向的方向
        Io_direction = -three_phase_sum / abs(three_phase_sum)  # 单位矢量，反方向
        Io = Io_magnitude * Io_direction
    elif Io_magnitude > 0:
        # 如果有测量值但三相电流和为零，假设Io方向为负实轴方向
        Io = complex(-Io_magnitude, 0)
    else:
        # 如果测量值为零，则Io为零
        Io = complex(0, 0)
    
    # 用于验证的计算值
    Io_calculated = -three_phase_sum
    Io_measured = complex(data['I_o(mA)'], 0) if data['I_o(mA)'] != 0 else complex(0, 0)
    
    return {
        'Ia': Ia, 'Ib': Ib, 'Ic': Ic, 'Io': Io,
        'Io_measured': Io_measured, 'Io_calculated': Io_calculated,
        'three_phase_sum': three_phase_sum
    }

def plot_voltage_and_current_vectors(voltage_data, current_data, data):
    """绘制电压和电流矢量图的组合图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # === 左侧子图：电压矢量图 ===
    # 提取点位置和矢量
    N_point = (voltage_data['N_point'].real, voltage_data['N_point'].imag)  # 重心
    n_point = (voltage_data['n_point'].real, voltage_data['n_point'].imag)  # 约束点
    A_point = (voltage_data['A_point'].real, voltage_data['A_point'].imag)
    B_point = (voltage_data['B_point'].real, voltage_data['B_point'].imag)
    C_point = (voltage_data['C_point'].real, voltage_data['C_point'].imag)
    
    # 相电压和线电压
    Uan = voltage_data['Uan']
    Ubn = voltage_data['Ubn']
    Ucn = voltage_data['Ucn']
    Uab = voltage_data['Uab']
    Ubc = voltage_data['Ubc']
    Uca = voltage_data['Uca']
    UNn = voltage_data['UNn']  # N到n的矢量
    
    # 颜色定义
    phase_colors = ['red', 'green', 'blue']  # 相电压：红绿蓝
    line_colors = ['orange', 'purple', 'brown']  # 线电压：橙紫棕
    
    # 绘制相电压（从各相点指向n点）
    phase_data = [
        ('Uan', Uan, A_point, n_point, 0),  # 从A点指向n点
        ('Ubn', Ubn, B_point, n_point, 1),  # 从B点指向n点
        ('Ucn', Ucn, C_point, n_point, 2)   # 从C点指向n点
    ]
    
    for label, voltage, start_point, end_point, color_idx in phase_data:
        # 绘制相电压矢量
        arrow = FancyArrowPatch(start_point, end_point,
                               connectionstyle="arc3", 
                               arrowstyle='->', 
                               mutation_scale=20, 
                               color=phase_colors[color_idx],
                               linewidth=3)
        ax1.add_patch(arrow)
        
        # 添加相电压标签（使用测量值）
        mid_x = (start_point[0] + end_point[0]) / 2
        mid_y = (start_point[1] + end_point[1]) / 2
        
        # 获取对应的测量值
        if label == 'Uan':
            measured_value = data['U_an(V)']
        elif label == 'Ubn':
            measured_value = data['U_bn(V)']
        elif label == 'Ucn':
            measured_value = data['U_cn(V)']
        
        # 根据大小决定是否显示角度
        if measured_value > 0:
            label_text = f'{label}\n{measured_value:.1f}V\n∠{np.degrees(np.angle(voltage)):.1f}°'
        else:
            label_text = f'{label}\n{measured_value:.1f}V'
        
        ax1.annotate(label_text, 
                   xy=(mid_x, mid_y), 
                   xytext=(15, 15), 
                   textcoords='offset points',
                   fontsize=9,
                   fontweight='bold',
                   color=phase_colors[color_idx],
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 绘制线电压（构成矢量三角形）
    line_data = [
        ('Uab', Uab, A_point, B_point, 0),
        ('Ubc', Ubc, B_point, C_point, 1), 
        ('Uca', Uca, C_point, A_point, 2)
    ]
    
    for label, voltage, start_point, end_point, color_idx in line_data:
        # 绘制线电压矢量箭头（改为实线）
        arrow = FancyArrowPatch(start_point, end_point,
                               connectionstyle="arc3", 
                               arrowstyle='->', 
                               mutation_scale=15, 
                               color=line_colors[color_idx],
                               linewidth=2)
        ax1.add_patch(arrow)
        
        # 计算线电压标签位置
        mid_x = (start_point[0] + end_point[0]) / 2
        mid_y = (start_point[1] + end_point[1]) / 2
        
        # 获取对应的测量值
        if label == 'Uab':
            measured_value = data['U_ab(V)']
        elif label == 'Ubc':
            measured_value = data['U_bc(V)']
        elif label == 'Uca':
            measured_value = data['U_ca(V)']
        
        # 根据大小决定是否显示角度
        if measured_value > 0:
            label_text = f'{label}\n{measured_value:.1f}V\n∠{np.degrees(np.angle(voltage)):.1f}°'
        else:
            label_text = f'{label}\n{measured_value:.1f}V'
        
        ax1.annotate(label_text, 
                   xy=(mid_x, mid_y), 
                   xytext=(10, -25), 
                   textcoords='offset points',
                   fontsize=8,
                   color=line_colors[color_idx],
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.7))
    
    # 绘制UNn矢量（从N点到n点，使用黑色）
    UNn_arrow = FancyArrowPatch(N_point, n_point,
                               connectionstyle="arc3", 
                               arrowstyle='->', 
                               mutation_scale=18, 
                               color='black',
                               linewidth=2.5)
    ax1.add_patch(UNn_arrow)
    
    # 添加UNn标签（使用测量值）
    mid_x = (N_point[0] + n_point[0]) / 2
    mid_y = (N_point[1] + n_point[1]) / 2
    UNn_measured = data['U_Nn(V)']
    
    # 根据大小决定是否显示角度
    if UNn_measured > 0:
        UNn_label_text = f'UNn\n{UNn_measured:.1f}V\n∠{np.degrees(np.angle(UNn)):.1f}°'
    else:
        UNn_label_text = f'UNn\n{UNn_measured:.1f}V'
    
    ax1.annotate(UNn_label_text, 
               xy=(mid_x, mid_y), 
               xytext=(-25, 25), 
               textcoords='offset points',
               fontsize=8,
               fontweight='bold',
               color='black',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.8))
    
    # 标记各个点
    ax1.plot(*N_point, 'ko', markersize=10)  # 重心N
    ax1.plot(*n_point, 'mo', markersize=10)  # 约束点n
    ax1.plot(*A_point, 'ro', markersize=8)
    ax1.plot(*B_point, 'go', markersize=8)
    ax1.plot(*C_point, 'bo', markersize=8)
    
    # 添加点的标签
    ax1.annotate('N', xy=N_point, xytext=(-20, -20), textcoords='offset points', 
               fontsize=12, fontweight='bold', color='black')
    ax1.annotate('n', xy=n_point, xytext=(-20, 20), textcoords='offset points', 
               fontsize=12, fontweight='bold', color='magenta')
    ax1.annotate('A', xy=A_point, xytext=(10, 10), textcoords='offset points', 
               fontsize=10, fontweight='bold', color='red')
    ax1.annotate('B', xy=B_point, xytext=(10, 10), textcoords='offset points', 
               fontsize=10, fontweight='bold', color='green')
    ax1.annotate('C', xy=C_point, xytext=(10, 10), textcoords='offset points', 
               fontsize=10, fontweight='bold', color='blue')
    
    # 设置电压图坐标轴
    all_points = [N_point, n_point, A_point, B_point, C_point]
    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    
    margin = 50  # 边距
    ax1.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
    ax1.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    ax1.set_aspect('equal')
    
    # 先设置网格的刻度位置
    x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
    y_min, y_max = min(y_coords) - margin, max(y_coords) + margin
    x_ticks = np.arange(int(x_min//50)*50, int(x_max//50+1)*50, 50)
    y_ticks = np.arange(int(y_min//50)*50, int(y_max//50+1)*50, 50)
    
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 隐藏刻度标签但保留网格
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    # 隐藏边框
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    # 使用类别名称作为标题（英文）
    category_name = data['类别']
    # 将中文转换为英文标题
    title_map = {
        '负载对称有中线': 'Symmetric Load with Neutral',
        '负载对称无中线': 'Symmetric Load without Neutral', 
        '负载不对称有中线': 'Asymmetric Load with Neutral',
        '负载不对称无中线': 'Asymmetric Load without Neutral',
        'A相开路有中线': 'Phase A Open with Neutral',
        'A相开路无中线': 'Phase A Open without Neutral',
        'C相短路无中线': 'Phase C Short without Neutral'
    }
    title = title_map.get(category_name, category_name)
    ax1.set_title(f'{title} - Voltage Vectors', fontsize=13, fontweight='bold')
    
    # === 右侧子图：电流矢量图 ===
    colors = ['red', 'green', 'blue', 'black']
    origin = (0, 0)
    
    # 绘制电流矢量
    current_list = [
        ('Ia', current_data['Ia'], 0),
        ('Ib', current_data['Ib'], 1), 
        ('Ic', current_data['Ic'], 2),
        ('Io', current_data['Io'], 3)
    ]
    
    for label, current, color_idx in current_list:
        if abs(current) > 0:  # 只绘制非零电流
            end_point = (current.real, current.imag)
            
            # 绘制电流矢量箭头
            arrow = FancyArrowPatch(origin, end_point,
                                   connectionstyle="arc3", 
                                   arrowstyle='->', 
                                   mutation_scale=20, 
                                   color=colors[color_idx],
                                   linewidth=3)
            ax2.add_patch(arrow)
            
            # 添加电流标签（使用测量值）
            mid_x = (origin[0] + end_point[0]) / 2
            mid_y = (origin[1] + end_point[1]) / 2
            
            # 获取对应的测量值
            if label == 'Ia':
                measured_value = data['I_a(mA)']
            elif label == 'Ib':
                measured_value = data['I_b(mA)']
            elif label == 'Ic':
                measured_value = data['I_c(mA)']
            elif label == 'Io':
                measured_value = data['I_o(mA)']
            
            # 根据大小决定是否显示角度
            if measured_value > 0:
                label_text = f'{label}\n{measured_value:.2f}mA\n∠{np.degrees(np.angle(current)):.1f}°'
            else:
                label_text = f'{label}\n{measured_value:.2f}mA'
            
            ax2.annotate(label_text, 
                       xy=(mid_x, mid_y), 
                       xytext=(15, 15), 
                       textcoords='offset points',
                       fontsize=9,
                       fontweight='bold',
                       color=colors[color_idx],
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # 标记电流矢量终点
            ax2.plot(*end_point, 'o', color=colors[color_idx], markersize=6)
    
    # 标记原点
    ax2.plot(*origin, 'ko', markersize=10)
    ax2.annotate('O', xy=origin, xytext=(-20, -20), textcoords='offset points', 
               fontsize=12, fontweight='bold')
    
    # 设置电流图坐标轴
    non_zero_currents = [c for c in current_data.values() if abs(c) > 0]
    if non_zero_currents:
        max_val = max(abs(c) for c in non_zero_currents) * 1.4
        ax2.set_xlim(-max_val, max_val)
        ax2.set_ylim(-max_val, max_val)
    else:
        max_val = 50
        ax2.set_xlim(-max_val, max_val)
        ax2.set_ylim(-max_val, max_val)
    
    ax2.set_aspect('equal')
    
    # 先设置网格的刻度位置  
    ticks = np.arange(-int(max_val//25)*25, int(max_val//25+1)*25, 25)
    
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks) 
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 隐藏刻度标签但保留网格
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    # 隐藏边框
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    ax2.set_title(f'{title} - Current Vectors', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """Main function"""
    # Read all data
    all_data = read_all_data()
    print(f"Successfully read {len(all_data)} groups of data")
    
    # Generate vector diagrams for each group of data
    for i, data in enumerate(all_data):
        print(f"\nProcessing group {i+1} data:")
        print(f"Load type: {data['类别']}")
        
        try:
            # Calculate vectors
            voltage_data = calculate_voltage_vectors(data)
            current_vectors = calculate_current_vectors(data)
            
            # Print verification information
            print(f"Ubc measured: {data['U_bc(V)']} V")
            print(f"Ubc calculated: {abs(voltage_data['Ubc_calculated']):.1f}V ∠{np.degrees(np.angle(voltage_data['Ubc_calculated'])):.1f}°")
            print(f"Ucn measured: {data['U_cn(V)']} V")
            print(f"Ucn verification: {voltage_data['Ucn_verification']:.1f}V")
            
            print(f"Phase voltage info:")
            print(f"Uan: {abs(voltage_data['Uan']):.1f}V ∠{np.degrees(np.angle(voltage_data['Uan'])):.1f}°")
            print(f"Ubn: {abs(voltage_data['Ubn']):.1f}V ∠{np.degrees(np.angle(voltage_data['Ubn'])):.1f}°")
            print(f"Ucn: {abs(voltage_data['Ucn']):.1f}V ∠{np.degrees(np.angle(voltage_data['Ucn'])):.1f}°")
            
            print(f"Point position info:")
            print(f"N point(centroid): ({voltage_data['N_point'].real:.1f}, {voltage_data['N_point'].imag:.1f})")
            print(f"n point(constraint): ({voltage_data['n_point'].real:.1f}, {voltage_data['n_point'].imag:.1f})")
            print(f"UNn voltage: {abs(voltage_data['UNn']):.1f}V ∠{np.degrees(np.angle(voltage_data['UNn'])):.1f}°")
            
            print(f"Neutral current info:")
            print(f"Io measured magnitude: {data['I_o(mA)']}mA")
            print(f"Io used value: {abs(current_vectors['Io']):.2f}mA ∠{np.degrees(np.angle(current_vectors['Io'])):.1f}°")
            print(f"Three-phase current sum: {abs(current_vectors['three_phase_sum']):.2f}mA ∠{np.degrees(np.angle(current_vectors['three_phase_sum'])):.1f}°")
            print(f"Theoretical Io: {abs(current_vectors['Io_calculated']):.2f}mA ∠{np.degrees(np.angle(current_vectors['Io_calculated'])):.1f}°")
            
            # Draw diagrams
            fig = plot_voltage_and_current_vectors(voltage_data, current_vectors, data)
            
            # Use category name as filename (remove special characters)
            category_name = data['类别']
            safe_filename = category_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            filename = f'{safe_filename}.png'
            
            # Save image
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Vector diagram saved as: {filename}")
            
            # Close figure to free memory
            plt.close(fig)
            
        except Exception as e:
            print(f"Error processing group {i+1} data: {e}")
            continue
    
    print(f"\nAll {len(all_data)} groups of vector diagrams have been generated!")
    print("Generated files:")
    for data in all_data:
        category_name = data['类别']
        safe_filename = category_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        print(f"- {safe_filename}.png")

if __name__ == "__main__":
    main()