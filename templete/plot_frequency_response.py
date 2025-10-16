import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 参数设置
Q1 = 10  # 品质因数1 (较大)
Q2 = 3   # 品质因数2 (较小)
omega_0 = 1  # 归一化谐振角频率

# 频率范围 (归一化频率 omega/omega_0)
omega_ratio = np.linspace(0.1, 3, 1000)

# 计算Q1的幅频特性和相频特性
I_ratio_Q1 = 1 / np.sqrt(1 + Q1**2 * (omega_ratio - 1/omega_ratio)**2)
phi_Q1 = np.arctan(Q1 * (omega_ratio - 1/omega_ratio))
phi_degrees_Q1 = np.degrees(phi_Q1)

# 计算Q2的幅频特性和相频特性
I_ratio_Q2 = 1 / np.sqrt(1 + Q2**2 * (omega_ratio - 1/omega_ratio)**2)
phi_Q2 = np.arctan(Q2 * (omega_ratio - 1/omega_ratio))
phi_degrees_Q2 = np.degrees(phi_Q2)

# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Frequency Response of Series RLC Circuit', fontsize=16, fontweight='bold')

# 绘制幅频特性
ax1.plot(omega_ratio, I_ratio_Q1, 'b-', linewidth=2, label=f'$Q_1 = {Q1}$')
ax1.plot(omega_ratio, I_ratio_Q2, 'r-', linewidth=2, label=f'$Q_2 = {Q2}$')
ax1.axhline(y=1/np.sqrt(2), color='gray', linestyle='--', linewidth=1, 
            label=f'$1/\\sqrt{{2}} \\approx {1/np.sqrt(2):.3f}$')
ax1.axvline(x=1, color='g', linestyle='--', linewidth=1, alpha=0.5, 
            label='$\\omega_0$')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Normalized Frequency $\\omega/\\omega_0$', fontsize=12)
ax1.set_ylabel('Normalized Current $I/I_0$', fontsize=12)
ax1.set_title('Magnitude Response', fontsize=14)
ax1.legend(fontsize=10, loc='upper right')
ax1.set_xlim([0, 3])
ax1.set_ylim([0, 1.1])

# 标注Q1的半功率点
omega_h_Q1 = np.sqrt(1 + 1/(4*Q1**2)) + 1/(2*Q1)
omega_l_Q1 = np.sqrt(1 + 1/(4*Q1**2)) - 1/(2*Q1)
ax1.plot([omega_l_Q1, omega_h_Q1], [1/np.sqrt(2), 1/np.sqrt(2)], 'bo', markersize=5)

# 标注Q2的半功率点
omega_h_Q2 = np.sqrt(1 + 1/(4*Q2**2)) + 1/(2*Q2)
omega_l_Q2 = np.sqrt(1 + 1/(4*Q2**2)) - 1/(2*Q2)
ax1.plot([omega_l_Q2, omega_h_Q2], [1/np.sqrt(2), 1/np.sqrt(2)], 'ro', markersize=5)

# 标注带宽
bandwidth_Q1 = omega_h_Q1 - omega_l_Q1
bandwidth_Q2 = omega_h_Q2 - omega_l_Q2
ax1.annotate(f'$\\Delta\\omega_1 = {bandwidth_Q1:.3f}$', 
             xy=((omega_h_Q1 + omega_l_Q1)/2, 1/np.sqrt(2)), 
             xytext=((omega_h_Q1 + omega_l_Q1)/2, 0.5),
             arrowprops=dict(arrowstyle='->', color='blue'),
             fontsize=9, color='blue')
ax1.annotate(f'$\\Delta\\omega_2 = {bandwidth_Q2:.3f}$', 
             xy=((omega_h_Q2 + omega_l_Q2)/2, 1/np.sqrt(2)), 
             xytext=((omega_h_Q2 + omega_l_Q2)/2, 0.3),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=9, color='red')

# 绘制相频特性
ax2.plot(omega_ratio, phi_degrees_Q1, 'b-', linewidth=2, label=f'$Q_1 = {Q1}$')
ax2.plot(omega_ratio, phi_degrees_Q2, 'r-', linewidth=2, label=f'$Q_2 = {Q2}$')
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, label='$\\phi = 0°$')
ax2.axhline(y=45, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax2.axhline(y=-45, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax2.axvline(x=1, color='g', linestyle='--', linewidth=1, alpha=0.5, 
            label='$\\omega_0$')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Normalized Frequency $\\omega/\\omega_0$', fontsize=12)
ax2.set_ylabel('Phase Shift $\\phi$ (degrees)', fontsize=12)
ax2.set_title('Phase Response', fontsize=14)
ax2.legend(fontsize=10, loc='upper right')
ax2.set_xlim([0, 3])
ax2.set_ylim([-90, 90])

# 标注关键点
ax2.plot(1, 0, 'go', markersize=8)
ax2.annotate('Resonance ($\\omega_0$, 0°)', 
             xy=(1, 0), 
             xytext=(1.5, 30),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10)

# 添加说明文本


plt.tight_layout()
plt.savefig('E11/frequency_response.png', dpi=300, bbox_inches='tight')
print("Image saved to: E11/frequency_response.png")
plt.show()
