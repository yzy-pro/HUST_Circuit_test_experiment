# 先计算I，以I为纵坐标，fc为横坐标，标题为"幅频曲线"
# 再计算phi，以phi为纵坐标，fc为横坐标，标题为"相频曲线"
# 分别作图

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Read data
df = pd.read_csv('1.csv')

# Extract frequency values
fc = df['f_c'].values

# Calculate I values from I/I_0 ratios
I_0 = 6.43  # mA
I_ratio_str = df['I/I_0(I_0 = 6.43mA)'].values
I_ratio_values = []
for ratio in I_ratio_str:
    if '/' in str(ratio):
        if 'sqrt(2)' in str(ratio):
            I_ratio_values.append(1/np.sqrt(2))
        else:
            parts = str(ratio).split('/')
            I_ratio_values.append(eval(parts[0]) / eval(parts[1]))
    else:
        I_ratio_values.append(float(ratio))

I_ratio_values = np.array(I_ratio_values)
# Calculate actual I values in mA
I_values = I_ratio_values * I_0

# Extract phi values
phi = df['\\phi'].values

# Create smooth curves using spline interpolation in log space
from scipy.interpolate import UnivariateSpline
log_fc = np.log10(fc)
log_fc_smooth = np.linspace(log_fc.min(), log_fc.max(), 300)
fc_smooth = 10**log_fc_smooth

# Use spline with smoothing parameter in log space
spline_I = UnivariateSpline(log_fc, I_values, s=0.1, k=3)
I_smooth = spline_I(log_fc_smooth)
spline_phi = UnivariateSpline(log_fc, phi, s=10, k=3)
phi_smooth = spline_phi(log_fc_smooth)

# Plot amplitude-frequency curve
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(fc, I_values, 'o', markersize=6, label='Data Points')
plt.plot(fc_smooth, I_smooth, '-', linewidth=2, label='Smooth Curve')
plt.xlabel('Frequency fc (Hz)')
plt.ylabel('I (mA)')
plt.title('Amplitude-Frequency Curve')
plt.grid(True)
plt.legend()

# Plot phase-frequency curve
plt.subplot(1, 2, 2)
plt.plot(fc, phi, 'o', markersize=6, color='orange', label='Data Points')
plt.plot(fc_smooth, phi_smooth, '-', linewidth=2, color='orange', label='Smooth Curve')
plt.xlabel('Frequency fc (Hz)')
plt.ylabel('Phase φ (degrees)')
plt.title('Phase-Frequency Curve')
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
plt.legend()

plt.tight_layout()
plt.savefig('frequency_response.png', dpi=300)
plt.show()