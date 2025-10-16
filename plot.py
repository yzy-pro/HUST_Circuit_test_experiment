# Calculate power factor using formula cos\phi = P/(U*I)
# Generate two separate plots: cos\phi vs C and I vs C
# Use smooth curve fitting

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Remove Chinese font settings for compatibility
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

# Read data
data = pd.read_csv('data.csv')

# Extract data
C = data['C(uF)'].values
U = data['U(V)'].values
I = data['I(mA)'].values
P = data['P(W)'].values

# Calculate power factor cos φ = P/(U*I)
# Note: I is in mA, need to convert to A
cos_phi = P / (U * I / 1000)

# Create smooth curves
C_smooth = np.linspace(C.min(), C.max(), 300)

# Use spline interpolation for smooth curves
spl_cos_phi = make_interp_spline(C, cos_phi, k=3)
cos_phi_smooth = spl_cos_phi(C_smooth)

spl_I = make_interp_spline(C, I, k=3)
I_smooth = spl_I(C_smooth)

# Use spline interpolation for smooth curves
spl_cos_phi = make_interp_spline(C, cos_phi, k=3)
cos_phi_smooth = spl_cos_phi(C_smooth)

spl_I = make_interp_spline(C, I, k=3)
I_smooth = spl_I(C_smooth)

# Create first plot: cos φ vs C
plt.figure(figsize=(10, 6))
plt.plot(C_smooth, cos_phi_smooth, 'b-', linewidth=2, label='Fitted curve')
plt.scatter(C, cos_phi, color='red', s=50, zorder=5, label='Experimental data')
plt.xlabel('Capacitance C (μF)', fontsize=12)
plt.ylabel('Power factor cos φ', fontsize=12)
plt.title('Power factor cos φ vs Capacitance C', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('power_factor_curve.png', dpi=300, bbox_inches='tight')
print("Image saved as 'power_factor_curve.png'")
plt.show()

# Create second plot: I vs C
plt.figure(figsize=(10, 6))
plt.plot(C_smooth, I_smooth, 'g-', linewidth=2, label='Fitted curve')
plt.scatter(C, I, color='red', s=50, zorder=5, label='Experimental data')
plt.xlabel('Capacitance C (μF)', fontsize=12)
plt.ylabel('Current I (mA)', fontsize=12)
plt.title('Current I vs Capacitance C', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('current_curve.png', dpi=300, bbox_inches='tight')
print("Image saved as 'current_curve.png'")
plt.show()

# Print calculation results
print("\nCalculation results:")
print("C(μF)\tcos φ")
print("-" * 20)
for i in range(len(C)):
    print(f"{C[i]:.2f}\t{cos_phi[i]:.4f}")
