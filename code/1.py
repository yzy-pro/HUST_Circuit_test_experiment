# Plot I vs f (Amplitude-Frequency Curve)
# Plot phi vs f (Phase-Frequency Curve)

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Manually input data (parsed from CSV file)
frequency = np.array([320, 640, 854, 980, 1362, 1640, 1840, 2180, 3280])  # Hz
u_rpp = np.array([9.87, 9.48, 9.05, 8.73, 7.71, 7.14, 6.71, 5.85, 4.31])  # V
I = np.array([2.72, 5.52, 7.75, 9.18, 10.96, 9.20, 7.74, 5.48, 2.75])  # mA
phi = np.array([-80.12, -67.69, -54.0, -41.96, 0, 25.88, 39.33, 53.68, 72.00])  # degrees

# Find maximum current value I_0
I_0 = 10.96  # mA (at frequency 1362 Hz)
f_0 = 1362  # Hz (resonance frequency)

# Create smooth curves that pass through all data points
# Use cubic interpolation for smooth curves
f_smooth = np.linspace(frequency.min(), frequency.max(), 500)

# Cubic interpolation that passes through all points
interp_I = interp1d(frequency, I, kind='cubic')
I_smooth = interp_I(f_smooth)

interp_phi = interp1d(frequency, phi, kind='cubic')
phi_smooth = interp_phi(f_smooth)

# Create amplitude-frequency curve
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(f_smooth, I_smooth, '-', linewidth=2, color='blue', alpha=0.7, label='Fitted curve')
ax1.plot(frequency, I, 'o', markersize=8, color='red', label='Data points', zorder=3)
ax1.axhline(y=I_0, color='green', linewidth=1, linestyle='--', alpha=0.5, label=f'I_0 = {I_0:.2f} mA')
ax1.axvline(x=f_0, color='purple', linewidth=1, linestyle='--', alpha=0.5, label=f'f_0 = {f_0} Hz')
ax1.set_xlabel('Frequency f (Hz)', fontsize=12)
ax1.set_ylabel('Current I (mA)', fontsize=12)
ax1.set_title('Amplitude-Frequency Curve', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xlim(0, 3500)
ax1.set_ylim(0, 12)
plt.tight_layout()
plt.savefig('amplitude_frequency.png', dpi=300, bbox_inches='tight')
print("Amplitude-frequency curve saved as amplitude_frequency.png")
plt.close()

# Create phase-frequency curve
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(f_smooth, phi_smooth, '-', linewidth=2, color='orange', alpha=0.7, label='Fitted curve')
ax2.plot(frequency, phi, 'o', markersize=8, color='red', label='Data points', zorder=3)
ax2.axhline(y=0, color='green', linewidth=1, linestyle='--', alpha=0.5, label='phi = 0 deg')
ax2.axvline(x=f_0, color='purple', linewidth=1, linestyle='--', alpha=0.5, label=f'f_0 = {f_0} Hz')
ax2.set_xlabel('Frequency f (Hz)', fontsize=12)
ax2.set_ylabel('Phase phi (deg)', fontsize=12)
ax2.set_title('Phase-Frequency Curve', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xlim(0, 3500)
ax2.set_ylim(-90, 80)
plt.tight_layout()
plt.savefig('phase_frequency.png', dpi=300, bbox_inches='tight')
print("Phase-frequency curve saved as phase_frequency.png")
plt.close()

# Calculate and analyze power factor
print("\n" + "="*60)
print("Power Factor Analysis")
print("="*60)
# Power factor = cos(phi), where phi is in degrees
phi_rad = np.deg2rad(phi)  # Convert to radians
power_factor = np.cos(phi_rad)

print("\nFrequency (Hz) | Phase (deg) | Power Factor (cos(phi))")
print("-" * 60)
for i in range(len(frequency)):
    print(f"{frequency[i]:8.0f}       | {phi[i]:8.2f}    | {power_factor[i]:8.4f}")

print("\n" + "="*60)
print("Key Observations:")
print("="*60)
print(f"1. At resonance (f_0 = {f_0} Hz):")
print(f"   - Phase angle: {phi[4]:.2f} degrees")
print(f"   - Power factor: {power_factor[4]:.4f} (unity)")
print(f"   - Current is maximum: {I[4]:.2f} mA")
print(f"   - Voltage and current are in phase\n")

# Find frequencies where power factor is specific values
pf_half = np.abs(power_factor - 0.707)  # cos(45°) ≈ 0.707
idx_pf = np.argmin(pf_half)
print(f"2. Power factor ≈ 0.707 (cos(45°)):")
print(f"   - Closest at f = {frequency[idx_pf]} Hz")
print(f"   - Phase: {phi[idx_pf]:.2f} degrees")
print(f"   - Power factor: {power_factor[idx_pf]:.4f}\n")

print(f"3. At low frequency (f = {frequency[0]} Hz):")
print(f"   - Phase: {phi[0]:.2f} degrees (capacitive)")
print(f"   - Power factor: {power_factor[0]:.4f}")
print(f"   - Circuit is capacitive (I leads V)\n")

print(f"4. At high frequency (f = {frequency[-1]} Hz):")
print(f"   - Phase: {phi[-1]:.2f} degrees (inductive)")
print(f"   - Power factor: {power_factor[-1]:.4f}")
print(f"   - Circuit is inductive (I lags V)\n")

# Create power factor curve
fig3, ax3 = plt.subplots(figsize=(10, 6))
interp_pf = interp1d(frequency, power_factor, kind='cubic')
pf_smooth = interp_pf(f_smooth)
ax3.plot(f_smooth, pf_smooth, '-', linewidth=2, color='green', alpha=0.7, label='Fitted curve')
ax3.plot(frequency, power_factor, 'o', markersize=8, color='red', label='Data points', zorder=3)
ax3.axhline(y=1.0, color='blue', linewidth=1, linestyle='--', alpha=0.5, label='PF = 1 (unity)')
ax3.axhline(y=0.707, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='PF = 0.707')
ax3.axvline(x=f_0, color='purple', linewidth=1, linestyle='--', alpha=0.5, label=f'f_0 = {f_0} Hz')
ax3.set_xlabel('Frequency f (Hz)', fontsize=12)
ax3.set_ylabel('Power Factor (cos(phi))', fontsize=12)
ax3.set_title('Power Factor vs Frequency', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_xlim(0, 3500)
ax3.set_ylim(-0.2, 1.1)
plt.tight_layout()
plt.savefig('power_factor.png', dpi=300, bbox_inches='tight')
print("Power factor curve saved as power_factor.png")
print("="*60)
plt.show()