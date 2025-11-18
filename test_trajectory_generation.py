#!/usr/bin/env python
"""Test trajectory generation to verify gradient values."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, 'modules/libspiral/src')
from tensorflow_mri_trajectory import gen_spiral_traj_tfmri
import numpy as np

print("Testing TensorFlow-MRI trajectory generation")
print("=" * 60)

# Generate trajectory with exact parameters from config
traj = gen_spiral_traj_tfmri(
    base_resolution=240,
    field_of_view=400,  # mm
    vd_spiral_arms=16,
    vd_inner_cutoff=0.15,
    pre_vd_outer_cutoff=0.41288,
    vd_outer_density=0.07,
    vd_type='hanning',
    max_grad_ampl=22.0,  # mT/m
    min_rise_time=10.0,  # us/mT/m
    dwell_time=1.4,  # us
)

print(f"\nTrajectory generated:")
print(f"  n_arms: {traj['n_arms']}")
print(f"  n_samples: {traj['n_samples']}")
print(f"  readout_time: {traj['readout_time']:.3f} ms")
print(f"  dwell_time: {traj['dwell_time']*1e6:.1f} us")

# Check gradients directly
gx = traj['gx']  # [n_arms, n_samples] in mT/m
gy = traj['gy']  # [n_arms, n_samples] in mT/m

print(f"\nGradient analysis:")
print(f"  gx shape: {gx.shape}")
print(f"  gy shape: {gy.shape}")

# Check each arm
for arm in range(min(3, traj['n_arms'])):
    g_mag = np.sqrt(gx[arm, :]**2 + gy[arm, :]**2)
    print(f"  Arm {arm}: max gradient = {np.max(g_mag):.1f} mT/m")

# Overall max
g_mag_all = np.sqrt(gx**2 + gy**2)
print(f"  Overall max gradient: {np.max(g_mag_all):.1f} mT/m")
print(f"  Overall mean gradient: {np.mean(g_mag_all):.1f} mT/m")

# Check k-space
kx = traj['kx']  # [n_arms, n_samples] in rad/m
ky = traj['ky']  # [n_arms, n_samples] in rad/m

k_mag = np.sqrt(kx**2 + ky**2)
k_max_rad = np.max(k_mag)
k_max_cycles = k_max_rad / (2 * np.pi)

print(f"\nK-space analysis:")
print(f"  Max |k|: {k_max_cycles:.1f} 1/m")

# Expected
fov_m = 0.4  # 400 mm
res_m = fov_m / 240
k_nyquist = 0.5 / res_m

print(f"  Expected Nyquist: {k_nyquist:.1f} 1/m")
print(f"  Ratio: {k_max_cycles / k_nyquist:.3f}")

# Verify gradient-k-space consistency
gamma = 2 * np.pi * 42.577478518e6  # rad/s/T
dt = traj['dwell_time']  # s

# Integrate first arm's gradients
kx_integrated = np.zeros(traj['n_samples'])
ky_integrated = np.zeros(traj['n_samples'])

for i in range(1, traj['n_samples']):
    kx_integrated[i] = kx_integrated[i-1] + gx[0, i] * gamma * 1e-3 * dt
    ky_integrated[i] = ky_integrated[i-1] + gy[0, i] * gamma * 1e-3 * dt

print(f"\nGradient-k-space consistency check (arm 0):")
print(f"  kx from trajectory: {np.max(np.abs(kx[0, :])):.1f} rad/m")
print(f"  kx from gradient integration: {np.max(np.abs(kx_integrated)):.1f} rad/m")
print(f"  Difference: {np.max(np.abs(kx[0, :] - kx_integrated)):.1f} rad/m")

if np.max(np.abs(kx[0, :] - kx_integrated)) < 10:
    print("  ✓ K-space and gradients are consistent")
else:
    print("  ✗ K-space and gradients are inconsistent")

print(f"\nSummary:")
print(f"  Gradients: {np.max(g_mag_all):.1f} mT/m (expected ~21 mT/m)")
print(f"  K-space: {k_max_cycles:.1f} 1/m (expected {k_nyquist:.1f} 1/m)")
print(f"  Readout: {traj['readout_time']:.2f} ms (expected ~1.67 ms)")