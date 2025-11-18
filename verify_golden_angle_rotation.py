#!/usr/bin/env python
"""
Verify golden angle rotation pattern in saved trajectory.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt

# Load the latest trajectory
traj_dir = Path("out_trajectory")
readout_files = list(traj_dir.glob("*_readout.mat"))

if not readout_files:
    print("No trajectory files found")
    exit()

latest_readout = max(readout_files, key=lambda p: p.stat().st_mtime)
print(f"Loading: {latest_readout.name}")

# Load trajectory
readout_data = sio.loadmat(latest_readout)['k_readout']
kx_readout = readout_data['kx'][0,0].flatten()
ky_readout = readout_data['ky'][0,0].flatten()
n_rotations = int(readout_data['n_rotations'][0,0])
n_samples = int(readout_data['n_samples_per_rotation'][0,0])
ga_angle = float(readout_data['ga_angle'][0,0])

print(f"\nTrajectory info:")
print(f"  Rotations: {n_rotations}")
print(f"  Samples per rotation: {n_samples}")
print(f"  Golden angle: {ga_angle:.4f}°")
print(f"  Total samples: {len(kx_readout)}")

# Verify rotation angles
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f'Golden Angle Rotation Verification (GA = {ga_angle:.4f}°)', fontsize=14)

# Plot first few rotations
n_plot = min(8, n_rotations)
colors = plt.cm.hsv(np.linspace(0, 0.8, n_plot))

for i in range(n_plot):
    start_idx = i * n_samples
    end_idx = (i + 1) * n_samples

    kx = kx_readout[start_idx:end_idx]
    ky = ky_readout[start_idx:end_idx]

    axes[0].plot(kx, ky, color=colors[i], linewidth=1, alpha=0.7, label=f'Rot {i}')

    # Mark the endpoints
    axes[0].scatter(kx[-1], ky[-1], color=colors[i], s=30, marker='x')

axes[0].set_xlabel('kx [1/m]')
axes[0].set_ylabel('ky [1/m]')
axes[0].set_title(f'First {n_plot} Rotations')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8, ncol=2)

# Check angle between consecutive rotations
angles = []
for i in range(min(20, n_rotations-1)):
    # Get endpoint of rotation i and i+1
    idx1 = (i + 1) * n_samples - 1
    idx2 = (i + 2) * n_samples - 1

    if idx2 < len(kx_readout):
        # Calculate angle from origin to endpoint
        angle1 = np.arctan2(ky_readout[idx1], kx_readout[idx1]) * 180 / np.pi
        angle2 = np.arctan2(ky_readout[idx2], kx_readout[idx2]) * 180 / np.pi

        # Calculate difference
        diff = angle2 - angle1
        # Wrap to [-180, 180]
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360

        angles.append(diff)

axes[1].plot(angles, 'bo-', linewidth=1, markersize=4)
axes[1].axhline(y=ga_angle, color='r', linestyle='--', label=f'Expected: {ga_angle:.4f}°')
axes[1].axhline(y=-ga_angle, color='r', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Rotation Index')
axes[1].set_ylabel('Angle Difference [°]')
axes[1].set_title('Angle Between Consecutive Rotations')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot all endpoints to show golden angle coverage
endpoints_x = []
endpoints_y = []
for i in range(n_rotations):
    idx = (i + 1) * n_samples - 1
    if idx < len(kx_readout):
        endpoints_x.append(kx_readout[idx])
        endpoints_y.append(ky_readout[idx])

# Color by rotation number
colors = plt.cm.hsv(np.linspace(0, 1, len(endpoints_x)))
axes[2].scatter(endpoints_x, endpoints_y, c=colors, s=20, alpha=0.6)
axes[2].set_xlabel('kx [1/m]')
axes[2].set_ylabel('ky [1/m]')
axes[2].set_title(f'All {len(endpoints_x)} Endpoints (Golden Angle Coverage)')
axes[2].set_aspect('equal')
axes[2].grid(True, alpha=0.3)

# Add circle at 300 1/m
circle = plt.Circle((0, 0), 300, fill=False, color='gray', linestyle='--', alpha=0.5)
axes[2].add_patch(circle)

plt.tight_layout()
plt.savefig('golden_angle_verification.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: golden_angle_verification.png")

# Verify that we're using a single base trajectory
print("\nVerifying single base trajectory:")
# Compare first rotation with rotation 0 (should be identical before rotation)
kx0 = kx_readout[0:n_samples]
ky0 = ky_readout[0:n_samples]

# Check if all rotations are derived from the same base
is_single_base = True
for i in range(1, min(5, n_rotations)):
    start_idx = i * n_samples
    end_idx = (i + 1) * n_samples

    kx_i = kx_readout[start_idx:end_idx]
    ky_i = ky_readout[start_idx:end_idx]

    # Rotate back by i * golden_angle and compare with base
    angle = -i * ga_angle * np.pi / 180
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    kx_rotated_back = kx_i * cos_theta - ky_i * sin_theta
    ky_rotated_back = kx_i * sin_theta + ky_i * cos_theta

    # Check if it matches the base
    diff = np.max(np.abs(kx_rotated_back - kx0)) + np.max(np.abs(ky_rotated_back - ky0))
    print(f"  Rotation {i} difference from base: {diff:.6f} 1/m")

    if diff > 0.01:  # Tolerance for numerical errors
        is_single_base = False

if is_single_base:
    print("  ✓ All rotations derived from single base trajectory")
else:
    print("  ✗ Rotations NOT from single base (unexpected)")

print("\n" + "="*60)
print("SUMMARY:")
print(f"  • {n_rotations} unique golden angle rotations")
print(f"  • Golden angle: {ga_angle:.4f}°")
print(f"  • Single base trajectory: {'Yes ✓' if is_single_base else 'No ✗'}")
print(f"  • Total samples: {len(kx_readout):,}")
print("="*60)