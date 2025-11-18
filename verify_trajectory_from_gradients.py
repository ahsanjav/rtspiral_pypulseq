#!/usr/bin/env python
"""Verify that the k-space trajectory from gradients is correct."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
import scipy.io as sio
from pathlib import Path
sys.path.insert(0, 'modules/libspiral/src')

from tensorflow_mri_trajectory import gen_spiral_traj_tfmri

# Load saved gradient file
grad_files = list(Path("out_trajectory").glob("*_gradients.mat"))
if grad_files:
    latest_file = max(grad_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading gradient file: {latest_file}")

    data = sio.loadmat(latest_file)

    # Get gradients
    gx = data['gx']  # [n_arms, n_samples] in mT/m
    gy = data['gy']
    kx = data['kx']  # [n_arms, n_samples] in rad/m
    ky = data['ky']

    print(f"\nGradient data from saved file:")
    print(f"  gx shape: {gx.shape}")
    print(f"  Max gradient: {np.max(np.sqrt(gx**2 + gy**2)):.1f} mT/m")

    # Calculate k-space from gradients
    gamma = 2 * np.pi * 42.577478518e6  # rad/s/T
    dt = float(data['dwell_time'][0, 0])  # seconds

    print(f"\nIntegrating gradients to get k-space:")
    print(f"  dt = {dt*1e6:.1f} us")

    # Integrate first arm
    kx_calc = np.zeros_like(gx[0, :])
    ky_calc = np.zeros_like(gy[0, :])

    for i in range(1, len(gx[0, :])):
        kx_calc[i] = kx_calc[i-1] + gx[0, i] * gamma * 1e-3 * dt  # mT/m -> T/m
        ky_calc[i] = ky_calc[i-1] + gy[0, i] * gamma * 1e-3 * dt

    # Compare with saved k-space
    print(f"\nK-space comparison (first arm):")
    print(f"  Saved kx max: {np.max(np.abs(kx[0, :])):.1f} rad/m")
    print(f"  Calculated kx max: {np.max(np.abs(kx_calc)):.1f} rad/m")
    print(f"  Difference: {np.max(np.abs(kx[0, :] - kx_calc)):.3f} rad/m")

    # Convert to physical units
    kx_m = kx[0, :] / (2 * np.pi)  # rad/m -> 1/m
    k_max_m = np.max(np.sqrt((kx[0, :]**2 + ky[0, :]**2))) / (2 * np.pi)

    print(f"\nK-space extent:")
    print(f"  Max |k|: {k_max_m:.1f} 1/m")

    # What should it be?
    fov_mm = float(data['fov_mm'][0, 0])
    base_res = float(data['base_resolution'][0, 0])
    fov_m = fov_mm * 1e-3
    res_m = fov_m / base_res
    k_nyquist = 0.5 / res_m

    print(f"  Expected Nyquist: {k_nyquist:.1f} 1/m")
    print(f"  Ratio: {k_max_m / k_nyquist:.3f}")

    if abs(k_max_m / k_nyquist - 1.0) < 0.1:
        print("  ✓ K-space extent is correct!")
    else:
        print("  ✗ K-space extent mismatch")

    print(f"\nConclusion:")
    print(f"  The saved gradients are {np.max(np.sqrt(gx**2 + gy**2)):.1f} mT/m")
    print(f"  The k-space reaches {k_max_m / k_nyquist:.1%} of Nyquist")
    print(f"  The trajectory is {'correct' if abs(k_max_m / k_nyquist - 1.0) < 0.1 else 'incorrect'}")

else:
    print("No gradient files found in out_trajectory/")