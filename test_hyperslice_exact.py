#!/usr/bin/env python
"""
Test exact HyperSLICE trajectory generation matching the original implementation
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow_mri import spiral_trajectory

# Exact parameters from config_optimized_traj()
config = {
    'flow': 0,
    'base_resolution': 240,
    'field_of_view': 400,  # mm
    'phases': 12,
    'ordering': 'linear',
    'max_tempres': 55,  # ms
    'min_max_arm_time': [0.88, 1.67],  # ms
    'vd_spiral_arms': 16,
    'vd_inner_cutoff': 0.15,
    'pre_vd_outer_cutoff': 0.41288,
    'vd_outer_density': 0.07,
    'vd_type': 'hanning',
    'max_grad_ampl': 22.0,  # mT/m
    'min_rise_time': 10.0,  # us/mT/m
    'dwell_time': 1.4,  # us
    'gradient_delay': 0.56,  # us
    'readoutOS': 2.0,
    'deadtime': 2.0,  # ms
    'reverse': True,
}

# Calculate vd_outer_cutoff exactly as HyperSLICE does
vd_outer_cutoff = config['vd_inner_cutoff'] + 0.1 + config['pre_vd_outer_cutoff'] * (1 - config['vd_inner_cutoff'] - 0.1)
print(f"Calculated vd_outer_cutoff: {vd_outer_cutoff:.5f}")

# Test 1: Single arm generation (as HyperSLICE does first)
print("\n1. Single arm generation (views=1, phases=1):")
traj_single = spiral_trajectory(
    base_resolution=config['base_resolution'],
    spiral_arms=config['vd_spiral_arms'],
    field_of_view=config['field_of_view'],
    max_grad_ampl=config['max_grad_ampl'],
    min_rise_time=config['min_rise_time'],
    dwell_time=config['dwell_time'],
    views=1,
    phases=1,
    ordering=config['ordering'],
    vd_inner_cutoff=config['vd_inner_cutoff'],
    vd_outer_cutoff=vd_outer_cutoff,
    vd_outer_density=config['vd_outer_density'],
    gradient_delay=config['gradient_delay'],
    vd_type=config['vd_type'],
    readout_os=config['readoutOS'],
    # Use default larmor_const as HyperSLICE does
)

print(f"  Shape: {traj_single.shape}")
n_samples = traj_single.shape[1] if len(traj_single.shape) > 1 else traj_single.shape[0]
readout_time = n_samples * config['dwell_time'] / 1000  # us to ms
print(f"  Samples: {n_samples}")
print(f"  Readout time: {readout_time:.3f} ms")
print(f"  Temporal resolution: {readout_time + config['deadtime']:.3f} ms")

# Test 2: Full trajectory with phases
print(f"\n2. Full trajectory (views=1, phases={config['phases']}):")
traj_full = spiral_trajectory(
    base_resolution=config['base_resolution'],
    spiral_arms=config['vd_spiral_arms'],
    field_of_view=config['field_of_view'],
    max_grad_ampl=config['max_grad_ampl'],
    min_rise_time=config['min_rise_time'],
    dwell_time=config['dwell_time'],
    views=1,
    phases=config['phases'],
    ordering=config['ordering'],
    vd_inner_cutoff=config['vd_inner_cutoff'],
    vd_outer_cutoff=vd_outer_cutoff,
    vd_outer_density=config['vd_outer_density'],
    gradient_delay=config['gradient_delay'],
    vd_type=config['vd_type'],
    readout_os=config['readoutOS'],
)

print(f"  Shape: {traj_full.shape}")

# Test 3: Density estimation (views=vd_spiral_arms)
print(f"\n3. Density estimation (views={config['vd_spiral_arms']}, phases=1):")
traj_density = spiral_trajectory(
    base_resolution=config['base_resolution'],
    spiral_arms=config['vd_spiral_arms'],
    field_of_view=config['field_of_view'],
    max_grad_ampl=config['max_grad_ampl'],
    min_rise_time=config['min_rise_time'],
    dwell_time=config['dwell_time'],
    views=config['vd_spiral_arms'],
    ordering=config['ordering'],
    vd_inner_cutoff=config['vd_inner_cutoff'],
    vd_outer_cutoff=vd_outer_cutoff,
    vd_outer_density=config['vd_outer_density'],
    gradient_delay=config['gradient_delay'],
    vd_type=config['vd_type'],
    readout_os=config['readoutOS'],
)

print(f"  Shape: {traj_density.shape}")

# Extract trajectory and calculate gradients
traj_np = traj_single.numpy()
print(f"\nDebug - trajectory shape: {traj_np.shape}")

if len(traj_np.shape) == 4:
    # Shape is (phases, views, samples, 2) or (views, phases, samples, 2)
    kx = traj_np[0, 0, :, 0]
    ky = traj_np[0, 0, :, 1]
    n_samples = traj_np.shape[2]
elif len(traj_np.shape) == 3:
    # Shape is (views, samples, 2) or (phases, samples, 2)
    kx = traj_np[0, :, 0]
    ky = traj_np[0, :, 1]
    n_samples = traj_np.shape[1]
elif len(traj_np.shape) == 2:
    kx = traj_np[:, 0]
    ky = traj_np[:, 1]
    n_samples = traj_np.shape[0]
else:
    raise ValueError(f"Unexpected shape: {traj_np.shape}")

# Update readout time with correct sample count
readout_time = n_samples * config['dwell_time'] / 1000  # us to ms
print(f"Corrected readout time: {readout_time:.3f} ms")

# Calculate gradients
dt = config['dwell_time'] * 1e-6  # us to s
gamma = 42.577478518e6  # Hz/T
dkx_dt = np.gradient(kx, dt)
dky_dt = np.gradient(ky, dt)
gx = dkx_dt / (2 * np.pi * gamma) * 1e3  # T/m to mT/m
gy = dky_dt / (2 * np.pi * gamma) * 1e3

print(f"\n4. Gradient analysis:")
print(f"  Max gradient magnitude: {np.max(np.sqrt(gx**2 + gy**2)):.2f} mT/m")
print(f"  Max gradient X: {np.max(np.abs(gx)):.2f} mT/m")
print(f"  Max gradient Y: {np.max(np.abs(gy)):.2f} mT/m")

# Calculate slew rates
dgx_dt = np.gradient(gx, dt * 1000)  # mT/m/ms
dgy_dt = np.gradient(gy, dt * 1000)
slew_x = dgx_dt / 1000  # T/m/s
slew_y = dgy_dt / 1000

print(f"  Max slew rate: {np.max(np.sqrt(slew_x**2 + slew_y**2)):.1f} T/m/s")

print("\n5. Summary:")
print(f"  Target temporal resolution: {config['max_tempres']} ms")
print(f"  Achieved temporal resolution: {readout_time + config['deadtime']:.3f} ms")
print(f"  Target readout window: {config['min_max_arm_time']} ms")
print(f"  Achieved readout time: {readout_time:.3f} ms")

if readout_time >= config['min_max_arm_time'][0] and readout_time <= config['min_max_arm_time'][1]:
    print("  ✓ Readout time within target window")
else:
    print("  ✗ Readout time outside target window")

if (readout_time + config['deadtime']) <= config['max_tempres']:
    print("  ✓ Temporal resolution meets requirement")
else:
    print("  ✗ Temporal resolution exceeds limit")