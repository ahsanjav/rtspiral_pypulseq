#!/usr/bin/env python
"""
Test TensorFlow-MRI trajectory generation
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# Test direct tensorflow_mri call
print("Testing direct tensorflow_mri call...")
from tensorflow_mri import spiral_trajectory

params = {
    'base_resolution': 240,
    'spiral_arms': 16,
    'field_of_view': 400,  # mm
    'max_grad_ampl': 22.0,  # mT/m
    'min_rise_time': 10.0,  # us/mT/m
    'dwell_time': 1.4,  # us
    'readout_os': 2.0,
    'gradient_delay': 0.56,  # us
    'vd_inner_cutoff': 0.15,
    'vd_outer_cutoff': 0.56288,
    'vd_outer_density': 0.07,
    'vd_type': 'hanning'
}

result = spiral_trajectory(**params)
print(f"Direct call shape: {result.shape}")
print(f"Direct call readout: {result.shape[1] * 1.4 / 1000:.3f} ms")

# Test our wrapper
print("\nTesting our wrapper...")
import sys
sys.path.insert(0, 'modules/libspiral/src')

# Clear any cached imports
if 'tensorflow_mri_trajectory' in sys.modules:
    del sys.modules['tensorflow_mri_trajectory']

from tensorflow_mri_trajectory import gen_spiral_traj_tfmri

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
    gradient_delay=0.56,  # us
    readoutOS=2.0,
    deadtime=2.0  # ms
)

print(f"Wrapper result shape: {traj['kx'].shape}")
print(f"Wrapper readout time: {traj['readout_time']:.3f} ms")