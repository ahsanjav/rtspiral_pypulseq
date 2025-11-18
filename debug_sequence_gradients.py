#!/usr/bin/env python
"""Debug the actual gradient values in the generated sequence."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
import toml
from pathlib import Path

# Add local paths
sys.path.insert(0, 'modules/libspiral/src')
sys.path.insert(0, 'modules/libspiral')

from pypulseq import Opts, make_arbitrary_grad, make_delay
from pypulseq.Sequence.sequence import Sequence
from tensorflow_mri_trajectory import gen_spiral_traj_tfmri, convert_to_pypulseq_format
from libspiral import raster_to_grad

# Load configuration
config = toml.load('example_config_hyperslice.toml')
GRT = config['system']['grad_raster_time']

# Generate trajectory
print("Generating trajectory...")
traj = gen_spiral_traj_tfmri(
    base_resolution=240,
    field_of_view=400,
    vd_spiral_arms=16,
    vd_inner_cutoff=0.15,
    pre_vd_outer_cutoff=0.41288,
    vd_outer_density=0.07,
    vd_type='hanning',
    max_grad_ampl=22.0,
    min_rise_time=10.0,
    dwell_time=1.4,
)

k, g, s, t = convert_to_pypulseq_format(traj)
print(f"Original gradient max: {np.max(np.abs(g)):.1f} mT/m")

# Raster to gradient raster time
t_grad, g_grad = raster_to_grad(g, traj['dwell_time'], GRT)
print(f"After interpolation: {np.max(np.abs(g_grad)):.1f} mT/m")
print(f"Gradient shape after interpolation: {g_grad.shape}")

# Create PyPulseq system
system = Opts(max_grad=22, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
              grad_raster_time=GRT)

print("\nCreating test sequence with correct scaling...")

# Create a test sequence
seq = Sequence()

# The issue might be that we need to specify the discard delay
ndiscard = 0  # For simplicity
discard_delay_t = 0

# Create gradients with correct conversion
# PyPulseq expects Hz/m for make_arbitrary_grad
gamma_hz_mt = 42577.478518  # Hz/mT (or Hz*T/mT*m = Hz/m per mT/m)

try:
    # Method 1: Convert to Hz/m as PyPulseq expects
    gx_hz = g_grad[:, 0] * gamma_hz_mt  # mT/m * Hz/mT = Hz/m
    gy_hz = g_grad[:, 1] * gamma_hz_mt

    print(f"\nGradient waveforms:")
    print(f"  gx max: {np.max(np.abs(g_grad[:, 0])):.1f} mT/m -> {np.max(np.abs(gx_hz)):.0f} Hz/m")
    print(f"  gy max: {np.max(np.abs(g_grad[:, 1])):.1f} mT/m -> {np.max(np.abs(gy_hz)):.0f} Hz/m")

    # Create gradient objects
    gsp_x = make_arbitrary_grad(channel='x', waveform=gx_hz, delay=discard_delay_t, system=system)
    gsp_y = make_arbitrary_grad(channel='y', waveform=gy_hz, delay=discard_delay_t, system=system)

    print("\n✓ Gradients created successfully")

    # Add to sequence
    seq.add_block(gsp_x, gsp_y, make_delay(0.002))

    # Get test report
    test_report = seq.test_report()
    print("\nSequence test report (gradient info):")
    for line in test_report:
        if 'gradient' in line.lower() or 'slew' in line.lower():
            print(f"  {line}")

except Exception as e:
    print(f"\n✗ Error creating gradients: {e}")
    print("\nTrying alternative approach...")

    # Method 2: Check if the issue is with first/last parameters
    try:
        gsp_x = make_arbitrary_grad(channel='x', waveform=gx_hz, first=0, last=0, delay=discard_delay_t, system=system)
        gsp_y = make_arbitrary_grad(channel='y', waveform=gy_hz, first=0, last=0, delay=discard_delay_t, system=system)
        print("✓ Gradients created with first=0, last=0")

    except Exception as e2:
        print(f"✗ Still failed: {e2}")

print("\nDone.")