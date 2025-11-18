#!/usr/bin/env python
"""Debug gradient scaling issue in PyPulseq."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, 'modules/libspiral/src')

from pypulseq import Opts, make_arbitrary_grad
from tensorflow_mri_trajectory import gen_spiral_traj_tfmri, convert_to_pypulseq_format
from libspiral import raster_to_grad

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
GRT = 10e-6  # 10 us
t_grad, g_grad = raster_to_grad(g, traj['dwell_time'], GRT)
print(f"After interpolation: {np.max(np.abs(g_grad)):.1f} mT/m")

# Create PyPulseq system
system = Opts(max_grad=22, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
              grad_raster_time=GRT, rf_raster_time=1e-6)

# Test gradient conversion
print("\nTesting PyPulseq gradient creation...")

# Method 1: Direct conversion (what we're doing now)
try:
    gamma_hz_per_mt = 42577.478518  # Hz/mT
    gx_waveform_hz = g_grad[:, 0] * gamma_hz_per_mt  # mT/m * Hz/mT = Hz/m

    print(f"Waveform in Hz/m: max = {np.max(np.abs(gx_waveform_hz)):.0f} Hz/m")
    print(f"That should be: {np.max(np.abs(gx_waveform_hz))/gamma_hz_per_mt:.1f} mT/m")

    # Create gradient with proper parameters
    gsp_x = make_arbitrary_grad(
        channel='x',
        waveform=gx_waveform_hz,
        first=0,
        last=0,
        system=system
    )

    print("✓ Gradient created successfully")

    # Check the actual values in the gradient object
    if hasattr(gsp_x, 'waveform'):
        print(f"PyPulseq internal waveform max: {np.max(np.abs(gsp_x.waveform)):.0f}")

        # The issue might be that PyPulseq is further scaling the waveform
        # Let's check what the actual amplitude is
        print(f"Gradient amplitude attribute: {gsp_x.amplitude if hasattr(gsp_x, 'amplitude') else 'N/A'}")

except Exception as e:
    print(f"✗ Error: {e}")

# Method 2: Try without unit conversion (if PyPulseq expects mT/m)
print("\nTrying without unit conversion...")
try:
    gsp_x_direct = make_arbitrary_grad(
        channel='x',
        waveform=g_grad[:, 0],  # Direct mT/m values
        first=0,
        last=0,
        system=system
    )
    print("✓ Gradient created with direct mT/m values")

    if hasattr(gsp_x_direct, 'waveform'):
        print(f"PyPulseq internal waveform max: {np.max(np.abs(gsp_x_direct.waveform)):.2f}")

except Exception as e:
    print(f"✗ Error: {e}")

print("\nDone.")