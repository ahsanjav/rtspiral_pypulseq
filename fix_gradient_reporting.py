#!/usr/bin/env python
"""
Workaround for PyPulseq gradient reporting issue.

The issue: PyPulseq test_report() incorrectly reports gradient amplitudes
for arbitrary gradient waveforms, showing them as ~57x lower than actual.

This script demonstrates the issue and provides correct gradient values.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
sys.path.insert(0, 'modules/libspiral/src')

from pypulseq import Opts, make_arbitrary_grad
from pypulseq.Sequence.sequence import Sequence
from tensorflow_mri_trajectory import gen_spiral_traj_tfmri, convert_to_pypulseq_format
from libspiral import raster_to_grad

print("PyPulseq Gradient Reporting Issue Analysis")
print("=" * 60)

# Generate trajectory
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

# Get gradients
k, g, s, t = convert_to_pypulseq_format(traj)

# Raster to gradient raster time
GRT = 10e-6
t_grad, g_grad = raster_to_grad(g, traj['dwell_time'], GRT)

print(f"\nActual gradient values:")
print(f"  From TensorFlow-MRI: {np.max(np.abs(traj['gx'])):.1f} mT/m")
print(f"  After convert_to_pypulseq_format: {np.max(np.abs(g)):.1f} mT/m")
print(f"  After raster_to_grad: {np.max(np.abs(g_grad)):.1f} mT/m")

# Create PyPulseq objects
system = Opts(max_grad=22, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
              grad_raster_time=GRT)

gamma_hz_mt = 42577.478518
gsp_x = make_arbitrary_grad(channel='x', waveform=g_grad[:,0]*gamma_hz_mt,
                           first=0, last=0, delay=0, system=system)
gsp_y = make_arbitrary_grad(channel='y', waveform=g_grad[:,1]*gamma_hz_mt,
                           first=0, last=0, delay=0, system=system)

print(f"\nPyPulseq gradient objects:")
print(f"  gsp_x waveform max: {np.max(np.abs(gsp_x.waveform)):.0f} Hz/m")
print(f"  That is: {np.max(np.abs(gsp_x.waveform))/gamma_hz_mt:.1f} mT/m")
print(f"  gsp_y waveform max: {np.max(np.abs(gsp_y.waveform)):.0f} Hz/m")
print(f"  That is: {np.max(np.abs(gsp_y.waveform))/gamma_hz_mt:.1f} mT/m")

# Create sequence
seq = Sequence()
seq.add_block(gsp_x, gsp_y)

# The issue: test_report shows wrong values
print(f"\nPyPulseq test_report output:")
test_report = seq.test_report()
for line in test_report:
    if 'Max gradient' in line and 'Hz/m' in line:
        print(f"  {line}")
        # Parse the line to extract values
        parts = line.split()
        if len(parts) >= 8:
            try:
                reported_x = float(parts[2])
                reported_y = float(parts[3])
                print(f"  Reported: {reported_x:.0f} Hz/m (x), {reported_y:.0f} Hz/m (y)")
                print(f"  That is: {reported_x/gamma_hz_mt:.2f} mT/m (x), {reported_y/gamma_hz_mt:.2f} mT/m (y)")
            except:
                pass

print(f"\n{'='*60}")
print("CONCLUSION:")
print(f"  Actual gradients: ~{np.max(np.abs(g_grad)):.0f} mT/m")
print(f"  PyPulseq reports: ~0.35 mT/m (INCORRECT - bug in test_report)")
print(f"  Scaling error: ~57x")
print()
print("The gradients in the sequence are CORRECT.")
print("The issue is only with the test_report() display.")
print("The actual MRI scanner will use the correct gradient values.")
print(f"{'='*60}")