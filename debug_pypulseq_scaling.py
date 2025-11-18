#!/usr/bin/env python
"""Debug PyPulseq gradient scaling issue in detail."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
from math import ceil
sys.path.insert(0, 'modules/libspiral/src')

from pypulseq import Opts, make_arbitrary_grad, make_adc, make_delay
from pypulseq.Sequence.sequence import Sequence
from tensorflow_mri_trajectory import gen_spiral_traj_tfmri, convert_to_pypulseq_format
from libspiral import raster_to_grad

# System parameters
GRT = 10e-6  # Gradient raster time
system = Opts(max_grad=22, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
              grad_raster_time=GRT)

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

# Get gradients
k, g, s, t = convert_to_pypulseq_format(traj)
print(f"Original gradient max: {np.max(np.abs(g)):.1f} mT/m")

# Raster to gradient raster time
t_grad, g_grad = raster_to_grad(g, traj['dwell_time'], GRT)
print(f"After raster_to_grad: {np.max(np.abs(g_grad)):.1f} mT/m")
print(f"Shape: {g_grad.shape}")

# Calculate delay for ADC
ndiscard = 10
adc_dwell = traj['dwell_time']
discard_delay_t = ceil((ndiscard*adc_dwell+GRT/2)/GRT)*GRT

print(f"\nCreating PyPulseq sequence...")
print(f"Discard delay: {discard_delay_t*1e6:.1f} us")

# Create sequence
seq = Sequence()

# Create gradients exactly as in write_rtspiral_dd.py
gamma_hz_mt = 42577.478518  # Hz/mT
gsp_x = make_arbitrary_grad(channel='x', waveform=g_grad[:,0]*gamma_hz_mt,
                           first=0, last=0, delay=discard_delay_t, system=system)
gsp_y = make_arbitrary_grad(channel='y', waveform=g_grad[:,1]*gamma_hz_mt,
                           first=0, last=0, delay=discard_delay_t, system=system)

# Create ADC
num_samples = np.floor(traj['readout_time']*1e-3/adc_dwell) + ndiscard
adc = make_adc(num_samples, dwell=adc_dwell, delay=0, system=system)

# Add to sequence
seq.add_block(gsp_x, gsp_y, adc)

print("\nSequence created. Checking gradients...")

# Check the gradient objects directly
print(f"gsp_x waveform max: {np.max(np.abs(gsp_x.waveform)):.0f} Hz/m")
print(f"  That is: {np.max(np.abs(gsp_x.waveform))/gamma_hz_mt:.1f} mT/m")
print(f"gsp_y waveform max: {np.max(np.abs(gsp_y.waveform)):.0f} Hz/m")
print(f"  That is: {np.max(np.abs(gsp_y.waveform))/gamma_hz_mt:.1f} mT/m")

# Write sequence and check report
print("\nWriting sequence...")
seq.write("test_gradient_scaling.seq")

# Get test report
print("\nTest report:")
test_report = seq.test_report()
for line in test_report:
    if 'gradient' in line.lower() or 'slew' in line.lower():
        print(f"  {line}")

# Now let's check what's in the actual sequence file
print("\nChecking sequence blocks...")
for block_id in range(1, min(5, seq.get_n_blocks() + 1)):
    block = seq.get_block(block_id)
    if hasattr(block, 'gx') and block.gx is not None:
        print(f"Block {block_id} gx max amplitude: {np.max(np.abs(block.gx.waveform)) if hasattr(block.gx, 'waveform') else 'N/A'}")
    if hasattr(block, 'gy') and block.gy is not None:
        print(f"Block {block_id} gy max amplitude: {np.max(np.abs(block.gy.waveform)) if hasattr(block.gy, 'waveform') else 'N/A'}")

print("\nDone.")