#!/usr/bin/env python
"""Analyze the gradient scaling issue."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np

# The issue:
# We input: 842704 Hz/m (19.8 mT/m)
# PyPulseq reports: 14789 Hz/m (0.35 mT/m)

ratio = 842704 / 14789
print(f"Scaling ratio: {ratio:.2f}")

# Check if this is related to the number of samples
n_samples_orig = 1172
n_samples_interp = 164
sample_ratio = n_samples_orig / n_samples_interp
print(f"Sample ratio: {sample_ratio:.2f}")

# Check if there's a time scaling issue
adc_dwell = 1.4e-6  # us
grad_raster = 10e-6  # us
time_ratio = grad_raster / adc_dwell
print(f"Time ratio: {time_ratio:.2f}")

# The ratio is close to 57, which doesn't match any obvious conversion factor
# Let me check if there's something with the interpolation

# Possible issue: The gradient waveform might be getting scaled by the number of points
# or there might be an issue with how PyPulseq interprets the waveform

print(f"\nPossible scaling factors:")
print(f"  57 ≈ {ratio:.1f} (actual ratio)")
print(f"  7.14 ≈ {time_ratio:.2f} (time ratio)")
print(f"  7.15 ≈ {sample_ratio:.2f} (sample ratio)")

# Wait, 57 ≈ 8 * 7.14
# Could there be an 8x factor somewhere?

print(f"\n8 * time_ratio = {8 * time_ratio:.1f}")

# Or perhaps it's related to the spiral arms?
n_arms = 16
print(f"\nn_arms = {n_arms}")
print(f"ratio / 3.5 = {ratio / 3.5:.1f}  (close to 16)")

# Actually, let me check the exact numbers
exact_ratio = 842704 / 14789.0
print(f"\nExact ratio: {exact_ratio:.6f}")

# This is suspiciously close to 57
# Let's check if there's a gamma factor issue
gamma_mhz = 42.577478518  # MHz/T
gamma_hz = 42577478.518  # Hz/T
gamma_khz = 42577.478518  # kHz/T

print(f"\nGamma factors:")
print(f"  MHz/T: {gamma_mhz}")
print(f"  kHz/T: {gamma_khz}")
print(f"  Hz/T: {gamma_hz}")

# If we accidentally used Hz instead of kHz somewhere...
print(f"\n1000 / ratio = {1000 / ratio:.2f}")

# Actually, wait - let me think about this differently
# What if PyPulseq is expecting a different unit?

# We're converting: mT/m * 42577.478518 Hz/mT = Hz/m
# But what if PyPulseq expects something else?

print(f"\nIf PyPulseq expects kHz/m instead of Hz/m:")
print(f"  842704 Hz/m / 1000 = {842704/1000:.1f} kHz/m")
print(f"  But that would give us 0.84 kHz/m reported, not 14.8 kHz/m")

print(f"\nIf the waveform is being divided by the number of points:")
print(f"  842704 / 164 = {842704/164:.1f}")
print(f"  Still doesn't match 14789")

# One more check - what if there's a normalization happening?
max_system_grad_hz = 22 * 42577.478518  # 22 mT/m in Hz/m
print(f"\nMax system gradient: {max_system_grad_hz:.0f} Hz/m")
print(f"Ratio to our max: {max_system_grad_hz / 842704:.3f}")