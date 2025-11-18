#!/usr/bin/env python
"""Test if delay parameter affects gradient scaling in PyPulseq."""

import numpy as np
from pypulseq import Opts, make_arbitrary_grad
from pypulseq.Sequence.sequence import Sequence

# Create system
system = Opts(max_grad=22, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
              grad_raster_time=10e-6)

# Create a simple gradient waveform
# Smooth ramp up and down to avoid slew rate violations
n_points = 100
t = np.linspace(0, 1, n_points)
waveform_mt = 20 * np.sin(np.pi * t)  # Smooth sine wave, max 20 mT/m

# Convert to Hz/m
gamma_hz_mt = 42577.478518  # Hz/mT
waveform_hz = waveform_mt * gamma_hz_mt

print(f"Input waveform:")
print(f"  Max value: {np.max(waveform_mt):.1f} mT/m")
print(f"  In Hz/m: {np.max(waveform_hz):.0f} Hz/m")
print(f"  Number of points: {n_points}")

# Test 1: Without delay
print("\nTest 1: No delay")
try:
    grad1 = make_arbitrary_grad(channel='x', waveform=waveform_hz, system=system)

    seq1 = Sequence()
    seq1.add_block(grad1)

    # Check the gradient directly
    if hasattr(grad1, 'waveform'):
        print(f"  Internal waveform max: {np.max(np.abs(grad1.waveform)):.0f}")
        print(f"  That is: {np.max(np.abs(grad1.waveform))/gamma_hz_mt:.1f} mT/m")

    # Try to get test report
    try:
        report1 = seq1.test_report()
        for line in report1:
            if 'gradient' in line.lower():
                print(f"  {line}")
    except:
        print("  Could not generate test report")

except Exception as e:
    print(f"  Error: {e}")

# Test 2: With delay
print("\nTest 2: With 100us delay")
try:
    delay_time = 100e-6  # 100 us
    grad2 = make_arbitrary_grad(channel='x', waveform=waveform_hz, delay=delay_time, system=system)

    seq2 = Sequence()
    seq2.add_block(grad2)

    # Get test report
    report2 = seq2.test_report()
    for line in report2:
        if 'Max gradient' in line and 'Hz/m' in line:
            print(f"  {line}")

except Exception as e:
    print(f"  Error: {e}")

# Test 3: With first=0, last=0
print("\nTest 3: With first=0, last=0")
try:
    grad3 = make_arbitrary_grad(channel='x', waveform=waveform_hz, first=0, last=0, system=system)

    seq3 = Sequence()
    seq3.add_block(grad3)

    # Get test report
    report3 = seq3.test_report()
    for line in report3:
        if 'Max gradient' in line and 'Hz/m' in line:
            print(f"  {line}")

except Exception as e:
    print(f"  Error: {e}")

print("\nDone.")