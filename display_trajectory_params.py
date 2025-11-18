#!/usr/bin/env python
"""
Display all trajectory parameters saved in the readout file.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import json

# Find the most recent trajectory file
traj_dir = Path("out_trajectory")
readout_files = list(traj_dir.glob("*_readout.mat"))

if not readout_files:
    print("No trajectory files found")
    exit()

latest_readout = max(readout_files, key=lambda p: p.stat().st_mtime)
print(f"Loading: {latest_readout.name}")
print("=" * 70)

# Load trajectory data
data = sio.loadmat(latest_readout)
k_readout = data['k_readout']

# Function to safely extract scalar values
def get_scalar(val):
    """Extract scalar from various MATLAB formats."""
    if hasattr(val, 'shape'):
        if val.shape == (1, 1):
            return val[0, 0]
        elif val.shape == (1,):
            return val[0]
    return val

# Display basic information
print("\n📊 BASIC TRAJECTORY INFO")
print("-" * 40)
print(f"Ordering:              {get_scalar(k_readout['ordering'][0,0])}")
print(f"Rotations saved:       {get_scalar(k_readout['n_rotations'][0,0])}")
print(f"Samples per rotation:  {get_scalar(k_readout['n_samples_per_rotation'][0,0])}")
print(f"Total samples:         {len(k_readout['kx'][0,0].flatten()):,}")

# Display acquisition parameters
print("\n🎯 ACQUISITION PARAMETERS")
print("-" * 40)
print(f"FOV:                   {get_scalar(k_readout['fov'][0,0]):.1f} cm")
print(f"Base resolution:       {get_scalar(k_readout['base_resolution'][0,0])}")
print(f"Resolution:            {get_scalar(k_readout['resolution'][0,0]):.2f} mm")
print(f"Temporal resolution:   {get_scalar(k_readout['temporal_resolution_ms'][0,0]):.1f} ms")
print(f"Total TRs:             {get_scalar(k_readout['total_TRs'][0,0])}")

# Display golden angle specific parameters
if 'n_unique_ga_rotations' in k_readout.dtype.names:
    n_ga = get_scalar(k_readout['n_unique_ga_rotations'][0,0])
    if n_ga > 0:
        print("\n🌟 GOLDEN ANGLE PARAMETERS")
        print("-" * 40)
        print(f"Golden angle:          {get_scalar(k_readout['ga_angle'][0,0]):.4f}°")
        print(f"Unique GA rotations:   {n_ga}")
        print(f"Base arms:             {get_scalar(k_readout['n_base_arms'][0,0])}")

# Display detailed trajectory parameters if available
if 'traj_params' in k_readout.dtype.names:
    traj_params = k_readout['traj_params'][0,0]

    print("\n🔧 HYPERSLICE PARAMETERS")
    print("-" * 40)
    if 'vd_spiral_arms' in traj_params.dtype.names:
        print(f"VD spiral arms:        {get_scalar(traj_params['vd_spiral_arms'][0,0])}")
    if 'vd_inner_cutoff' in traj_params.dtype.names:
        print(f"VD inner cutoff:       {get_scalar(traj_params['vd_inner_cutoff'][0,0]):.3f}")
    if 'pre_vd_outer_cutoff' in traj_params.dtype.names:
        print(f"Pre-VD outer cutoff:   {get_scalar(traj_params['pre_vd_outer_cutoff'][0,0]):.5f}")
    if 'vd_outer_density' in traj_params.dtype.names:
        print(f"VD outer density:      {get_scalar(traj_params['vd_outer_density'][0,0]):.3f}")
    if 'vd_type' in traj_params.dtype.names:
        print(f"VD type:               {get_scalar(traj_params['vd_type'][0,0])}")

    print("\n⚡ HARDWARE PARAMETERS")
    print("-" * 40)
    if 'max_grad_ampl' in traj_params.dtype.names:
        print(f"Max gradient:          {get_scalar(traj_params['max_grad_ampl'][0,0]):.1f} mT/m")
    if 'max_slew_rate' in traj_params.dtype.names:
        print(f"Max slew rate:         {get_scalar(traj_params['max_slew_rate'][0,0]):.0f} T/m/s")
    if 'grad_raster_time' in traj_params.dtype.names:
        print(f"Gradient raster time:  {get_scalar(traj_params['grad_raster_time'][0,0])*1e6:.1f} μs")

    print("\n⏱️ TIMING PARAMETERS")
    print("-" * 40)
    if 'readout_time_ms' in traj_params.dtype.names:
        print(f"Readout time:          {get_scalar(traj_params['readout_time_ms'][0,0]):.3f} ms")
    if 'dwell_time' in traj_params.dtype.names:
        print(f"ADC dwell time:        {get_scalar(traj_params['dwell_time'][0,0])*1e6:.1f} μs")
    if 'TR_ms' in traj_params.dtype.names:
        print(f"TR:                    {get_scalar(traj_params['TR_ms'][0,0]):.2f} ms")
    if 'deadtime_ms' in traj_params.dtype.names:
        print(f"Deadtime:              {get_scalar(traj_params['deadtime_ms'][0,0]):.1f} ms")

    print("\n📐 K-SPACE PARAMETERS")
    print("-" * 40)
    if 'k_max' in traj_params.dtype.names:
        print(f"k_max:                 {get_scalar(traj_params['k_max'][0,0]):.1f} 1/m")
    if 'k_nyquist' in traj_params.dtype.names:
        k_nyq = get_scalar(traj_params['k_nyquist'][0,0])
        k_max = get_scalar(traj_params['k_max'][0,0])
        print(f"k_nyquist:             {k_nyq:.1f} 1/m")
        print(f"Coverage:              {k_max/k_nyq*100:.1f}%")

    print("\n🔄 GENERATION INFO")
    print("-" * 40)
    if 'generation_method' in traj_params.dtype.names:
        print(f"Method:                {get_scalar(traj_params['generation_method'][0,0])}")
    if 'trajectory_type' in traj_params.dtype.names:
        print(f"Type:                  {get_scalar(traj_params['trajectory_type'][0,0])}")
    if 'variable_density' in traj_params.dtype.names:
        vd = get_scalar(traj_params['variable_density'][0,0])
        print(f"Variable density:      {'Yes' if vd else 'No'}")

    # Add unique arms info
    print("\n🌀 SPIRAL ARM INFO")
    print("-" * 40)
    print(f"Unique arms (original):{get_scalar(traj_params['vd_spiral_arms'][0,0]) if 'vd_spiral_arms' in traj_params.dtype.names else 'N/A'}")
    if 'n_base_arms' in k_readout.dtype.names:
        print(f"Base arms saved:       {get_scalar(k_readout['n_base_arms'][0,0])}")
    if 'n_unique_ga_rotations' in k_readout.dtype.names:
        n_ga = get_scalar(k_readout['n_unique_ga_rotations'][0,0])
        if n_ga > 0:
            print(f"GA rotations:          {n_ga}")
            print(f"Total trajectories:    {get_scalar(k_readout['n_rotations'][0,0])}")
else:
    print("\n⚠️  Note: Detailed trajectory parameters not available in this file")
    print("    Re-generate trajectory with latest code to include full metadata")

# Display rotation angle data if available
if 'rotation_data' in k_readout.dtype.names:
    rotation_data = k_readout['rotation_data'][0,0]
    print("\n📐 ROTATION ANGLES")
    print("-" * 40)
    if 'golden_angle_deg' in rotation_data.dtype.names:
        print(f"Golden angle:          {get_scalar(rotation_data['golden_angle_deg'][0,0]):.4f}° ({get_scalar(rotation_data['golden_angle_rad'][0,0]):.6f} rad)")
    if 'rotation_angles_deg' in rotation_data.dtype.names:
        angles_deg = rotation_data['rotation_angles_deg'][0,0].flatten()
        angles_rad = rotation_data['rotation_angles_rad'][0,0].flatten()
        print(f"Number of rotations:   {len(angles_deg)}")
        print(f"Angle range:           {angles_deg[0]:.1f}° to {angles_deg[-1]:.1f}°")
        # Show first few angles
        print(f"First 5 angles:        ", end="")
        for i in range(min(5, len(angles_deg))):
            print(f"{angles_deg[i]:.1f}°", end=" ")
        print()

print("\n" + "=" * 70)
print("All parameters successfully loaded and ready for reconstruction!")
print("=" * 70)