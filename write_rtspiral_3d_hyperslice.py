#%%
"""
3D Real-Time Spiral Sequence Generator with HyperSLICE-style Trajectory Design

This script generates PyPulseq 3D stack-of-spirals sequences using HyperSLICE
trajectory parameterization:
- Variable-density control via cutoff radii and density factors
- Temporal resolution optimization (iterates arms to fit time window)
- 3D kz phase encoding with ping-pong or gaussian ordering
- Maintains PyPulseq output for scanner deployment

Based on write_rtspiral_dd.py (2D HyperSLICE) and write_rtspiral_3d.py (3D encoding)

Author: rtspiral_pypulseq team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from pypulseq import Opts
from pypulseq import (make_adc, make_sinc_pulse, make_arbitrary_rf, make_digital_output_pulse, make_delay,
                      make_arbitrary_grad, make_trapezoid,
                      calc_duration, calc_rf_center,
                      rotate, add_gradients, make_label)
from pypulseq.Sequence.sequence import Sequence
from utils import schedule_FA, load_params
from utils.traj_utils import save_metadata
from utils.temporal_utils import calculate_temporal_resolution, print_temporal_summary
try:
    from utils.plot_utils import plot_trajectory_summary, plot_gradient_correction_info
except ImportError:
    plot_trajectory_summary = None
    plot_gradient_correction_info = None
from libspiral import plotgradinfo, raster_to_grad

# Import trajectory design functions from local source
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'modules' / 'libspiral' / 'src'))

# Check if TensorFlow-MRI is available for trajectory generation
try:
    import tensorflow_mri
    USE_TENSORFLOW_MRI = True
    from tensorflow_mri_trajectory import gen_spiral_traj_tfmri, convert_to_pypulseq_format
    print("Using TensorFlow-MRI for trajectory generation")
except ImportError:
    USE_TENSORFLOW_MRI = False
    from libspiral_hyperslice import vds_hyperslice
    print("TensorFlow-MRI not available, using native HyperSLICE implementation")

from librewinder.design_rewinder import design_rewinder
from kernels.kernel_handle_preparations import kernel_handle_preparations, kernel_handle_end_preparations
from math import ceil
from sigpy.mri.rf import slr
import copy
import argparse
import os
import warnings

# Cmd args
parser = argparse.ArgumentParser(
                    prog='Write3DRTSpiralHyperSLICE',
                    description='Generates a 3D stack-of-spirals Pulseq sequence with HyperSLICE trajectory design.')

parser.add_argument('-c', '--config', type=str, default='config_3d_hyperslice.toml', help='Config file path.')

args = parser.parse_args()


print(f'Using config file: {args.config}.')
print('=' * 60)
print('3D HyperSLICE-style Stack-of-Spirals Trajectory Design Mode')
print('=' * 60)

# Load and prep system and sequence parameters
params = load_params(args.config, './')

system = Opts(
    max_grad = params['system']['max_grad'], grad_unit="mT/m",
    max_slew = params['system']['max_slew'], slew_unit="T/m/s",
    grad_raster_time = params['system']['grad_raster_time'],  # [s] ( 10 us)
    rf_raster_time   = params['system']['rf_raster_time'],    # [s] (  1 us)
    rf_ringdown_time = params['system']['rf_ringdown_time'],  # [s] ( 10 us)
    rf_dead_time     = params['system']['rf_dead_time'],      # [s] (100 us)
    adc_dead_time    = params['system']['adc_dead_time'],     # [s] ( 10 us)
)

GRT = params['system']['grad_raster_time']

spiral_sys = {
    'max_slew'          :  params['system']['max_slew']*params['spiral']['slew_ratio'],   # [T/m/s]
    'max_grad'          :  params['system']['max_grad']*0.99,   # [mT/m]
    'adc_dwell'         :  params['spiral']['adc_dwell'],  # [s]
    'grad_raster_time'  :  GRT, # [s]
    'os'                :  8
    }

fov   = params['acquisition']['fov'] # [cm]

# Check if HyperSLICE parameters are provided
if 'hyperslice' not in params['spiral']:
    raise ValueError(
        "HyperSLICE parameters not found in config file. "
        "Please add [spiral.hyperslice] section. "
        "See config_3d_hyperslice.toml for reference."
    )

hs_params = params['spiral']['hyperslice']

# HyperSLICE parameters
base_resolution = hs_params.get('base_resolution', 240)
vd_spiral_arms = hs_params.get('vd_spiral_arms', 16)
vd_inner_cutoff = hs_params.get('vd_inner_cutoff', 0.15)
pre_vd_outer_cutoff = hs_params.get('pre_vd_outer_cutoff', 0.41288)
vd_outer_density = hs_params.get('vd_outer_density', 0.07)
vd_type = hs_params.get('vd_type', 'hanning')
min_max_arm_time = hs_params.get('min_max_arm_time', [0.88, 1.67])  # [ms]
max_tempres = hs_params.get('max_tempres', 55.0)  # [ms]
deadtime_ms = hs_params.get('deadtime', 2.0)  # [ms]
reverse_traj = hs_params.get('reverse', False)

print(f'\nHyperSLICE Parameters:')
print(f'  Base resolution:      {base_resolution}')
print(f'  Initial spiral arms:  {vd_spiral_arms}')
print(f'  VD inner cutoff:      {vd_inner_cutoff}')
print(f'  VD outer cutoff (pre):{pre_vd_outer_cutoff}')
print(f'  VD outer density:     {vd_outer_density}')
print(f'  VD type:              {vd_type}')
print(f'  Min/max arm time:     {min_max_arm_time} ms')
print(f'  Max temporal res:     {max_tempres} ms')
print(f'  Deadtime:             {deadtime_ms} ms')
print(f'  Reverse trajectory:   {reverse_traj}')
print()

# Design the spiral trajectory
if USE_TENSORFLOW_MRI:
    # Use TensorFlow-MRI for exact HyperSLICE implementation
    print("Generating trajectory with TensorFlow-MRI...")

    # Convert parameters to TensorFlow-MRI format
    trajectory = gen_spiral_traj_tfmri(
        base_resolution=base_resolution,
        field_of_view=fov[0] * 10,  # cm to mm
        vd_spiral_arms=vd_spiral_arms,
        vd_inner_cutoff=vd_inner_cutoff,
        pre_vd_outer_cutoff=pre_vd_outer_cutoff,
        vd_outer_density=vd_outer_density,
        vd_type=vd_type,
        max_grad_ampl=spiral_sys['max_grad'],  # mT/m
        min_rise_time=1000.0 / spiral_sys['max_slew'],  # T/m/s to us/mT/m
        dwell_time=spiral_sys['adc_dwell'] * 1e6,  # s to us
        gradient_delay=0.56,  # us (from HyperSLICE)
        readoutOS=2.0,  # from HyperSLICE
        deadtime=deadtime_ms
    )

    # Extract trajectory for first arm (PyPulseq format)
    k, g, s, t = convert_to_pypulseq_format(trajectory)

    # Set parameters from TensorFlow-MRI results
    n_int = trajectory['n_arms']
    views = n_int  # For single shot
    ro_time_ms = trajectory['readout_time']
    temp_res_ms = ro_time_ms + deadtime_ms

    print(f"TensorFlow-MRI trajectory generated:")
    print(f"  Samples per arm: {trajectory['n_samples']}")
    print(f"  Number of arms: {trajectory['n_arms']}")
    print(f"  Readout time: {ro_time_ms:.3f} ms")

else:
    # Use native HyperSLICE implementation
    k, g, s, t, n_int, views, ro_time_ms, temp_res_ms = vds_hyperslice(
        sys=spiral_sys,
        base_resolution=base_resolution,
        fov=fov,
        vd_spiral_arms=vd_spiral_arms,
        vd_inner_cutoff=vd_inner_cutoff,
        pre_vd_outer_cutoff=pre_vd_outer_cutoff,
        vd_outer_density=vd_outer_density,
        vd_type=vd_type,
        min_max_arm_time=min_max_arm_time,
        max_tempres=max_tempres,
        deadtime=deadtime_ms,
        optimize_arms=hs_params.get('optimize_arms', True)
    )

    if g is None:
        raise RuntimeError("Failed to design spiral trajectory. Please check your parameters.")

# Calculate resolution from base_resolution and FOV
res = (fov[0] * 1e-2 / base_resolution) * 1e3  # [cm] -> [m] -> [mm]
print(f'\nCalculated spatial resolution: {res:.3f} mm')
print(f'Number of spiral arms (interleaves): {n_int}')

# Reverse trajectory if requested
if reverse_traj:
    print('Reversing trajectory...')
    k = np.flip(k, axis=0)
    g = np.flip(g, axis=0)
    s = np.flip(s, axis=0)
    t = np.flip(t)

# Raster to gradient raster time
t_grad, g_grad = raster_to_grad(g, spiral_sys['adc_dwell'], GRT)

# Design rewinder
# Only rotate gradients if both rotate_grads=True AND grad_rew_method='gropt'
if params['spiral']['rotate_grads'] and params['spiral']['grad_rew_method'] == 'gropt':
    g_rewind_x, g_rewind_y, g_grad = design_rewinder(g_grad, params['spiral']['rewinder_time'], system,
                                             slew_ratio=params['spiral']['slew_ratio'],
                                             grad_rew_method=params['spiral']['grad_rew_method'],
                                             M1_nulling=params['spiral']['M1_nulling'], rotate_grads=True)
else:
    g_rewind_x, g_rewind_y = design_rewinder(g_grad, params['spiral']['rewinder_time'], system,
                                             slew_ratio=params['spiral']['slew_ratio'],
                                             grad_rew_method=params['spiral']['grad_rew_method'],
                                             M1_nulling=params['spiral']['M1_nulling'], rotate_grads=False)

# concatenate g and g_rewind, and plot.
g_grad = np.concatenate((g_grad, np.stack([g_rewind_x[0:], g_rewind_y[0:]]).T))

if params['user_settings']['show_plots']:
    # Use enhanced plotting if available
    if USE_TENSORFLOW_MRI and plot_trajectory_summary and 'trajectory' in locals():
        fig = plot_trajectory_summary(trajectory, params)
        plt.show()

        if plot_gradient_correction_info:
            info_fig = plot_gradient_correction_info()
            plt.show()

    # Also show the standard gradient info plot
    plotgradinfo(g_grad, GRT)
    plt.show()

#%%
# Excitation - support both sinc and SLR for slab-selective 3D
excitation_type = params['acquisition'].get('excitation', 'sinc')
tbwp = params['acquisition']['tbwp']

if excitation_type == 'sinc':
    rf, gz, gzr = make_sinc_pulse(flip_angle=params['acquisition']['flip_angle']/180*np.pi,
                                    duration=params['acquisition']['rf_duration'],
                                    slice_thickness=params['acquisition']['slice_thickness']*1e-3, # [mm] -> [m]
                                    time_bw_product=tbwp,
                                    return_gz=True,
                                    use='excitation', system=system)
else:  # SLR pulse
    alpha = params['acquisition']['flip_angle']
    dt = system.rf_raster_time
    raster_ratio = int(system.grad_raster_time / system.rf_raster_time)
    Trf = params['acquisition']['rf_duration']

    n = ceil((Trf/dt)/(4*raster_ratio))*4*raster_ratio
    Trf = n*dt
    bw = tbwp/Trf
    signal = slr.dzrf(n=n, tb=tbwp, ptype='st', ftype='ls', d1=0.01, d2=0.01, cancel_alpha_phs=False)

    rf, gz = make_arbitrary_rf(signal=signal, slice_thickness=params['acquisition']['slice_thickness']*1e-3,
                               bandwidth=bw, flip_angle=alpha * np.pi / 180,
                               system=system, return_gz=True, use="excitation")
    gzr = make_trapezoid(channel='z', area=-gz.area/2, system=system)

gzrr = copy.deepcopy(gzr)
gzrr.delay = 0

additional_delay = 0
if excitation_type == 'slr':
    # For SLR, we may need additional delay for rf_dead_time
    rf_delay_needed = calc_duration(gzrr) + gz.rise_time
    if rf_delay_needed < params['system']['rf_dead_time']:
        additional_delay = params['system']['rf_dead_time'] - rf_delay_needed

rf.delay = calc_duration(gzrr) + gz.rise_time + additional_delay
gz.delay = calc_duration(gzrr) + additional_delay
gzr.delay = calc_duration(gzrr, gz)
gzz = add_gradients([gzrr, gz, gzr], system=system)

# ADC
ndiscard = 10  # Number of samples to discard from beginning
Tread = ro_time_ms * 1e-3  # [ms] -> [s]
num_samples = np.floor(Tread/spiral_sys['adc_dwell']) + ndiscard
discard_delay_t = ceil((ndiscard*spiral_sys['adc_dwell']+GRT/2)/GRT)*GRT
adc = make_adc(num_samples, dwell=spiral_sys['adc_dwell'], delay=discard_delay_t, system=system)
grad_delay = 0  # Gradients start immediately

# Readout gradients
gsp_x = make_arbitrary_grad(channel='x', waveform=g_grad[:,0]*42577.478518, first=0, last=0, delay=grad_delay, system=system)
gsp_y = make_arbitrary_grad(channel='y', waveform=g_grad[:,1]*42577.478518, first=0, last=0, delay=grad_delay, system=system)

# Set the Slice rewinder balance gradients delay
gzrr.delay = calc_duration(gsp_x, gsp_y, adc)

# create a crusher gradient (only for FLASH/FISP)
if params['spiral']['contrast'] == 'FLASH' or params['spiral']['contrast'] == 'FISP':
    crush_area = (4 / (params['acquisition']['slice_thickness'] * 1e-3)) + (-1 * gzr.area)
    gz_crush = make_trapezoid(channel='z',
                              area=crush_area,
                              max_grad=system.max_grad,
                              system=system)

# Set up spiral arm rotations
gsp_xs = []
gsp_ys = []

# Override arm ordering for HyperSLICE mode
hs_ordering = hs_params.get('ordering', 'linear')
print(f"Spiral arm ordering (HyperSLICE mode): {hs_ordering}.")

if hs_ordering == 'linear':
    if (n_int%2) == 1 and (params['acquisition']['repetitions']%2) == 1:
        warnings.warn("Number of interleaves is odd. To solve this, we increased it by 1.")
        n_int += 1
    for i in range(0, n_int):
        gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=2*np.pi*i/n_int)
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        params['spiral']['GA_angle'] = 360/n_int
    n_arms = n_int

elif hs_ordering == 'golden' or hs_ordering == 'ga':
    n_arms = params['spiral'].get('GA_steps', n_int * params['acquisition']['repetitions'])
    if (n_arms%2) == 1 and (params['acquisition']['repetitions']%2) == 1:
        warnings.warn(
                    '''
                      ========================================
                      Number of arms in the sequence is odd.
                      This may create steady state artifacts.
                      To avoid this issue, set repetitions to an even number.
                      ========================================
                      ''')

    ang = 0
    ga_angle = params['spiral'].get('GA_angle', 137.5077)
    for i in range(0, n_arms):
        gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=ang)
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        ang += ga_angle*np.pi/180
        ang = ang % (2*np.pi)
    params['spiral']['GA_angle'] = ga_angle

elif hs_ordering == 'tiny' or hs_ordering == 'tinyga':
    tiny_number = hs_params.get('tiny_number', 7)
    tiny_ga = 360.0 / (n_int / tiny_number)
    print(f"  Using tiny golden angle: {tiny_ga:.4f} deg (tiny_number={tiny_number})")

    n_arms = n_int * params['acquisition']['repetitions']
    ang = 0
    for i in range(0, n_arms):
        gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=ang)
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        ang += tiny_ga*np.pi/180
        ang = ang % (2*np.pi)
    params['spiral']['GA_angle'] = tiny_ga

else:
    raise Exception(f"Unknown arm ordering: {hs_ordering}")

#%%
# Set up 3D kz encoding
acquisition_type = '3D' if 'kz_encoding' in params['acquisition'] else '2D'
gzs = []
dummy_trap = make_trapezoid(channel="z", area=0, system=system)
kz_encoding_str = ''
nkz = 1
kz_idx = np.array([0])

if acquisition_type == '3D':
    kz_fov = params['acquisition']['kz_encoding']['FOV'] * 1e-3  # mm to m
    nkz = params['acquisition']['kz_encoding']['repetitions']
    phase_areas = (np.arange(nkz) - (nkz / 2)) * (1 / kz_fov)

    # Make the largest trapezoid, and use its duration for all of them
    dummy_trap = make_trapezoid(channel="z", area=phase_areas[0], system=system)

    kz_encoding_str = params['acquisition']['kz_encoding']['ordering']
    print(f"Kz encoding ordering is {kz_encoding_str}.")

    if kz_encoding_str == 'ping-pong':
        kz_idx = np.hstack((np.arange(nkz), np.flip(np.arange(nkz))))
        for i in range(0, kz_idx.shape[0]):
            gzs.append(make_trapezoid(channel='z', area=phase_areas[kz_idx[i]],
                                      duration=calc_duration(dummy_trap), system=system))
    elif kz_encoding_str == 'gaussian':
        # Center-weighted ordering (adjust for your nkz)
        if nkz == 16:
            kz_idx = np.array([8,7,9,6,10,5,11,4,12,3,13,2,14,1,15,0])
        else:
            # Default center-out ordering
            center = nkz // 2
            kz_idx = np.zeros(nkz, dtype=int)
            kz_idx[0] = center
            for i in range(1, nkz):
                if i % 2 == 1:
                    kz_idx[i] = center + (i + 1) // 2
                else:
                    kz_idx[i] = center - i // 2
            kz_idx = np.clip(kz_idx, 0, nkz - 1)

        for i in range(0, len(kz_idx)):
            gzs.append(make_trapezoid(channel='z', area=phase_areas[kz_idx[i]],
                                      duration=calc_duration(dummy_trap), system=system))
    elif kz_encoding_str == 'linear':
        kz_idx = np.arange(nkz)
        for i in range(0, nkz):
            gzs.append(make_trapezoid(channel='z', area=phase_areas[i],
                                      duration=calc_duration(dummy_trap), system=system))
    else:
        raise ValueError(f"Unknown kz encoding ordering: {kz_encoding_str}")

    # Add the rotation_type to the encoding string
    kz_encoding_str = kz_encoding_str + '_' + params['acquisition']['kz_encoding']['rotation_type']

    print(f'3D acquisition: {n_arms} spiral arms x {nkz} kz partitions')
    print(f'kz FOV: {kz_fov*1e3:.1f} mm, Resolution: {kz_fov*1e3/nkz:.2f} mm')

# Set the delays
# TE - include kz gradient time for 3D
if params['acquisition']['TE'] == 0:
    TEd = 0
    TE = rf.shape_dur - calc_rf_center(rf)[0] + calc_duration(gzr) - gzr.delay + gsp_x.delay
    if acquisition_type == '3D':
        TE = TE + calc_duration(gzs[0])
    print(f'Min TE is set: {TE*1e3:.3f} ms.')
    params['acquisition']['TE'] = TE
else:
    TE = params['acquisition']['TE']*1e-3
    TEd = TE - (rf.shape_dur - calc_rf_center(rf)[0] + calc_duration(gzr) + gsp_x.delay)
    if acquisition_type == '3D':
        TEd = TEd - calc_duration(gzs[0])
    assert TEd >= 0, "Required TE can not be achieved."

# TR - include kz gradient time (encode + rewind) for 3D
if params['acquisition']['TR'] == 0:
    TRd = 0
    TR = calc_duration(rf, gzz) + TEd + calc_duration(gsp_xs[0], gsp_ys[0], adc)
    if params['spiral']['contrast'] in ('FLASH', 'FISP'):
        TR = TR + calc_duration(gz_crush)
    if acquisition_type == '3D':
        TR = TR + (calc_duration(gzs[0]) * 2)  # encode + rewind
    print(f'Min TR is set: {TR*1e3:.3f} ms.')
    params['acquisition']['TR'] = TR
else:
    TR = params['acquisition']['TR']*1e-3
    TRd = TR - (calc_duration(rf, gzz) + TEd + calc_duration(gsp_xs[0], gsp_ys[0], adc))
    if params['spiral']['contrast'] in ('FLASH', 'FISP'):
        TRd = TRd - calc_duration(gz_crush)
    if acquisition_type == '3D':
        TRd = TRd - (calc_duration(gzs[0]) * 2)
    assert TRd >= 0, "Required TR can not be achieved."

TE_delay = make_delay(TEd)
TR_delay = make_delay(TRd)

# Calculate total TRs
# For 3D: outer loop = arms, inner loop = kz partitions
n_TRs = n_arms * nkz * params['acquisition']['repetitions']

# Print temporal resolution summary
print('\n')
print_temporal_summary(
    arm_time=ro_time_ms,
    deadtime=deadtime_ms,
    num_views=views,
    n_arms_total=n_TRs,
    temporal_res=temp_res_ms
)
print(f'Total TRs: {n_TRs} ({n_arms} arms x {nkz} kz x {params["acquisition"]["repetitions"]} reps)')

# Sequence looping
seq = Sequence(system)

# handle any preparation pulses.
prep_str = kernel_handle_preparations(seq, params, system, rf=rf, gz=gzz)

# useful for end_preparation pulses.
params['flip_angle_last'] = params['acquisition']['flip_angle']

# tagging pulse pre-prep (only if fa_schedule exists)
rf_amplitudes, FA_schedule_str = schedule_FA(params, n_TRs)

# RF spoiling setup
rf_phase = 0
rf_inc = 0

if params['spiral']['contrast'] == 'FLASH':
    linear_phase_increment = 0
    quadratic_phase_increment = np.deg2rad(117)
elif params['spiral']['contrast'] in ('trueFISP', 'FISP'):
    linear_phase_increment = np.deg2rad(180)
    quadratic_phase_increment = 0
else:
    print("Unknown contrast type. Assuming trueFISP.")
    linear_phase_increment = np.deg2rad(180)
    quadratic_phase_increment = 0
    params['spiral']['contrast'] = 'trueFISP'

enable_trigger = True
trig = make_digital_output_pulse(channel='ext1', duration=0.001, system=system)

_, rf.shape_IDs = seq.register_rf_event(rf)

# Main sequence loop
# Outer loop: spiral arms
# Inner loop: kz partitions
tr_counter = 0
for arm_i in range(n_arms):
    for kz_i in range(len(gzs) if acquisition_type == '3D' else 1):
        curr_rf = copy.deepcopy(rf)

        # Check if we are using a ramped FA scheme
        if len(rf_amplitudes) > 0:
            if tr_counter >= len(rf_amplitudes):
                break
            curr_rf.signal = rf.signal * rf_amplitudes[tr_counter] / np.deg2rad(params['acquisition']['flip_angle'])

        curr_rf.phase_offset = rf_phase
        adc.phase_offset = rf_phase

        rf_inc = np.mod(rf_inc + quadratic_phase_increment, np.pi * 2)
        rf_phase = np.mod(rf_phase + linear_phase_increment + rf_inc, np.pi * 2)

        # RF excitation with trigger
        if enable_trigger:
            seq.add_block(trig, curr_rf, gzz)
        else:
            seq.add_block(curr_rf, gzz)

        # Add kz phase encoding gradient (3D only)
        if acquisition_type == '3D':
            seq.add_block(gzs[kz_i])

        seq.add_block(TE_delay)

        # LABEL extensions
        if acquisition_type == '3D':
            seq.add_block(make_label('PAR', 'SET', int(kz_idx[kz_i % len(kz_idx)])))
        seq.add_block(make_label('LIN', 'SET', arm_i % len(gsp_xs)))

        # Readout
        seq.add_block(gsp_xs[arm_i % len(gsp_xs)], gsp_ys[arm_i % len(gsp_ys)], adc)

        # kz rewinder (3D only) - negate the phase encoding
        if acquisition_type == '3D':
            gz_rewind = copy.deepcopy(gzs[kz_i])
            gz_rewind.amplitude = -gz_rewind.amplitude
            seq.add_block(gz_rewind)

        # Crusher gradient (FLASH/FISP only)
        if params['spiral']['contrast'] in ('FLASH', 'FISP'):
            seq.add_block(gz_crush)

        seq.add_block(TR_delay)
        tr_counter += 1

# Update n_TRs to actual count
n_TRs = tr_counter

# handle any end_preparation pulses.
end_prep_str = kernel_handle_end_preparations(seq, params, system, rf=rf, gz=gzz)

# Quick timing check
ok, error_report = seq.check_timing()

if ok:
    print("\nTiming check passed successfully")
else:
    print("\nTiming check failed. Error listing follows:")
    [print(e) for e in error_report]

# Plot the sequence
if params['user_settings']['show_plots']:
    seq.plot(show_blocks=True, grad_disp='mT/m', plot_now=False, time_disp='ms')
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    if acquisition_type == '3D':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(k_traj_adc[0,:], k_traj_adc[1,:], k_traj_adc[2,:])
        ax.set_xlabel('$k_x [mm^{-1}]$')
        ax.set_ylabel('$k_y [mm^{-1}]$')
        ax.set_zlabel('$k_z [mm^{-1}]$')
        ax.set_title('3D k-Space Trajectory (HyperSLICE Stack-of-Spirals)')
    else:
        plt.figure()
        plt.plot(k_traj_adc[0,:], k_traj_adc[1,:])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('$k_x [mm^{-1}]$')
        plt.ylabel('$k_y [mm^{-1}]$')
        plt.title('k-Space Trajectory - Readout Only (HyperSLICE Design)')

    plt.rcParams.update({'font.size': 14})

    if 'acoustic_resonances' in params and 'frequencies' in params['acoustic_resonances']:
        resonances = []
        for idx in range(len(params['acoustic_resonances']['frequencies'])):
            resonances.append({'frequency': params['acoustic_resonances']['frequencies'][idx],
                             'bandwidth': params['acoustic_resonances']['bandwidths'][idx]})
        try:
            seq.calculate_gradient_spectrum(acoustic_resonances=resonances)
            plt.title('Gradient spectrum')
        except (ValueError, UserWarning) as e:
            print(f"\nWarning: Could not calculate gradient spectrum: {e}")
            print("Continuing without acoustic resonance analysis...")
    plt.show()

# Detailed report if requested
if params['user_settings']['detailed_rep']:
    print("\n===== Detailed Test Report =====\n")
    rep_str = seq.test_report()
    print(rep_str)

    if USE_TENSORFLOW_MRI and 'trajectory' in locals():
        print("\n" + "="*60)
        print("IMPORTANT: ACTUAL GRADIENT VALUES (from TensorFlow-MRI)")
        print("="*60)
        max_grad = np.max(np.sqrt(trajectory['gx']**2 + trajectory['gy']**2))
        print(f"Maximum Gradient: {max_grad:.1f} mT/m (actual)")
        print(f"Hardware Limit:   22.0 mT/m")
        print(f"Utilization:      {max_grad/22.0*100:.1f}%")
        print("="*60 + "\n")

# Write the sequence to file
if params['user_settings']['write_seq']:

    seq.set_definition(key="FOV", value=[fov[0]*1e-2, fov[0]*1e-2, params['acquisition']['slice_thickness']*1e-3])
    seq.set_definition(key="Slice_Thickness", value=params['acquisition']['slice_thickness']*1e-3)
    seq.set_definition(key="Name", value="sprssfp_3d_hyperslice")
    seq.set_definition(key="TE", value=TE)
    seq.set_definition(key="TR", value=TR)
    seq.set_definition(key="FA", value=params['acquisition']['flip_angle'])
    seq.set_definition(key="Resolution_mm", value=res)
    seq.set_definition(key="Temporal_Res_ms", value=temp_res_ms)
    if acquisition_type == '3D':
        seq.set_definition(key="kz_partitions", value=nkz)

    m1_str = "M1" if params['spiral']['M1_nulling'] else ""
    rev_str = "rev" if reverse_traj else ""
    seq_filename = f"spiral_3d_hs_{kz_encoding_str}_{params['spiral']['contrast']}{FA_schedule_str}{prep_str}{end_prep_str}_{hs_ordering}{params['spiral']['GA_angle']:.4f}_nTR{n_TRs}_arms{n_arms}_kz{nkz}_Tread{ro_time_ms:.2f}_TR{TR*1e3:.2f}ms_FA{params['acquisition']['flip_angle']}_{m1_str}_{rev_str}_{params['user_settings']['filename_ext']}"

    # remove double, triple, quadruple underscores, and trailing underscores
    seq_filename = seq_filename.replace("__", "_").replace("__", "_").replace("__", "_").strip("_")

    seq_path = os.path.join('out_seq', f"{seq_filename}.seq")

    os.makedirs("out_seq", exist_ok=True)
    seq.write(seq_path)

    # Export k-space trajectory
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    # Save metadata
    params_save = {
        'adc_dwell': spiral_sys['adc_dwell'],
        'ndiscard': ndiscard,
        'n_TRs': n_TRs,
        'n_int': n_int,
        'n_arms': n_arms,
        'nkz': nkz,
        'ga_rotation': params['spiral']['GA_angle'],
        'fov': fov,
        'spatial_resolution': res,
        'arm_ordering': hs_ordering,
        # HyperSLICE-specific metadata
        'hyperslice_mode': True,
        'base_resolution': base_resolution,
        'views_per_frame': views,
        'temporal_resolution_ms': temp_res_ms,
        'readout_time_ms': ro_time_ms,
        'vd_inner_cutoff': vd_inner_cutoff,
        'vd_outer_density': vd_outer_density,
        'reverse_traj': reverse_traj,
        # 3D-specific metadata
        'acquisition_type': acquisition_type,
        'kz_encoding': kz_encoding_str if acquisition_type == '3D' else None,
        'kz_fov_mm': params['acquisition']['kz_encoding']['FOV'] if acquisition_type == '3D' else None,
    }
    save_metadata(seq.signature_value, k_traj_adc, params_save, params['user_settings']['show_plots'],
                  dcf_method="hoge", out_dir="out_trajectory")

    print(f'\nMetadata file for {seq_filename} is saved as {seq.signature_value} in out_trajectory/.')

    # Save readout trajectory and gradient waveforms
    traj_dir = Path("out_trajectory")
    traj_dir.mkdir(exist_ok=True)

    # Save gradient waveforms if using TensorFlow-MRI
    if USE_TENSORFLOW_MRI:
        grad_filename = traj_dir / f"{seq.signature_value}_gradients.mat"

        grad_data = {
            'gx': trajectory['gx'],
            'gy': trajectory['gy'],
            'kx': trajectory['kx'],
            'ky': trajectory['ky'],
            'n_arms': trajectory['n_arms'],
            'n_samples_per_arm': trajectory['n_samples'],
            'dwell_time': trajectory['dwell_time'],
            'readout_time_ms': trajectory['readout_time'],
            'fov_mm': fov[0] * 10,
            'base_resolution': base_resolution,
            'nkz': nkz,
            'acquisition_type': acquisition_type,
        }

        sio.savemat(grad_filename, grad_data)
        print(f'Gradient waveforms saved to: {grad_filename}')

    # Save trajectory data
    if USE_TENSORFLOW_MRI:
        n_unique_arms = trajectory['n_arms']
        n_samples_per_arm = trajectory['n_samples']

        if hs_ordering in ['golden', 'ga', 'tinyga']:
            kx_all = []
            ky_all = []
            rotation_angles_deg = []
            rotation_angles_rad = []

            golden_angle_deg = params['spiral']['GA_angle']
            golden_angle_rad = golden_angle_deg * np.pi / 180

            n_unique_ga_rotations = params['spiral'].get('hyperslice', {}).get('n_unique_rotations',
                                    params['spiral'].get('n_unique_rotations', n_unique_arms))

            kx_base = trajectory['kx'][0, :] / (2 * np.pi)
            ky_base = trajectory['ky'][0, :] / (2 * np.pi)

            for rot in range(n_unique_ga_rotations):
                rotation_angle_rad = rot * golden_angle_rad
                rotation_angle_deg = rot * golden_angle_deg
                rotation_angles_rad.append(rotation_angle_rad)
                rotation_angles_deg.append(rotation_angle_deg)

                cos_theta = np.cos(rotation_angle_rad)
                sin_theta = np.sin(rotation_angle_rad)
                kx_rotated = kx_base * cos_theta - ky_base * sin_theta
                ky_rotated = kx_base * sin_theta + ky_base * cos_theta

                kx_all.extend(kx_rotated)
                ky_all.extend(ky_rotated)

            kx_all = np.array(kx_all)
            ky_all = np.array(ky_all)
            rotation_angles_deg = np.array(rotation_angles_deg)
            rotation_angles_rad = np.array(rotation_angles_rad)

            n_saved_trajectories = n_unique_ga_rotations
            kx_reshaped = kx_all.reshape(n_saved_trajectories, n_samples_per_arm).T
            ky_reshaped = ky_all.reshape(n_saved_trajectories, n_samples_per_arm).T
            n_saved_rotations = n_saved_trajectories

            print(f"Golden angle trajectory: 1 base arm x {n_unique_ga_rotations} rotations = {n_saved_trajectories} unique trajectories")

        else:
            kx_all = []
            ky_all = []

            for arm_idx in range(n_unique_arms):
                kx_arm = trajectory['kx'][arm_idx, :] / (2 * np.pi)
                ky_arm = trajectory['ky'][arm_idx, :] / (2 * np.pi)
                kx_all.extend(kx_arm)
                ky_all.extend(ky_arm)

            kx_all = np.array(kx_all)
            ky_all = np.array(ky_all)

            kx_reshaped = kx_all.reshape(n_unique_arms, n_samples_per_arm).T
            ky_reshaped = ky_all.reshape(n_unique_arms, n_samples_per_arm).T
            n_saved_rotations = n_unique_arms

        traj = np.stack([kx_reshaped, ky_reshaped], axis=0)
        t_adc_tfmri = np.arange(len(kx_all)) * trajectory['dwell_time']

        traj_params = {
            'base_resolution': base_resolution,
            'vd_spiral_arms': vd_spiral_arms,
            'vd_inner_cutoff': vd_inner_cutoff,
            'pre_vd_outer_cutoff': pre_vd_outer_cutoff,
            'vd_outer_density': vd_outer_density,
            'vd_type': vd_type,
            'max_grad_ampl': trajectory.get('max_grad_ampl', 22.0),
            'max_slew_rate': trajectory.get('max_slew_rate', 120.0),
            'grad_raster_time': GRT,
            'readout_time_ms': trajectory['readout_time'],
            'dwell_time': trajectory['dwell_time'],
            'TR_ms': TR * 1e3,
            'deadtime_ms': params['spiral']['hyperslice'].get('deadtime', 0.5),
            'generation_method': 'TensorFlow-MRI',
            'trajectory_type': 'spiral',
            'variable_density': True,
            'k_max': np.sqrt(np.max(kx_all**2 + ky_all**2)),
            'k_nyquist': 0.5 / (fov[0] * 10 / base_resolution),
            'acquisition_type': acquisition_type,
            'nkz': nkz,
        }

        if hs_ordering in ['golden', 'ga', 'tinyga']:
            rotation_data = {
                'rotation_angles_deg': rotation_angles_deg,
                'rotation_angles_rad': rotation_angles_rad,
                'golden_angle_deg': golden_angle_deg,
                'golden_angle_rad': golden_angle_rad,
            }
        else:
            rotation_data = None

        readout_traj = {
            'kx': kx_all,
            'ky': ky_all,
            'traj': traj,
            't': t_adc_tfmri,
            'n_rotations': n_saved_rotations,
            'n_samples_per_rotation': n_samples_per_arm,
            'adc_dwell': trajectory['dwell_time'],
            'fov': fov[0],
            'resolution': res,
            'base_resolution': base_resolution,
            'temporal_resolution_ms': temp_res_ms,
            'ordering': hs_ordering,
            'ga_angle': params['spiral']['GA_angle'],
            'n_base_arms': n_unique_arms,
            'n_unique_ga_rotations': n_unique_ga_rotations if hs_ordering in ['golden', 'ga', 'tinyga'] else 0,
            'total_TRs': n_TRs,
            'traj_params': traj_params,
            'acquisition_type': acquisition_type,
            'nkz': nkz,
        }

        if rotation_data is not None:
            readout_traj['rotation_data'] = rotation_data
    else:
        kx_reshaped = k_traj_adc[0, :].reshape(n_TRs, int(num_samples)).T
        ky_reshaped = k_traj_adc[1, :].reshape(n_TRs, int(num_samples)).T
        traj = np.stack([kx_reshaped, ky_reshaped], axis=0)

        readout_traj = {
            'kx': k_traj_adc[0, :],
            'ky': k_traj_adc[1, :],
            'kz': k_traj_adc[2, :] if acquisition_type == '3D' else None,
            'traj': traj,
            't': t_adc,
            'n_rotations': n_TRs,
            'n_samples_per_rotation': int(num_samples),
            'adc_dwell': spiral_sys['adc_dwell'],
            'fov': fov[0],
            'resolution': res,
            'base_resolution': base_resolution,
            'temporal_resolution_ms': temp_res_ms,
            'ordering': hs_ordering,
            'ga_angle': params['spiral']['GA_angle'],
            'acquisition_type': acquisition_type,
            'nkz': nkz,
        }

    traj_filename = traj_dir / f"{seq.signature_value}_readout.mat"
    sio.savemat(traj_filename, {'k_readout': readout_traj})

    print(f'Readout trajectory saved to {traj_filename}.')

# %%
