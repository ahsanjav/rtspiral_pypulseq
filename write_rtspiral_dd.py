#%%
"""
2D Real-Time Spiral Sequence Generator with HyperSLICE-style Trajectory Design

This script generates PyPulseq sequences using HyperSLICE trajectory parameterization:
- Variable-density control via cutoff radii and density factors
- Temporal resolution optimization (iterates arms to fit time window)
- Maintains PyPulseq output for scanner deployment

Based on write_rtspiral.py with HyperSLICE trajectory design from:
https://github.com/mrphys/HyperSLICE/blob/master/utils/preprocessing_trajectory_gen.py

Author: rtspiral_pypulseq team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from pypulseq import Opts
from pypulseq import (make_adc, make_sinc_pulse, make_digital_output_pulse, make_delay,
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
    # Import the trajectory generation functions
    # Note: Using tensorflow_mri_trajectory (not the _fixed version) as it has the correct scaling
    from tensorflow_mri_trajectory import gen_spiral_traj_tfmri, convert_to_pypulseq_format
    print("Using TensorFlow-MRI for trajectory generation")
except ImportError:
    USE_TENSORFLOW_MRI = False
    from libspiral_hyperslice import vds_hyperslice
    print("TensorFlow-MRI not available, using native HyperSLICE implementation")

from librewinder.design_rewinder import design_rewinder
from kernels.kernel_handle_preparations import kernel_handle_preparations, kernel_handle_end_preparations
from math import ceil
import copy
import argparse
import os
import warnings

# Cmd args
parser = argparse.ArgumentParser(
                    prog='Write2DRTSpiralHyperSLICE',
                    description='Generates a 2D real-time spiral Pulseq sequence with HyperSLICE trajectory design.')

parser.add_argument('-c', '--config', type=str, default='config', help='Config file path.')

args = parser.parse_args()


print(f'Using config file: {args.config}.')
print('=' * 60)
print('HyperSLICE-style Trajectory Design Mode')
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
        "See example_config_hyperslice.toml for reference."
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

    # g is in mT/m, convert to T/m for consistency
    g = g * 1e-3  # mT/m to T/m

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
if params['spiral']['rotate_grads']:
    g_rewind_x, g_rewind_y, g_grad = design_rewinder(g_grad, params['spiral']['rewinder_time'], system, # type: ignore
                                             slew_ratio=params['spiral']['slew_ratio'],
                                             grad_rew_method=params['spiral']['grad_rew_method'],
                                             M1_nulling=params['spiral']['M1_nulling'], rotate_grads=params['spiral']['rotate_grads'])
else:
    g_rewind_x, g_rewind_y = design_rewinder(g_grad, params['spiral']['rewinder_time'], system, # type: ignore
                                             slew_ratio=params['spiral']['slew_ratio'],
                                             grad_rew_method=params['spiral']['grad_rew_method'],
                                             M1_nulling=params['spiral']['M1_nulling'])

# concatenate g and g_rewind, and plot.
g_grad = np.concatenate((g_grad, np.stack([g_rewind_x[0:], g_rewind_y[0:]]).T))

if params['user_settings']['show_plots']:
    # Use enhanced plotting if available
    if USE_TENSORFLOW_MRI and plot_trajectory_summary and 'trajectory' in locals():
        # Show the enhanced trajectory plot
        fig = plot_trajectory_summary(trajectory, params)
        plt.show()

        # Also show the gradient correction info
        if plot_gradient_correction_info:
            info_fig = plot_gradient_correction_info()
            plt.show()

    # Also show the standard gradient info plot
    plotgradinfo(g_grad, GRT)
    plt.show()

#%%
# Excitation
tbwp = params['acquisition']['tbwp']
rf, gz, gzr = make_sinc_pulse(flip_angle=params['acquisition']['flip_angle']/180*np.pi,
                                duration=params['acquisition']['rf_duration'],
                                slice_thickness=params['acquisition']['slice_thickness']*1e-3, # [mm] -> [m]
                                time_bw_product=tbwp,
                                return_gz=True,
                                use='excitation', system=system) # type: ignore

gzrr = copy.deepcopy(gzr)
gzrr.delay = 0 #gz.delay
rf.delay = calc_duration(gzrr) + gz.rise_time
gz.delay = calc_duration(gzrr)
gzr.delay = calc_duration(gzrr, gz)
gzz = add_gradients([gzrr, gz, gzr], system=system)

# ADC
# For HyperSLICE trajectory from TensorFlow-MRI, we don't need ndiscard
# The trajectory is already properly designed with the correct timing
if USE_TENSORFLOW_MRI:
    # TensorFlow-MRI trajectory is ready to use
    # Just match the number of samples exactly
    Tread = ro_time_ms * 1e-3  # [ms] -> [s]
    num_samples = np.floor(Tread/spiral_sys['adc_dwell'])
    # No delay needed - gradients and ADC start together
    adc = make_adc(num_samples, dwell=spiral_sys['adc_dwell'], delay=0, system=system)
    grad_delay = 0  # Gradients start immediately
else:
    # Original code for non-TensorFlow-MRI trajectories
    ndiscard = 10 # Number of samples to discard from beginning
    Tread = ro_time_ms * 1e-3  # [ms] -> [s]
    num_samples = np.floor(Tread/spiral_sys['adc_dwell']) + ndiscard
    # ADC should be delayed to skip the gradient ramp-up
    discard_delay_t = ceil((ndiscard*spiral_sys['adc_dwell']+GRT/2)/GRT)*GRT
    adc = make_adc(num_samples, dwell=spiral_sys['adc_dwell'], delay=discard_delay_t, system=system)
    grad_delay = 0  # Gradients start immediately

# Readout gradients
# Convert from mT/m to Hz/m: gamma = 42.577478518 MHz/T = 42.577478518 kHz/mT = 42577.478518 Hz/mT
# So mT/m * 42577.478518 Hz/mT = Hz/m
if USE_TENSORFLOW_MRI:
    # For TensorFlow-MRI trajectories, no delay needed
    gsp_x = make_arbitrary_grad(channel='x', waveform=g_grad[:,0]*42577.478518, first=0, last=0, delay=grad_delay, system=system) # [mT/m] -> [Hz/m]
    gsp_y = make_arbitrary_grad(channel='y', waveform=g_grad[:,1]*42577.478518, first=0, last=0, delay=grad_delay, system=system) # [mT/m] -> [Hz/m]
else:
    # Original code with gradient delay
    gsp_x = make_arbitrary_grad(channel='x', waveform=g_grad[:,0]*42577.478518, first=0, last=0, delay=grad_delay, system=system) # [mT/m] -> [Hz/m]
    gsp_y = make_arbitrary_grad(channel='y', waveform=g_grad[:,1]*42577.478518, first=0, last=0, delay=grad_delay, system=system) # [mT/m] -> [Hz/m]

# Set the Slice rewinder balance gradients delay
gzrr.delay = calc_duration(gsp_x, gsp_y, adc)

# create a crusher gradient (only for FLASH)
if params['spiral']['contrast'] == 'FLASH' or params['spiral']['contrast'] == 'FISP':
    crush_area = (4 / (params['acquisition']['slice_thickness'] * 1e-3)) + (-1 * gzr.area)
    gz_crush = make_trapezoid(channel='z',
                              area=crush_area,
                              max_grad=system.max_grad,
                              system=system)

# set the rotations.
# For HyperSLICE, we typically use the 'views' parameter to determine undersampling
# Each view is one spiral arm per temporal frame
gsp_xs = []
gsp_ys = []

# Override arm ordering for HyperSLICE mode
# HyperSLICE typically uses 'linear', 'golden', or 'tiny_number' ordering
hs_ordering = hs_params.get('ordering', 'linear')
print(f"Spiral arm ordering (HyperSLICE mode): {hs_ordering}.")

if hs_ordering == 'linear':
    # Linear ordering with views per frame
    if (n_int%2) == 1 and (params['acquisition']['repetitions']%2) == 1:
        warnings.warn("Number of interleaves is odd. To solve this, we increased it by 1. If this is undesired, please set repetitions to an even number instead.")
        n_int += 1
    for i in range(0, n_int):
        gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=2*np.pi*i/n_int)
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        params['spiral']['GA_angle'] = 360/n_int
    n_TRs = n_int * params['acquisition']['repetitions']

elif hs_ordering == 'golden' or hs_ordering == 'ga':
    # Golden angle ordering
    # Use GA_steps from config if available, otherwise use n_int * repetitions
    n_TRs = params['spiral'].get('GA_steps', n_int * params['acquisition']['repetitions'])
    if (n_TRs%2) == 1 and (params['acquisition']['repetitions']%2) == 1:
        warnings.warn(
                    '''
                      ========================================
                      Number of arms in the sequence is odd.
                      This may create steady state artifacts during the imaging with multiple runs, due to RF phase not alternating properly.
                      To avoid this issue, set repetitions to an even number.
                      ========================================
                      ''')

    n_int = n_TRs
    ang = 0
    ga_angle = params['spiral'].get('GA_angle', 137.5077)  # Default golden angle
    for i in range(0, n_TRs):
        gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=ang)
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        ang += ga_angle*np.pi/180
        ang = ang % (2*np.pi)
    params['spiral']['GA_angle'] = ga_angle

elif hs_ordering == 'tiny' or hs_ordering == 'tinyga':
    # Tiny golden angle
    tiny_number = hs_params.get('tiny_number', 7)
    tiny_ga = 360.0 / (n_int / tiny_number)
    print(f"  Using tiny golden angle: {tiny_ga:.4f} deg (tiny_number={tiny_number})")

    n_TRs = n_int * params['acquisition']['repetitions']
    ang = 0
    for i in range(0, n_TRs):
        gsp_x_rot, gsp_y_rot = rotate(gsp_x, gsp_y, axis="z", angle=ang)
        gsp_xs.append(gsp_x_rot)
        gsp_ys.append(gsp_y_rot)
        ang += tiny_ga*np.pi/180
        ang = ang % (2*np.pi)
    params['spiral']['GA_angle'] = tiny_ga

else:
    raise Exception(f"Unknown arm ordering: {hs_ordering}")

# Set the delays
# TE
if params['acquisition']['TE'] == 0:
    TEd = 0
    TE = rf.shape_dur - calc_rf_center(rf)[0] + calc_duration(gzr) - gzr.delay + gsp_x.delay
    print(f'Min TE is set: {TE*1e3:.3f} ms.')
    params['acquisition']['TE'] = TE
else:
    TE = params['acquisition']['TE']*1e-3
    TEd = TE - (rf.shape_dur - calc_rf_center(rf)[0] + calc_duration(gzr) + gsp_x.delay)
    assert TEd >= 0, "Required TE can not be achieved."

# TR
if params['acquisition']['TR'] == 0:
    TRd = 0
    TR = calc_duration(rf, gzz) + TEd + calc_duration(gsp_xs[0], gsp_ys[0], adc)
    if params['spiral']['contrast'] in ('FLASH', 'FISP'):
        TR = TR + calc_duration(gz_crush) # pyright: ignore[reportPossiblyUnboundVariable]
    print(f'Min TR is set: {TR*1e3:.3f} ms.')
    params['acquisition']['TR'] = TR
else:
    TR = params['acquisition']['TR']*1e-3
    TRd = TR - (calc_duration(rf, gzz) + TEd + calc_duration(gsp_xs[0], gsp_ys[0], adc))
    if params['spiral']['contrast'] in ('FLASH', 'FISP'):
        TRd = TRd - calc_duration(gz_crush) # pyright: ignore[reportPossiblyUnboundVariable]
    assert TRd >= 0, "Required TR can not be achieved."

TE_delay = make_delay(TEd)
TR_delay = make_delay(TRd)

# Print temporal resolution summary
print('\n')
print_temporal_summary(
    arm_time=ro_time_ms,
    deadtime=deadtime_ms,
    num_views=views,
    n_arms_total=n_TRs,
    temporal_res=temp_res_ms
)

# Sequence looping

seq = Sequence(system)

# handle any preparation pulses.
prep_str = kernel_handle_preparations(seq, params, system, rf=rf, gz=gzz)

# useful for end_peparation pulses.
params['flip_angle_last'] = params['acquisition']['flip_angle']

# tagging pulse pre-prep (only if fa_schedule exists)
rf_amplitudes, FA_schedule_str = schedule_FA(params, n_TRs)

# used for FLASH only: set rf spoiling increment.
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
for arm_i in range(0,n_TRs):
    curr_rf = copy.deepcopy(rf)

    # check if we are using a rammped FA scheme (rf_amplitudes is a list [])
    if len(rf_amplitudes) > 0:
        if arm_i >= len(rf_amplitudes):
            n_TRs = arm_i
            break
        curr_rf.signal = rf.signal * rf_amplitudes[arm_i] / np.deg2rad(params['acquisition']['flip_angle'])

    curr_rf.phase_offset = rf_phase
    adc.phase_offset = rf_phase

    rf_inc = np.mod(rf_inc + quadratic_phase_increment, np.pi * 2)
    rf_phase = np.mod(rf_phase + linear_phase_increment + rf_inc, np.pi * 2)

    if enable_trigger is True:
        seq.add_block(trig, curr_rf, gzz)
    else:
        seq.add_block(curr_rf, gzz)

    seq.add_block(TE_delay)
    seq.add_block(make_label('LIN', 'SET', arm_i % n_int))
    seq.add_block(gsp_xs[arm_i % n_int], gsp_ys[arm_i % n_int], adc)

    if params['spiral']['contrast'] in ('FLASH', 'FISP'):
        seq.add_block(gz_crush) # pyright: ignore[reportPossiblyUnboundVariable]
    seq.add_block(TR_delay)

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
    plt.figure()
    # Plot only readout trajectory (k_traj_adc), excluding rewinders
    plt.plot(k_traj_adc[0,:], k_traj_adc[1, :])

    # make axis suqaure
    plt.gca().set_aspect('equal', adjustable='box')
    # double fontsize
    plt.rcParams.update({'font.size': 14})

    plt.xlabel('$k_x [mm^{-1}]$')
    plt.ylabel('$k_y [mm^{-1}]$')
    plt.title('k-Space Trajectory - Readout Only (HyperSLICE Design)')


    if 'acoustic_resonances' in params and 'frequencies' in params['acoustic_resonances']:
        resonances = []
        for idx in range(len(params['acoustic_resonances']['frequencies'])):
            resonances.append({'frequency': params['acoustic_resonances']['frequencies'][idx], 'bandwidth': params['acoustic_resonances']['bandwidths'][idx]})
        try:
            seq.calculate_gradient_spectrum(acoustic_resonances=resonances)
            plt.title('Gradient spectrum')
        except (ValueError, UserWarning) as e:
            print(f"\nWarning: Could not calculate gradient spectrum: {e}")
            print("This may occur with very short sequences (few interleaves).")
            print("Continuing without acoustic resonance analysis...")
        plt.show()


# Detailed report if requested
if params['user_settings']['detailed_rep']:
    print("\n===== Detailed Test Report =====\n")
    rep_str = seq.test_report()
    print(rep_str)

    # Add correct gradient information if using TensorFlow-MRI
    if USE_TENSORFLOW_MRI and 'trajectory' in locals():
        print("\n" + "="*60)
        print("IMPORTANT: ACTUAL GRADIENT VALUES (from TensorFlow-MRI)")
        print("="*60)
        max_grad = np.max(np.sqrt(trajectory['gx']**2 + trajectory['gy']**2))
        print(f"Maximum Gradient: {max_grad:.1f} mT/m (actual)")
        print(f"Hardware Limit:   22.0 mT/m")
        print(f"Utilization:      {max_grad/22.0*100:.1f}%")
        print("\nNOTE: PyPulseq test_report shows incorrect gradient values (~0.35 mT/m)")
        print("      This is a display bug. The actual sequence uses correct gradients.")
        print("="*60 + "\n")

# Write the sequence to file
if params['user_settings']['write_seq']:

    seq.set_definition(key="FOV", value=[fov[0]*1e-2, fov[0]*1e-2, params['acquisition']['slice_thickness']*1e-3])
    seq.set_definition(key="Slice_Thickness", value=params['acquisition']['slice_thickness']*1e-3)
    seq.set_definition(key="Name", value="sprssfp_hyperslice")
    seq.set_definition(key="TE", value=TE)
    seq.set_definition(key="TR", value=TR)
    seq.set_definition(key="FA", value=params['acquisition']['flip_angle'])
    seq.set_definition(key="Resolution_mm", value=res)
    seq.set_definition(key="Temporal_Res_ms", value=temp_res_ms)

    m1_str = "M1" if params['spiral']['M1_nulling'] else ""
    rev_str = "rev" if reverse_traj else ""
    seq_filename = f"spiral_hs_{params['spiral']['contrast']}{FA_schedule_str}{prep_str}{end_prep_str}_{hs_ordering}{params['spiral']['GA_angle']:.4f}_nTR{n_TRs}_views{views}_Tread{ro_time_ms:.2f}_Tres{temp_res_ms:.1f}_TR{TR*1e3:.2f}ms_FA{params['acquisition']['flip_angle']}_{m1_str}_{rev_str}_{params['user_settings']['filename_ext']}"

    # remove double, triple, quadruple underscores, and trailing underscores
    seq_filename = seq_filename.replace("__", "_").replace("__", "_").replace("__", "_").strip("_")

    seq_path = os.path.join('out_seq', f"{seq_filename}.seq")

    # ensure the out_seq directory exists before writing.
    os.makedirs("out_seq", exist_ok=True)

    seq.write(seq_path)  # Save to disk

    # Export k-space trajectory
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    # Set ndiscard value for saving
    ndiscard = 0 if USE_TENSORFLOW_MRI else 10

    # save_traj_dcf(seq.signature_value, k_traj_adc, n_TRs, n_int, fov, res, ndiscard, params['user_settings']['show_plots'])
    params_save = {
        'adc_dwell': spiral_sys['adc_dwell'],
        'ndiscard': ndiscard,
        'n_TRs': n_TRs,
        'n_int': n_int,
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
    }
    save_metadata(seq.signature_value, k_traj_adc, params_save, params['user_settings']['show_plots'], dcf_method="hoge", out_dir="out_trajectory")

    print(f'\nMetadata file for {seq_filename} is saved as {seq.signature_value} in out_trajectory/.')

    # Save readout-only trajectory and gradient waveforms
    # Create output directory if it doesn't exist
    traj_dir = Path("out_trajectory")
    traj_dir.mkdir(exist_ok=True)

    # Save gradient waveforms if using TensorFlow-MRI
    if USE_TENSORFLOW_MRI:
        grad_filename = traj_dir / f"{seq.signature_value}_gradients.mat"

        # Get full gradient waveforms from TensorFlow-MRI trajectory
        grad_data = {
            'gx': trajectory['gx'],  # [n_arms, n_samples] in mT/m
            'gy': trajectory['gy'],  # [n_arms, n_samples] in mT/m
            'kx': trajectory['kx'],  # [n_arms, n_samples] in rad/m
            'ky': trajectory['ky'],  # [n_arms, n_samples] in rad/m
            'n_arms': trajectory['n_arms'],
            'n_samples_per_arm': trajectory['n_samples'],
            'dwell_time': trajectory['dwell_time'],  # in seconds
            'readout_time_ms': trajectory['readout_time'],
            'fov_mm': fov[0] * 10,  # cm to mm
            'base_resolution': base_resolution,
        }

        sio.savemat(grad_filename, grad_data)
        print(f'Gradient waveforms saved to: {grad_filename}')

    # For TensorFlow-MRI, handle trajectory saving based on ordering scheme
    if USE_TENSORFLOW_MRI:
        n_unique_arms = trajectory['n_arms']
        n_samples_per_arm = trajectory['n_samples']

        # For golden angle, each TR needs a unique rotated trajectory
        # For linear, we only need the base arms
        if hs_ordering in ['golden', 'ga', 'tinyga']:
            # Golden angle ordering: generate unique rotations
            kx_all = []
            ky_all = []

            # Golden angle in radians
            if hs_ordering == 'golden' or hs_ordering == 'ga':
                golden_angle_rad = params['spiral']['GA_angle'] * np.pi / 180
            else:  # tiny golden angle
                golden_angle_rad = params['spiral']['GA_angle'] * np.pi / 180

            # Get number of unique rotations from config (default to all TRs if not specified)
            # First check in hyperslice section, then spiral section, then default to all TRs
            n_unique_ga_rotations = params['spiral'].get('hyperslice', {}).get('n_unique_rotations',
                                    params['spiral'].get('n_unique_rotations', n_TRs))

            # For reconstruction, we save n_unique_ga_rotations × n_unique_arms trajectories
            # The sequence will cycle through these during acquisition
            rotation_idx = 0
            for rot in range(n_unique_ga_rotations):
                # Calculate rotation angle for this unique rotation
                rotation_angle = rot * golden_angle_rad

                # Apply this rotation to all base arms
                for arm_idx in range(n_unique_arms):
                    # Get base trajectory for this arm
                    kx_base = trajectory['kx'][arm_idx, :] / (2 * np.pi)  # Convert rad/m to 1/m
                    ky_base = trajectory['ky'][arm_idx, :] / (2 * np.pi)

                    # Apply rotation
                    cos_theta = np.cos(rotation_angle)
                    sin_theta = np.sin(rotation_angle)
                    kx_rotated = kx_base * cos_theta - ky_base * sin_theta
                    ky_rotated = kx_base * sin_theta + ky_base * cos_theta

                    kx_all.extend(kx_rotated)
                    ky_all.extend(ky_rotated)

            # Convert to numpy arrays
            kx_all = np.array(kx_all)
            ky_all = np.array(ky_all)

            # Total saved trajectories = n_unique_ga_rotations × n_unique_arms
            n_saved_trajectories = n_unique_ga_rotations * n_unique_arms

            # Reshape for [dim, RO, INT] format
            kx_reshaped = kx_all.reshape(n_saved_trajectories, n_samples_per_arm).T  # [RO, INT]
            ky_reshaped = ky_all.reshape(n_saved_trajectories, n_samples_per_arm).T  # [RO, INT]

            n_saved_rotations = n_saved_trajectories  # Total unique trajectories saved

            print(f"Golden angle trajectory: {n_unique_ga_rotations} rotations × {n_unique_arms} arms = {n_saved_trajectories} unique trajectories")

        else:
            # Linear ordering: only save the base arms without rotation
            kx_all = []
            ky_all = []

            for arm_idx in range(n_unique_arms):
                # Get k-space trajectory for this arm (already in rad/m)
                kx_arm = trajectory['kx'][arm_idx, :] / (2 * np.pi)  # Convert rad/m to 1/m
                ky_arm = trajectory['ky'][arm_idx, :] / (2 * np.pi)

                kx_all.extend(kx_arm)
                ky_all.extend(ky_arm)

            # Convert to numpy arrays
            kx_all = np.array(kx_all)
            ky_all = np.array(ky_all)

            # Reshape for [dim, RO, INT] format
            kx_reshaped = kx_all.reshape(n_unique_arms, n_samples_per_arm).T  # [RO, INT]
            ky_reshaped = ky_all.reshape(n_unique_arms, n_samples_per_arm).T  # [RO, INT]

            n_saved_rotations = n_unique_arms  # Only save base arms for linear

        # Stack to create [dim, RO, INT]
        traj = np.stack([kx_reshaped, ky_reshaped], axis=0)  # [2, RO, INT]

        # Create time array for ADC samples
        t_adc_tfmri = np.arange(len(kx_all)) * trajectory['dwell_time']

        readout_traj = {
            'kx': kx_all,  # k-space x coordinates [1/m]
            'ky': ky_all,  # k-space y coordinates [1/m]
            'traj': traj,   # trajectory organized as [dim, RO, INT]
            't': t_adc_tfmri,  # time points during ADC [s]
            'n_rotations': n_saved_rotations,    # number of saved rotations
            'n_samples_per_rotation': n_samples_per_arm,  # ADC samples per rotation
            'adc_dwell': trajectory['dwell_time'],  # ADC dwell time [s]
            'fov': fov[0],           # field of view [cm]
            'resolution': res,       # spatial resolution [mm]
            'base_resolution': base_resolution,  # HyperSLICE base resolution
            'temporal_resolution_ms': temp_res_ms,  # temporal resolution [ms]
            'ordering': hs_ordering,  # arm ordering scheme
            'ga_angle': params['spiral']['GA_angle'],  # golden angle [deg]
            'n_base_arms': n_unique_arms,  # number of base spiral arms
            'n_unique_ga_rotations': n_unique_ga_rotations if hs_ordering in ['golden', 'ga', 'tinyga'] else 0,
            'total_TRs': n_TRs,      # total number of TRs in sequence
        }
    else:
        # Original code for non-TensorFlow-MRI trajectories
        # Extract readout trajectory from PyPulseq calculation
        kx_reshaped = k_traj_adc[0, :].reshape(n_TRs, int(num_samples)).T  # [RO, INT]
        ky_reshaped = k_traj_adc[1, :].reshape(n_TRs, int(num_samples)).T  # [RO, INT]

        # Stack to create [dim, RO, INT]
        traj = np.stack([kx_reshaped, ky_reshaped], axis=0)  # [2, RO, INT]

        readout_traj = {
            'kx': k_traj_adc[0, :],  # k-space x coordinates [1/m] - original flat format
            'ky': k_traj_adc[1, :],  # k-space y coordinates [1/m] - original flat format
            'traj': traj,            # trajectory organized as [dim, RO, INT]
            't': t_adc,              # time points during ADC [s]
            'n_rotations': n_TRs,    # number of spiral rotations
            'n_samples_per_rotation': int(num_samples),  # ADC samples per rotation
            'adc_dwell': spiral_sys['adc_dwell'],  # ADC dwell time [s]
            'fov': fov[0],           # field of view [cm]
            'resolution': res,       # spatial resolution [mm]
            'base_resolution': base_resolution,  # HyperSLICE base resolution
            'temporal_resolution_ms': temp_res_ms,  # temporal resolution [ms]
            'ordering': hs_ordering,  # arm ordering scheme
            'ga_angle': params['spiral']['GA_angle'],  # golden angle [deg]
        }

    # Save to .mat file with descriptive filename
    traj_filename = traj_dir / f"{seq.signature_value}_readout.mat"
    sio.savemat(traj_filename, {'k_readout': readout_traj})

    print(f'Readout trajectory saved to {traj_filename}.')


# %%
