"""
TensorFlow-MRI based spiral trajectory generation for HyperSLICE.

Based on: https://github.com/mrphys/HyperSLICE/blob/master/utils/preprocessing_trajectory_gen.py
This module provides a wrapper to generate spiral trajectories using TensorFlow-MRI,
matching the HyperSLICE implementation.
"""

import warnings
warnings.filterwarnings('ignore')
import os
# Set before importing TensorFlow
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

# Import TensorFlow with logging suppressed
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Now import tensorflow_mri
from tensorflow_mri import spiral_trajectory


def config_optimized_traj():
    """
    Configuration for optimized trajectory from HyperSLICE paper.

    These are the exact parameters from:
    https://github.com/mrphys/HyperSLICE/blob/master/utils/preprocessing_trajectory_gen.py
    """
    return {
        'base_resolution': 240,
        'field_of_view': 400,  # mm
        'phases': 12,
        'vd_spiral_arms': 16,
        'vd_inner_cutoff': 0.15,
        'pre_vd_outer_cutoff': 0.41288,
        'vd_outer_density': 0.07,
        'vd_type': 'hanning',
        'max_grad_ampl': 22.0,  # mT/m
        'min_rise_time': 10.0,  # us/mT/m (inverse of slew rate)
        'dwell_time': 1.4,  # us
        'gradient_delay': 0.56,  # us
        'readoutOS': 2.0,
        'deadtime': 2.0,  # ms
    }


def gen_spiral_traj_tfmri(
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
    deadtime=2.0,  # ms
    phases=None,
    **kwargs
):
    """
    Generate spiral trajectory using TensorFlow-MRI.

    This generates UNDERSAMPLED spiral arms where each arm covers 1/vd_spiral_arms
    of k-space. The arms are meant to be rotated and acquired sequentially.

    Returns
    -------
    trajectory : dict
        Dictionary containing trajectory data for all arms
    """

    # Calculate variable density parameters
    vd_outer_cutoff = vd_inner_cutoff + 0.1 + pre_vd_outer_cutoff * (1 - vd_inner_cutoff - 0.1)

    # Generate all rotated arms using views parameter
    trajectory_tensor = spiral_trajectory(
        base_resolution=base_resolution,
        spiral_arms=vd_spiral_arms,
        field_of_view=field_of_view,  # mm
        max_grad_ampl=max_grad_ampl,  # mT/m
        min_rise_time=min_rise_time,  # us/mT/m
        dwell_time=dwell_time,  # us
        views=vd_spiral_arms,  # Generate all rotated views
        readout_os=readoutOS,
        gradient_delay=gradient_delay,  # us
        vd_inner_cutoff=vd_inner_cutoff,
        vd_outer_cutoff=vd_outer_cutoff,
        vd_outer_density=vd_outer_density,
        vd_type=vd_type
    )

    # Extract trajectory - shape is [views, n_samples, 2]
    trajectory_np = trajectory_tensor.numpy()

    if len(trajectory_np.shape) != 3:
        raise ValueError(f"Expected 3D trajectory, got shape {trajectory_np.shape}")

    n_arms = trajectory_np.shape[0]
    n_samples = trajectory_np.shape[1]

    # Extract all arms
    kx = np.zeros((n_arms, n_samples))
    ky = np.zeros((n_arms, n_samples))

    for arm in range(n_arms):
        kx[arm, :] = trajectory_np[arm, :, 0]
        ky[arm, :] = trajectory_np[arm, :, 1]

    # The trajectory from TensorFlow-MRI is in normalized units
    # We need to properly scale it
    # Based on testing, the trajectory reaches k_max ≈ 3.12 in these units
    # This corresponds to vd_outer_cutoff * Nyquist frequency

    # Find actual k_max in the trajectory
    k_mag = np.sqrt(kx**2 + ky**2)
    k_max_actual = np.max(k_mag)

    # Expected k_max in cycles/FOV
    nyquist_k = base_resolution / 2  # 120 for base_resolution=240
    k_max_expected_cycles_fov = nyquist_k * vd_outer_cutoff

    # Scale to cycles/FOV
    scale_to_cycles_fov = k_max_expected_cycles_fov / k_max_actual if k_max_actual > 0 else 1.0
    kx = kx * scale_to_cycles_fov
    ky = ky * scale_to_cycles_fov

    # Convert to rad/m
    fov_m = field_of_view * 1e-3  # mm to m
    k_scale = 2 * np.pi / fov_m
    kx_rad_m = kx * k_scale
    ky_rad_m = ky * k_scale

    # Calculate gradients
    dt = dwell_time * 1e-6  # us to s
    gamma = 2 * np.pi * 42.577478518e6  # rad/s/T

    gx = np.zeros_like(kx_rad_m)
    gy = np.zeros_like(ky_rad_m)

    for arm in range(n_arms):
        dkx_dt = np.gradient(kx_rad_m[arm, :], dt)
        dky_dt = np.gradient(ky_rad_m[arm, :], dt)
        gx[arm, :] = dkx_dt / gamma * 1e3  # T/m to mT/m
        gy[arm, :] = dky_dt / gamma * 1e3

    # Calculate readout time
    readout_time = n_samples * dt * 1e3  # to ms

    # Create rotation angles for proper arm distribution
    # TensorFlow-MRI doesn't seem to rotate properly, so we'll do it manually
    angles = np.linspace(0, 2*np.pi, n_arms, endpoint=False)

    # Apply rotations to ensure proper angular distribution
    # Use the first arm as template and rotate it for all other arms
    kx_rot = np.zeros_like(kx_rad_m)
    ky_rot = np.zeros_like(ky_rad_m)
    gx_rot = np.zeros_like(gx)
    gy_rot = np.zeros_like(gy)

    # First check if all arms are identical (TensorFlow-MRI issue)
    all_same = True
    for arm in range(1, n_arms):
        if not np.allclose(kx[arm, :10], kx[0, :10]):
            all_same = False
            break

    if all_same:
        # All arms are the same, so we need to rotate them manually
        for arm in range(n_arms):
            angle = 2 * np.pi * arm / n_arms
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Rotate the first arm
            kx_rot[arm, :] = cos_a * kx_rad_m[0, :] - sin_a * ky_rad_m[0, :]
            ky_rot[arm, :] = sin_a * kx_rad_m[0, :] + cos_a * ky_rad_m[0, :]
            gx_rot[arm, :] = cos_a * gx[0, :] - sin_a * gy[0, :]
            gy_rot[arm, :] = sin_a * gx[0, :] + cos_a * gy[0, :]
    else:
        # Arms are already different, use them as-is
        kx_rot = kx_rad_m
        ky_rot = ky_rad_m
        gx_rot = gx
        gy_rot = gy

    trajectory = {
        'kx': kx_rot,  # [n_arms, n_samples] in rad/m
        'ky': ky_rot,  # [n_arms, n_samples] in rad/m
        'gx': gx_rot,  # [n_arms, n_samples] in mT/m
        'gy': gy_rot,  # [n_arms, n_samples] in mT/m
        'readout_time': readout_time,  # ms
        'n_samples': n_samples,
        'n_arms': n_arms,
        'dwell_time': dt,  # s
        'fov': fov_m,  # m
        'base_resolution': base_resolution,
        'vd_inner_cutoff': vd_inner_cutoff,
        'vd_outer_cutoff': vd_outer_cutoff,
        'vd_outer_density': vd_outer_density,
    }

    return trajectory


def convert_to_pypulseq_format(trajectory):
    """Convert to PyPulseq format."""

    # Extract first spiral arm
    kx = trajectory['kx'][0, :]  # rad/m
    ky = trajectory['ky'][0, :]  # rad/m
    gx = trajectory['gx'][0, :]  # mT/m
    gy = trajectory['gy'][0, :]  # mT/m

    # Convert k-space from rad/m to m^-1
    k = np.stack([kx / (2 * np.pi), ky / (2 * np.pi)])  # [2, n_samples]

    # Stack gradients
    g = np.column_stack([gx, gy])  # [n_samples, 2]

    # Calculate slew rates
    dt = trajectory['dwell_time']
    dgx_dt = np.gradient(gx, dt)
    dgy_dt = np.gradient(gy, dt)
    s = np.column_stack([dgx_dt, dgy_dt]) * 1e-3  # mT/m/s to T/m/s

    # Time vector
    t = np.arange(trajectory['n_samples']) * dt

    return k, g, s, t


if __name__ == "__main__":
    """Test the trajectory generation."""

    print("Testing TensorFlow-MRI based HyperSLICE trajectory generation\n")
    print("=" * 60)

    # Use optimized parameters
    params = config_optimized_traj()

    print("Parameters:")
    for key, value in params.items():
        print(f"  {key:20s}: {value}")
    print()

    # Generate trajectory
    print("Generating trajectory...")
    trajectory = gen_spiral_traj_tfmri(**params)

    print(f"\nTrajectory results:")
    print(f"  Number of arms:     {trajectory['n_arms']}")
    print(f"  Samples per arm:    {trajectory['n_samples']}")
    print(f"  Readout time:       {trajectory['readout_time']:.3f} ms")

    # Check gradients
    g_mag = np.sqrt(trajectory['gx']**2 + trajectory['gy']**2)
    print(f"\nGradient analysis:")
    print(f"  Max gradient:       {np.max(g_mag):.2f} mT/m")
    print(f"  Mean gradient:      {np.mean(g_mag):.2f} mT/m")

    # Check rotation - look at a point away from origin
    idx = 10  # Check 10th point to avoid origin
    angle0 = np.arctan2(trajectory['ky'][0, idx], trajectory['kx'][0, idx])
    angle1 = np.arctan2(trajectory['ky'][1, idx], trajectory['kx'][1, idx])
    angle_diff = (angle1 - angle0) * 180 / np.pi
    if angle_diff < 0:
        angle_diff += 360
    print(f"\nArm rotation:")
    print(f"  Angle difference:   {angle_diff:.1f} degrees")
    print(f"  Expected:           {360/16:.1f} degrees")