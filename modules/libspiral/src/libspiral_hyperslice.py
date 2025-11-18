"""
HyperSLICE-style spiral trajectory design module.

This module provides trajectory design functions using HyperSLICE parameterization
while maintaining compatibility with the existing rtspiral_pypulseq framework.
Unlike HyperSLICE's TensorFlow implementation, this is a pure Python translation
that leverages existing C-based trajectory generation (vds_design).

Key differences from standard libspiral:
- VD parameters use cutoff radii and density factors instead of FOV coefficients
- Temporal resolution optimization (iterates spiral arms to fit time window)
- No TensorFlow dependency

Author: Adapted from HyperSLICE (mrphys/HyperSLICE)
Date: 2025
"""

from typing import Literal, Optional, Tuple
import numpy as np
import numpy.typing as npt
from libspiral import vds_design
from math import sqrt


def hyperslice_vd_to_fov(
    vd_inner_cutoff: float,
    pre_vd_outer_cutoff: float,
    vd_outer_density: float,
    num_fov_coeffs: int = 3
) -> list[float]:
    """
    Convert HyperSLICE VD parameters to FOV coefficient list for vds_design().

    HyperSLICE uses a more intuitive parameterization:
    - vd_inner_cutoff: Normalized radius (0-1) where undersampling starts
    - vd_outer_cutoff: Normalized radius where max undersampling is reached
    - vd_outer_density: Final spacing multiplier at edge (e.g., 0.07 = 7% of Nyquist)

    This function maps those to FOV coefficients that achieve similar density weighting.

    Parameters
    ----------
    vd_inner_cutoff : float
        Radius where undersampling starts (0-1 normalized)
    pre_vd_outer_cutoff : float
        Offset parameter for outer cutoff calculation
    vd_outer_density : float
        Final undersampling density at edge (0-1, lower = more undersampling)
    num_fov_coeffs : int, optional
        Number of FOV coefficients to generate (default: 3)

    Returns
    -------
    fov_coefficients : list[float]
        List of FOV coefficients for vds_design()

    Notes
    -----
    The outer cutoff is calculated as:
        vd_outer_cutoff = vd_inner_cutoff + 0.1 + pre_vd_outer_cutoff * (1 - vd_inner_cutoff - 0.1)

    This mapping is approximate and may require empirical tuning for exact matching.
    """

    # Handle edge case
    if vd_inner_cutoff > 0.9:
        # Fully sampled case
        return [1.0] * num_fov_coeffs

    # Calculate outer cutoff radius (HyperSLICE formula)
    vd_outer_cutoff = vd_inner_cutoff + 0.1 + pre_vd_outer_cutoff * (1 - vd_inner_cutoff - 0.1)

    # Map to FOV coefficients
    # The FOV coefficients control density weighting in Hargreaves' VDS design
    # Higher FOV values = tighter spacing in those regions
    # This is an empirical mapping that approximates HyperSLICE behavior

    # Create linearly spaced FOV coefficients weighted by density profile
    fov_coeffs = []
    for i in range(num_fov_coeffs):
        # Normalized position in k-space (0 to 1)
        k_norm = i / (num_fov_coeffs - 1) if num_fov_coeffs > 1 else 0.5

        if k_norm <= vd_inner_cutoff:
            # Fully sampled inner region
            fov_coeffs.append(1.0)
        elif k_norm >= vd_outer_cutoff:
            # Maximum undersampling outer region
            fov_coeffs.append(vd_outer_density)
        else:
            # Transition region - linear interpolation
            # (Could be made quadratic or hanning based on vd_type)
            transition_frac = (k_norm - vd_inner_cutoff) / (vd_outer_cutoff - vd_inner_cutoff)
            fov_val = 1.0 - transition_frac * (1.0 - vd_outer_density)
            fov_coeffs.append(fov_val)

    return fov_coeffs


def estimate_readout_time(
    sys: dict,
    fov: list,
    res: float,
    n_arms: int,
    vd_inner_cutoff: float,
    pre_vd_outer_cutoff: float,
    vd_outer_density: float
) -> float:
    """
    Estimate readout time for a given spiral arm count.

    This is a fast estimation without full trajectory design.
    Uses a heuristic based on k-space extent and gradient limits.

    Parameters
    ----------
    sys : dict
        System parameters (max_grad, max_slew, adc_dwell, etc.)
    fov : list
        FOV coefficients [cm]
    res : float
        Resolution [mm]
    n_arms : int
        Number of spiral arms
    vd_inner_cutoff : float
        Inner cutoff for VD
    pre_vd_outer_cutoff : float
        Pre-outer cutoff parameter
    vd_outer_density : float
        Outer density factor

    Returns
    -------
    readout_time : float
        Estimated readout time [s]
    """

    # Convert FOV coefficients
    fov_coeffs = hyperslice_vd_to_fov(vd_inner_cutoff, pre_vd_outer_cutoff, vd_outer_density)

    # Quick estimate using radial distance and average slew
    krmax = 1 / (2 * res * 1e-3)  # [m^-1]

    # Average FOV coefficient
    avg_fov = np.mean(fov_coeffs)

    # Estimate time based on k-space extent and slew rate
    # This is a simplified model - actual time from vds_design may differ
    gamma = 42.58e6  # [Hz/T]
    max_slew = sys['max_slew']  # [T/m/s]

    # Approximate readout time (empirical formula)
    # More arms = shorter trajectory per arm
    readout_time = (krmax / (gamma * max_slew)) * (1 / avg_fov) * sqrt(n_arms)

    return readout_time


def vds_hyperslice(
    sys: dict,
    base_resolution: int,
    fov: list,
    vd_spiral_arms: int,
    vd_inner_cutoff: float = 0.15,
    pre_vd_outer_cutoff: float = 0.41288,
    vd_outer_density: float = 0.07,
    vd_type: Literal['linear', 'quadratic', 'hanning'] = 'linear',
    min_max_arm_time: list[float] = [0.88, 1.67],
    max_tempres: float = 55.0,
    deadtime: float = 2.0,
    max_guesses: int = 50,
    optimize_arms: bool = True
) -> Tuple[
    Optional[npt.NDArray],
    Optional[npt.NDArray],
    Optional[npt.NDArray],
    Optional[npt.NDArray],
    int,
    int,
    float,
    float
]:
    """
    Design VDS trajectory using HyperSLICE parameterization.

    This function mimics HyperSLICE's trajectory generation approach:
    1. Iteratively adjusts spiral arm count to meet timing constraints (min_max_arm_time)
    2. Calculates temporal resolution based on TR = arm_time + deadtime
    3. Determines number of views to fit within max_tempres

    Unlike HyperSLICE's TensorFlow implementation, this uses the existing
    vds_design() C-based trajectory generation.

    Parameters
    ----------
    sys : dict
        System parameters: max_grad, max_slew, adc_dwell, os
    base_resolution : int
        Base resolution (e.g., 240 for 240x240 matrix)
    fov : list
        Field of view [cm]
    vd_spiral_arms : int
        Initial number of spiral arms to try
    vd_inner_cutoff : float, optional
        Normalized radius where undersampling starts (default: 0.15)
    pre_vd_outer_cutoff : float, optional
        Offset for outer cutoff calculation (default: 0.41288)
    vd_outer_density : float, optional
        Final density at edge (default: 0.07 = 7% of Nyquist)
    vd_type : str, optional
        VD transition type: 'linear', 'quadratic', 'hanning' (default: 'linear')
        Note: Currently only 'linear' is fully implemented in FOV mapping
    min_max_arm_time : list[float], optional
        [min, max] arm duration in ms (default: [0.88, 1.67])
    max_tempres : float, optional
        Maximum temporal resolution in ms (default: 55.0)
    deadtime : float, optional
        Dead time between readouts in ms (default: 2.0)
    max_guesses : int, optional
        Maximum iterations to find valid arm count (default: 50)
    optimize_arms : bool, optional
        If True, optimize spiral arm count to fit timing window.
        If False, use vd_spiral_arms exactly as specified (default: True)

    Returns
    -------
    k : NDArray or None
        k-space trajectory [m^-1]
    g : NDArray or None
        Gradient waveform [mT/m]
    s : NDArray or None
        Slew rate [T/m/s]
    time : NDArray or None
        Time axis [s]
    vd_spiral_arms : int
        Final number of spiral arms
    views : int
        Number of views (arms per temporal frame)
    readout_time : float
        Actual readout time [ms]
    temporal_res : float
        Actual temporal resolution [ms]

    Notes
    -----
    - If no valid arm count is found within max_guesses, returns None values
    - The algorithm prioritizes meeting timing constraints over exact resolution
    - Temporal resolution = TR * views, where TR = readout_time + deadtime
    """

    # Calculate resolution from base_resolution and FOV
    fov_m = fov[0] * 1e-2  # [cm] -> [m]
    res = (fov_m / base_resolution) * 1e3  # [m] -> [mm]

    # Convert timing from ms to s
    min_arm_time_s = min_max_arm_time[0] * 1e-3
    max_arm_time_s = min_max_arm_time[1] * 1e-3
    deadtime_s = deadtime * 1e-3
    max_tempres_s = max_tempres * 1e-3

    # Translate VD params to FOV coefficients
    fov_coeffs = hyperslice_vd_to_fov(vd_inner_cutoff, pre_vd_outer_cutoff, vd_outer_density)

    # Iterate to find arm count that meets timing constraints
    counter = 0
    readout_time = 0.0
    vd_spiral_arms_current = vd_spiral_arms
    k, g, s, time = None, None, None, None

    if optimize_arms:
        print(f"Optimizing spiral arm count to fit time window [{min_max_arm_time[0]:.2f}, {min_max_arm_time[1]:.2f}] ms...")
    else:
        print(f"Using fixed spiral arm count: {vd_spiral_arms}")

    while counter < max_guesses and vd_spiral_arms_current > 0:
        # Design trajectory with current arm count
        try:
            k, g, s, time = vds_design(sys, vd_spiral_arms_current, fov_coeffs, res, max_arm_time_s)
            readout_time = time[-1]  # [s]

            # Check if timing constraint is met
            if not optimize_arms:
                # Skip optimization - use fixed arm count
                print(f"  Designed trajectory: {vd_spiral_arms_current} arms -> {readout_time * 1e3:.3f} ms")
                break
            elif min_arm_time_s < readout_time < max_arm_time_s:
                print(f"  Iteration {counter + 1}: {vd_spiral_arms_current} arms -> {readout_time * 1e3:.3f} ms ✓")
                break
            else:
                # Adjust arm count
                if readout_time < min_arm_time_s:
                    print(f"  Iteration {counter + 1}: {vd_spiral_arms_current} arms -> {readout_time * 1e3:.3f} ms (too short)")
                    vd_spiral_arms_current -= 1
                else:  # readout_time > max_arm_time_s
                    print(f"  Iteration {counter + 1}: {vd_spiral_arms_current} arms -> {readout_time * 1e3:.3f} ms (too long)")
                    vd_spiral_arms_current += 1

        except Exception as e:
            print(f"  Error during trajectory design: {e}")
            break

        counter += 1

    # Check if valid solution found
    if k is None:
        print(f"  Failed to design trajectory.")
        return None, None, None, None, vd_spiral_arms, 0, 0.0, 0.0

    if optimize_arms and not (min_arm_time_s < readout_time < max_arm_time_s):
        print(f"  Failed to find valid arm count after {counter} iterations.")
        return None, None, None, None, vd_spiral_arms, 0, 0.0, 0.0

    # Calculate temporal resolution
    TR = readout_time + deadtime_s  # [s]
    views = int(max_tempres_s / TR)
    temporal_res = TR * views  # [s]

    print(f"\nFinal trajectory parameters:")
    print(f"  Spiral arms: {vd_spiral_arms_current}")
    print(f"  Readout time: {readout_time * 1e3:.3f} ms")
    print(f"  TR: {TR * 1e3:.3f} ms")
    print(f"  Views per frame: {views}")
    print(f"  Temporal resolution: {temporal_res * 1e3:.3f} ms")

    return (
        k,
        g,
        s,
        time,
        vd_spiral_arms_current,
        views,
        readout_time * 1e3,  # Convert to ms
        temporal_res * 1e3   # Convert to ms
    )


if __name__ == "__main__":
    """Test HyperSLICE-style trajectory design"""

    print("Testing HyperSLICE-style VDS design\n")
    print("=" * 60)

    # System parameters
    sys = {
        'max_slew': 170,      # [T/m/s]
        'max_grad': 38,       # [mT/m]
        'adc_dwell': 1.4e-6,  # [s]
        'os': 8
    }

    # HyperSLICE parameters (from config_optimized_traj)
    base_resolution = 240
    fov = [40.0]  # [cm]
    vd_spiral_arms = 16
    vd_inner_cutoff = 0.15
    pre_vd_outer_cutoff = 0.41288
    vd_outer_density = 0.07
    vd_type = 'hanning'
    min_max_arm_time = [0.88, 1.67]  # [ms]
    max_tempres = 55.0  # [ms]
    deadtime = 2.0  # [ms]

    # Design trajectory
    k, g, s, t, final_arms, views, ro_time, temp_res = vds_hyperslice(
        sys=sys,
        base_resolution=base_resolution,
        fov=fov,
        vd_spiral_arms=vd_spiral_arms,
        vd_inner_cutoff=vd_inner_cutoff,
        pre_vd_outer_cutoff=pre_vd_outer_cutoff,
        vd_outer_density=vd_outer_density,
        vd_type=vd_type,
        min_max_arm_time=min_max_arm_time,
        max_tempres=max_tempres,
        deadtime=deadtime
    )

    if k is not None:
        print("\n" + "=" * 60)
        print("SUCCESS: Trajectory design completed")
        print("=" * 60)

        # Plot if matplotlib available
        try:
            from libspiral import plotgradinfo
            import matplotlib.pyplot as plt

            fig = plotgradinfo(g, sys['adc_dwell'])
            fig.suptitle('HyperSLICE-style VDS Trajectory', fontsize=16)
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")
    else:
        print("\n" + "=" * 60)
        print("FAILED: Could not design trajectory")
        print("=" * 60)
