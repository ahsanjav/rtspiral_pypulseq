"""
Temporal resolution optimization utilities for spiral trajectories.

This module provides helper functions for calculating and optimizing temporal
resolution in dynamic spiral imaging sequences. It's particularly useful for
real-time and cine imaging where temporal resolution is a critical parameter.

Functions in this module support both standard rtspiral_pypulseq workflows
and HyperSLICE-style temporal optimization.

Author: rtspiral_pypulseq team
Date: 2025
"""

from typing import Tuple, Optional
import numpy as np


def calculate_temporal_resolution(
    arm_time: float,
    deadtime: float,
    num_views: int
) -> Tuple[float, float]:
    """
    Calculate temporal resolution from timing parameters.

    In spiral imaging, one temporal frame is composed of multiple spiral arms
    (views). The temporal resolution is the time required to acquire all views
    needed to reconstruct one frame.

    Parameters
    ----------
    arm_time : float
        Time to acquire one spiral arm [ms or s, units preserved]
    deadtime : float
        Dead time between spiral arms (for rewinders, etc.) [same units as arm_time]
    num_views : int
        Number of spiral arms per temporal frame

    Returns
    -------
    temporal_res : float
        Temporal resolution (time per frame) [same units as input]
    TR : float
        Repetition time (time per arm) [same units as input]

    Examples
    --------
    >>> calculate_temporal_resolution(arm_time=1.5, deadtime=0.5, num_views=10)
    (20.0, 2.0)  # 20 ms temporal res, 2 ms TR

    >>> calculate_temporal_resolution(arm_time=0.0015, deadtime=0.0005, num_views=10)
    (0.02, 0.002)  # 20 ms temporal res, 2 ms TR (when using seconds)
    """

    TR = arm_time + deadtime
    temporal_res = TR * num_views

    return temporal_res, TR


def calculate_num_views(
    arm_time: float,
    deadtime: float,
    max_temporal_res: float
) -> int:
    """
    Calculate number of views that fit within a maximum temporal resolution.

    This is the inverse of calculate_temporal_resolution, useful when you have
    a target temporal resolution and want to know how many views you can acquire.

    Parameters
    ----------
    arm_time : float
        Time to acquire one spiral arm [ms or s]
    deadtime : float
        Dead time between spiral arms [same units as arm_time]
    max_temporal_res : float
        Maximum allowed temporal resolution [same units as arm_time]

    Returns
    -------
    num_views : int
        Number of views that fit within max_temporal_res

    Examples
    --------
    >>> calculate_num_views(arm_time=1.5, deadtime=0.5, max_temporal_res=50)
    25  # Can fit 25 views in 50 ms

    Notes
    -----
    Returns at least 1 view even if arm_time + deadtime > max_temporal_res
    """

    TR = arm_time + deadtime

    if TR <= 0:
        raise ValueError("TR must be positive")

    num_views = int(max_temporal_res / TR)

    # Ensure at least 1 view
    return max(1, num_views)


def optimize_arms_for_time(
    arm_times: np.ndarray,
    min_arm_time: float,
    max_arm_time: float,
    prefer_more_arms: bool = True
) -> Tuple[int, float]:
    """
    Find optimal number of spiral arms from pre-computed arm times.

    Given an array of arm times corresponding to different arm counts,
    select the arm count that best fits within the timing constraints.

    Parameters
    ----------
    arm_times : np.ndarray
        Array of arm times [ms or s], indexed by arm count
        arm_times[i] = time for i spiral arms
    min_arm_time : float
        Minimum acceptable arm time [same units]
    max_arm_time : float
        Maximum acceptable arm time [same units]
    prefer_more_arms : bool, optional
        If True, prefer more arms when multiple solutions exist (default: True)
        More arms = shorter per-arm time = better motion robustness

    Returns
    -------
    optimal_arm_count : int
        Optimal number of spiral arms (0 if no valid solution)
    actual_arm_time : float
        Actual arm time for optimal count

    Examples
    --------
    >>> arm_times = np.array([0, 5.0, 3.0, 2.0, 1.5, 1.2])
    >>> optimize_arms_for_time(arm_times, min_arm_time=1.0, max_arm_time=2.5)
    (4, 1.5)  # 4 arms gives 1.5 ms, which fits [1.0, 2.5]
    """

    valid_indices = np.where(
        (arm_times >= min_arm_time) & (arm_times <= max_arm_time)
    )[0]

    if len(valid_indices) == 0:
        return 0, 0.0

    if prefer_more_arms:
        # More arms = higher index (assuming arm_times decreases with more arms)
        optimal_idx = valid_indices[-1]
    else:
        # Fewer arms = lower index
        optimal_idx = valid_indices[0]

    return optimal_idx, arm_times[optimal_idx]


def validate_timing_constraints(
    readout_time: float,
    rewinder_time: float,
    rf_time: float,
    min_TR: Optional[float] = None,
    max_TR: Optional[float] = None,
    system_deadtime: float = 0.0
) -> Tuple[bool, float, str]:
    """
    Validate that timing constraints can be achieved with given parameters.

    Checks whether the total TR fits within specified bounds, accounting for
    all sequence components (RF, readout, rewinder, system deadtimes).

    Parameters
    ----------
    readout_time : float
        Spiral readout duration [ms or s]
    rewinder_time : float
        Gradient rewinder duration [same units]
    rf_time : float
        RF excitation duration [same units]
    min_TR : float, optional
        Minimum TR constraint [same units]
    max_TR : float, optional
        Maximum TR constraint [same units]
    system_deadtime : float, optional
        Additional system dead time (ADC deadtime, etc.) [same units]

    Returns
    -------
    is_valid : bool
        True if constraints are met (or no constraints specified)
    actual_TR : float
        Actual TR with given parameters [same units as input]
    message : str
        Descriptive message about validation result

    Examples
    --------
    >>> validate_timing_constraints(
    ...     readout_time=3.0, rewinder_time=1.0, rf_time=1.0,
    ...     min_TR=4.0, max_TR=6.0, system_deadtime=0.5
    ... )
    (True, 5.5, 'Timing constraints satisfied: TR = 5.50 ms')
    """

    actual_TR = readout_time + rewinder_time + rf_time + system_deadtime

    # Check constraints
    if min_TR is not None and actual_TR < min_TR:
        message = f"TR too short: {actual_TR:.2f} < {min_TR:.2f}"
        return False, actual_TR, message

    if max_TR is not None and actual_TR > max_TR:
        message = f"TR too long: {actual_TR:.2f} > {max_TR:.2f}"
        return False, actual_TR, message

    # Determine units (heuristic: if TR > 100, likely in ms; else in s)
    unit = "ms" if actual_TR > 1.0 else "s"
    message = f"Timing constraints satisfied: TR = {actual_TR:.2f} {unit}"

    return True, actual_TR, message


def estimate_undersampling_factor(
    n_arms_actual: int,
    n_arms_nyquist: int
) -> float:
    """
    Estimate undersampling factor from arm counts.

    In spiral imaging, the Nyquist-sampled trajectory requires a certain
    number of interleaves. Using fewer arms results in undersampling.

    Parameters
    ----------
    n_arms_actual : int
        Actual number of spiral arms used
    n_arms_nyquist : int
        Number of arms required for Nyquist sampling

    Returns
    -------
    undersampling_factor : float
        Undersampling factor (e.g., 2.0 = 2x undersampling)

    Examples
    --------
    >>> estimate_undersampling_factor(n_arms_actual=10, n_arms_nyquist=20)
    2.0

    >>> estimate_undersampling_factor(n_arms_actual=20, n_arms_nyquist=20)
    1.0  # Fully sampled
    """

    if n_arms_actual <= 0:
        raise ValueError("n_arms_actual must be positive")

    if n_arms_nyquist <= 0:
        raise ValueError("n_arms_nyquist must be positive")

    return n_arms_nyquist / n_arms_actual


def print_temporal_summary(
    arm_time: float,
    deadtime: float,
    num_views: int,
    n_arms_total: int,
    temporal_res: Optional[float] = None,
    undersampling: Optional[float] = None
) -> None:
    """
    Print a formatted summary of temporal resolution parameters.

    Useful for debugging and reporting sequence timing.

    Parameters
    ----------
    arm_time : float
        Spiral arm readout time [ms]
    deadtime : float
        Dead time between arms [ms]
    num_views : int
        Number of views per temporal frame
    n_arms_total : int
        Total number of spiral arms in sequence
    temporal_res : float, optional
        Temporal resolution [ms] (calculated if not provided)
    undersampling : float, optional
        Undersampling factor (e.g., 2.0 for 2x)

    Examples
    --------
    >>> print_temporal_summary(
    ...     arm_time=1.5, deadtime=0.5, num_views=10,
    ...     n_arms_total=20, undersampling=2.0
    ... )
    ========================================
    Temporal Resolution Summary
    ========================================
    Arm readout time:    1.50 ms
    Dead time:           0.50 ms
    TR (per arm):        2.00 ms
    Views per frame:     10
    Temporal resolution: 20.00 ms
    Total arms:          20
    Total frames:        2
    Undersampling:       2.0x
    ========================================
    """

    if temporal_res is None:
        temporal_res, TR = calculate_temporal_resolution(arm_time, deadtime, num_views)
    else:
        TR = arm_time + deadtime

    n_frames = n_arms_total // num_views if num_views > 0 else 0

    print("=" * 60)
    print("Temporal Resolution Summary")
    print("=" * 60)
    print(f"Arm readout time:    {arm_time:.2f} ms")
    print(f"Dead time:           {deadtime:.2f} ms")
    print(f"TR (per arm):        {TR:.2f} ms")
    print(f"Views per frame:     {num_views}")
    print(f"Temporal resolution: {temporal_res:.2f} ms")
    print(f"Total arms:          {n_arms_total}")
    print(f"Total frames:        {n_frames}")

    if undersampling is not None:
        print(f"Undersampling:       {undersampling:.1f}x")

    print("=" * 60)


if __name__ == "__main__":
    """Test temporal utilities"""

    print("Testing temporal resolution utilities\n")

    # Test 1: Calculate temporal resolution
    print("Test 1: Calculate temporal resolution")
    print("-" * 60)
    arm_time = 1.5  # ms
    deadtime = 0.5  # ms
    num_views = 10
    temp_res, TR = calculate_temporal_resolution(arm_time, deadtime, num_views)
    print(f"Input: arm_time={arm_time} ms, deadtime={deadtime} ms, views={num_views}")
    print(f"Output: TR={TR} ms, temporal_res={temp_res} ms\n")

    # Test 2: Calculate number of views
    print("Test 2: Calculate number of views")
    print("-" * 60)
    max_temp_res = 50  # ms
    views = calculate_num_views(arm_time, deadtime, max_temp_res)
    print(f"Input: arm_time={arm_time} ms, deadtime={deadtime} ms, max_temp_res={max_temp_res} ms")
    print(f"Output: {views} views can fit\n")

    # Test 3: Validate timing constraints
    print("Test 3: Validate timing constraints")
    print("-" * 60)
    is_valid, actual_TR, msg = validate_timing_constraints(
        readout_time=3.0,
        rewinder_time=1.0,
        rf_time=1.0,
        min_TR=4.0,
        max_TR=6.0,
        system_deadtime=0.5
    )
    print(f"Result: {msg}")
    print(f"Valid: {is_valid}\n")

    # Test 4: Print summary
    print("Test 4: Temporal summary")
    print_temporal_summary(
        arm_time=1.5,
        deadtime=0.5,
        num_views=10,
        n_arms_total=20,
        undersampling=2.0
    )
