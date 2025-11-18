try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import os
import warnings


def validate_hyperslice_params(params: dict) -> bool:
    """
    Validate HyperSLICE-specific parameters.

    Parameters
    ----------
    params : dict
        Configuration dictionary

    Returns
    -------
    is_valid : bool
        True if HyperSLICE parameters are valid

    Raises
    ------
    ValueError
        If critical parameters are missing or invalid
    """

    if 'spiral' not in params or 'hyperslice' not in params['spiral']:
        return False  # Not using HyperSLICE mode

    hs = params['spiral']['hyperslice']

    # Check required parameters
    required = ['base_resolution', 'vd_spiral_arms', 'vd_inner_cutoff',
                'pre_vd_outer_cutoff', 'vd_outer_density']

    for param in required:
        if param not in hs:
            raise ValueError(f"Missing required HyperSLICE parameter: {param}")

    # Validate ranges
    if not (0 <= hs['vd_inner_cutoff'] <= 1):
        raise ValueError(f"vd_inner_cutoff must be in [0, 1], got {hs['vd_inner_cutoff']}")

    if not (0 <= hs['vd_outer_density'] <= 1):
        raise ValueError(f"vd_outer_density must be in [0, 1], got {hs['vd_outer_density']}")

    if hs['base_resolution'] <= 0:
        raise ValueError(f"base_resolution must be positive, got {hs['base_resolution']}")

    if hs['vd_spiral_arms'] <= 0:
        raise ValueError(f"vd_spiral_arms must be positive, got {hs['vd_spiral_arms']}")

    # Validate timing parameters if present
    if 'min_max_arm_time' in hs:
        arm_times = hs['min_max_arm_time']
        if len(arm_times) != 2:
            raise ValueError(f"min_max_arm_time must have 2 elements, got {len(arm_times)}")
        if arm_times[0] >= arm_times[1]:
            raise ValueError(f"min_max_arm_time[0] must be < min_max_arm_time[1]")

    # Warnings for unusual values
    if hs['vd_inner_cutoff'] > 0.5:
        warnings.warn(
            f"vd_inner_cutoff = {hs['vd_inner_cutoff']} is unusually large. "
            "Typical range is 0-0.3."
        )

    if hs['vd_outer_density'] < 0.03:
        warnings.warn(
            f"vd_outer_density = {hs['vd_outer_density']} is very small. "
            "This may result in excessive undersampling at k-space edges."
        )

    return True


def load_params(param_filename: str, param_dir: str = "protocols", validate: bool = True):
    """
    Load and optionally validate sequence parameters from TOML file.

    Parameters
    ----------
    param_filename : str
        Parameter file name (with or without .toml extension)
    param_dir : str, optional
        Directory containing parameter file (default: "protocols")
    validate : bool, optional
        Whether to validate HyperSLICE parameters if present (default: True)

    Returns
    -------
    params : dict
        Configuration dictionary

    Raises
    ------
    ValueError
        If validation is enabled and parameters are invalid
    """
    fname_ = param_filename + ".toml" if not param_filename.endswith(".toml") else param_filename
    with open(os.path.join(param_dir, fname_), "rb") as f:
        toml_dict = tomllib.load(f)

    # Validate HyperSLICE parameters if present and validation is enabled
    if validate:
        try:
            validate_hyperslice_params(toml_dict)
        except ValueError as e:
            # Re-raise with more context
            raise ValueError(f"Invalid HyperSLICE parameters in {fname_}: {e}")

    return toml_dict