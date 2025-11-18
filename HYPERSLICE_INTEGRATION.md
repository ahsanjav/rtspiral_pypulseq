# HyperSLICE Integration for rtspiral_pypulseq

This document describes the integration of HyperSLICE-style trajectory design into the rtspiral_pypulseq framework.

## Overview

HyperSLICE ([mrphys/HyperSLICE](https://github.com/mrphys/HyperSLICE)) provides an advanced spiral trajectory design approach optimized for dynamic MRI with temporal resolution constraints. This integration brings HyperSLICE's trajectory parameterization to rtspiral_pypulseq while maintaining:

- **PyPulseq output** for scanner deployment
- **Existing rewinder design** methods (gropt, M1 nulling, etc.)
- **No TensorFlow dependency** (pure Python translation)
- **Backward compatibility** with standard rtspiral workflows

## Key Features

### 1. **Intuitive Variable-Density Control**

Instead of FOV coefficients, HyperSLICE uses:
- `vd_inner_cutoff`: Radius where undersampling starts (0-1)
- `vd_outer_cutoff`: Radius where max undersampling is reached
- `vd_outer_density`: Final density at k-space edge

This is more intuitive than the traditional FOV coefficient list.

### 2. **Temporal Resolution Optimization**

Automatically adjusts the number of spiral arms to:
- Fit within a specified readout time window (`min_max_arm_time`)
- Achieve a target temporal resolution (`max_tempres`)
- Calculate optimal views per temporal frame

### 3. **Dynamic Imaging Support**

Designed for applications requiring temporal resolution optimization:
- Real-time cardiac MRI
- Cine imaging
- Dynamic contrast-enhanced imaging
- Flow imaging

## Files Created

### Core Modules

1. **`modules/libspiral/src/libspiral_hyperslice.py`**
   - `vds_hyperslice()`: Main trajectory design function
   - `hyperslice_vd_to_fov()`: VD parameter translation
   - Pure Python implementation (no TensorFlow)

2. **`utils/temporal_utils.py`**
   - `calculate_temporal_resolution()`: TR and temporal res calculations
   - `calculate_num_views()`: Views for target temporal resolution
   - `validate_timing_constraints()`: Timing feasibility checks
   - `print_temporal_summary()`: Formatted output

### Main Script

3. **`write_rtspiral_dd.py`**
   - 2D real-time spiral sequence generator
   - Uses HyperSLICE trajectory design
   - Maintains PyPulseq output format
   - Compatible with existing kernels and preparations

### Configuration

4. **`example_config_hyperslice.toml`**
   - Complete example configuration
   - Detailed parameter documentation
   - Recommended presets for different applications

### Modified Files

5. **`utils/load_params.py`**
   - Added `validate_hyperslice_params()` for parameter validation
   - Range checking and warnings for unusual values

6. **`utils/traj_utils.py`**
   - Extended `save_metadata()` to save HyperSLICE-specific metadata
   - Temporal resolution, views per frame, VD parameters

7. **`modules/libspiral/src/__init__.py`**
   - Exports new HyperSLICE functions
   - `vds_hyperslice`, `hyperslice_vd_to_fov`

## Usage

### Basic Usage

```bash
# Using the example configuration
python write_rtspiral_dd.py -c example_config_hyperslice

# Using a custom configuration
python write_rtspiral_dd.py -c my_config_hyperslice
```

### Configuration Example

```toml
[spiral.hyperslice]
# Base resolution (matrix size)
base_resolution = 240

# Spiral arms (will be optimized)
vd_spiral_arms = 16

# Variable-density parameters
vd_inner_cutoff = 0.15         # Fully sample inner 15%
pre_vd_outer_cutoff = 0.41288  # Transition offset
vd_outer_density = 0.07        # 7% density at edge
vd_type = 'hanning'            # Smooth transition

# Temporal optimization
min_max_arm_time = [0.88, 1.67]  # Readout time window [ms]
max_tempres = 55.0               # Target temporal res [ms]
deadtime = 2.0                   # Dead time [ms]

# Options
reverse = false                  # Trajectory direction
ordering = 'linear'              # 'linear', 'golden', 'ga', 'tiny'
```

### Python API

```python
from libspiral.libspiral_hyperslice import vds_hyperslice

# System parameters
sys = {
    'max_slew': 170,      # [T/m/s]
    'max_grad': 38,       # [mT/m]
    'adc_dwell': 1.4e-6,  # [s]
    'os': 8
}

# Design trajectory
k, g, s, t, arms, views, ro_time, temp_res = vds_hyperslice(
    sys=sys,
    base_resolution=240,
    fov=[40.0],  # [cm]
    vd_spiral_arms=16,
    vd_inner_cutoff=0.15,
    pre_vd_outer_cutoff=0.41288,
    vd_outer_density=0.07,
    min_max_arm_time=[0.88, 1.67],  # [ms]
    max_tempres=55.0,  # [ms]
    deadtime=2.0  # [ms]
)

print(f"Optimized to {arms} arms")
print(f"Views per frame: {views}")
print(f"Temporal resolution: {temp_res:.1f} ms")
```

## Parameter Guidelines

### Variable-Density Parameters

**`vd_inner_cutoff`** (0-1, typical: 0.1-0.3)
- Normalized radius where undersampling starts
- Larger = more fully-sampled center = better SNR, longer readout
- Smaller = more undersampling = shorter readout, lower SNR

**`vd_outer_density`** (0-1, typical: 0.05-0.25)
- Final k-space density at edge (fraction of Nyquist)
- 0.07 = 7% of Nyquist ≈ 14x undersampling
- Smaller = more aggressive undersampling
- Must be balanced with reconstruction capability

**`pre_vd_outer_cutoff`** (0-1, typical: 0.2-0.6)
- Controls transition region width
- Larger = more gradual density transition
- Used to calculate actual outer cutoff:
  ```
  vd_outer_cutoff = vd_inner_cutoff + 0.1 +
                    pre_vd_outer_cutoff * (1 - vd_inner_cutoff - 0.1)
  ```

**`vd_type`** ('linear', 'quadratic', 'hanning')
- Density transition profile
- 'hanning': smooth, well-behaved (recommended)
- Note: Currently only affects FOV coefficient mapping

### Temporal Optimization Parameters

**`min_max_arm_time`** [min, max] in ms
- Acceptable readout duration window
- Algorithm iterates spiral arm count to fit this window
- Tighter window = fewer valid solutions
- Typical: [0.5, 2.0] ms for cardiac

**`max_tempres`** (ms)
- Maximum temporal resolution target
- Number of views = floor(max_tempres / TR)
- Smaller = better temporal res, more undersampling per frame
- Typical cardiac: 30-80 ms

**`deadtime`** (ms)
- Non-readout time per TR (rewinders, spoilers, delays)
- TR = readout_time + deadtime
- Typical: 0.5-2.0 ms depending on rewinder method

### Recommended Presets

#### Cardiac Cine (Balanced)
```toml
base_resolution = 240
vd_inner_cutoff = 0.15
pre_vd_outer_cutoff = 0.41
vd_outer_density = 0.07
min_max_arm_time = [0.88, 1.67]
max_tempres = 55
```
- Good balance of spatial/temporal resolution
- Suitable for most cardiac applications

#### High Spatial Resolution
```toml
base_resolution = 320
vd_inner_cutoff = 0.25
vd_outer_density = 0.15
min_max_arm_time = [1.5, 3.0]
max_tempres = 80
```
- Emphasizes spatial resolution
- Lower undersampling at k-space edge
- Longer temporal frames acceptable

#### High Temporal Resolution
```toml
base_resolution = 192
vd_inner_cutoff = 0.10
vd_outer_density = 0.05
min_max_arm_time = [0.5, 1.2]
max_tempres = 35
```
- Fast temporal dynamics
- Aggressive undersampling
- Requires robust reconstruction (e.g., compressed sensing)

## Differences from Standard rtspiral_pypulseq

| Aspect | Standard | HyperSLICE |
|--------|----------|------------|
| **Resolution** | Specified directly | Calculated from base_resolution and FOV |
| **VD Control** | FOV coefficient list | Cutoff radii + density factor |
| **Readout Time** | Fixed, specified | Optimized to fit time window |
| **Temporal Res** | Manual calculation | Automatic optimization |
| **Arm Count** | Iterated for resolution | Iterated for timing |
| **Output** | PyPulseq .seq | PyPulseq .seq (same) |

## Workflow Comparison

### Standard Workflow
1. Specify: FOV, resolution, readout time
2. Algorithm finds number of arms for resolution
3. User calculates temporal resolution manually

### HyperSLICE Workflow
1. Specify: FOV, base_resolution, time window, max temporal res
2. Algorithm finds number of arms for time window
3. Algorithm calculates views and temporal resolution
4. Resolution = FOV / base_resolution

## Output Files

### Sequence File (`.seq`)
Standard PyPulseq format with additional definitions:
- `Temporal_Res_ms`: Temporal resolution

### Trajectory File (`.mat`)
Standard format with additional metadata:
```matlab
param.hyperslice.base_resolution
param.hyperslice.views_per_frame
param.hyperslice.temporal_resolution_ms
param.hyperslice.readout_time_ms
param.hyperslice.vd_inner_cutoff
param.hyperslice.vd_outer_density
param.hyperslice.reverse_traj
```

## Algorithm Details

### Temporal Optimization Loop

```
Initialize: vd_spiral_arms = initial_guess

For attempt in 1..max_guesses:
    1. Design trajectory with current arm count
    2. Measure readout_time

    3. If min_arm_time < readout_time < max_arm_time:
          Success! Exit loop

    4. Else if readout_time < min_arm_time:
          vd_spiral_arms -= 1  (fewer arms = longer per arm)

    5. Else if readout_time > max_arm_time:
          vd_spiral_arms += 1  (more arms = shorter per arm)

Calculate:
    TR = readout_time + deadtime
    views = floor(max_tempres / TR)
    temporal_res = TR * views
```

### VD Parameter Translation

The `hyperslice_vd_to_fov()` function maps HyperSLICE VD parameters to FOV coefficients:

```python
# Calculate outer cutoff
vd_outer_cutoff = vd_inner_cutoff + 0.1 +
                  pre_vd_outer_cutoff * (1 - vd_inner_cutoff - 0.1)

# Generate FOV coefficients
for each k-space radius (normalized 0-1):
    if radius <= vd_inner_cutoff:
        fov_coeff = 1.0  # Fully sampled
    elif radius >= vd_outer_cutoff:
        fov_coeff = vd_outer_density  # Max undersampling
    else:
        # Linear interpolation (TODO: use vd_type)
        fov_coeff = interpolate(1.0, vd_outer_density)
```

Note: This is an approximate mapping. Exact matching with TensorFlow's implementation may require empirical tuning.

## Testing

### Test Trajectory Design Module

```bash
cd modules/libspiral/src
python libspiral_hyperslice.py
```

Expected output:
- Iteration log showing arm count adjustment
- Final trajectory parameters
- Trajectory plots (if matplotlib available)

### Test Temporal Utilities

```bash
cd utils
python temporal_utils.py
```

Expected output:
- Test results for temporal resolution calculations
- Timing validation examples
- Formatted summary output

### Test Full Sequence Generation

```bash
python write_rtspiral_dd.py -c example_config_hyperslice
```

Expected output:
- HyperSLICE parameter summary
- Trajectory optimization log
- Temporal resolution summary
- PyPulseq sequence file in `out_seq/`
- Trajectory metadata in `out_trajectory/`

## Troubleshooting

### "Failed to design spiral trajectory"

**Problem**: Optimization loop couldn't find valid arm count

**Solutions**:
1. Widen `min_max_arm_time` window
2. Adjust `vd_spiral_arms` initial guess (try ±5)
3. Check system limits (max_grad, max_slew)
4. Reduce base_resolution or increase vd_outer_density

### Temporal resolution too coarse

**Problem**: Not enough views fit in max_tempres

**Solutions**:
1. Increase `max_tempres`
2. Reduce `deadtime` (optimize rewinder)
3. Shorten readout (more arms, higher vd_outer_density)

### Temporal resolution too fine (excessive undersampling)

**Problem**: Too many views, high undersampling per frame

**Solutions**:
1. Decrease `max_tempres`
2. Increase `deadtime`
3. Lengthen readout (fewer arms, lower vd_outer_density)

### Reconstruction artifacts

**Problem**: k-space undersampling too aggressive

**Solutions**:
1. Increase `vd_outer_density` (less aggressive edge undersampling)
2. Increase `vd_inner_cutoff` (more fully-sampled center)
3. Use compressed sensing or parallel imaging reconstruction
4. Increase `views_per_frame` by adjusting temporal parameters

## References

1. HyperSLICE: https://github.com/mrphys/HyperSLICE
2. rtspiral_pypulseq: Original framework
3. PyPulseq: https://github.com/imr-framework/pypulseq

## Authors

- HyperSLICE trajectory design adapted from mrphys/HyperSLICE
- Integration by rtspiral_pypulseq team
- Date: 2025

## License

Follows rtspiral_pypulseq license. HyperSLICE components adapted under their respective license.
