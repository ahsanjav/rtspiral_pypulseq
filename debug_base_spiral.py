import numpy as np
import matplotlib.pyplot as plt
from pypulseq import Opts
from utils import load_params
from libspiral import vds_fixed_ro, raster_to_grad

# Load basic config
params = load_params('example_config_freemax.toml', './')

# Setup system
GRT = params['system']['grad_raster_time']
spiral_sys = {
    'max_slew': params['system']['max_slew']*params['spiral']['slew_ratio'],
    'max_grad': params['system']['max_grad']*0.99,
    'adc_dwell': params['spiral']['adc_dwell'],
    'grad_raster_time': GRT,
    'os': 8
}

fov = params['acquisition']['fov']
res = params['acquisition']['resolution']
Tread = params['spiral']['ro_duration']

print(f"Generating spiral with FOV={fov}, Res={res}, Tread={Tread}")

# Generate Spiral
k, g, t, n_int = vds_fixed_ro(spiral_sys, fov, res, Tread)
t_grad, g_grad = raster_to_grad(g, spiral_sys['adc_dwell'], GRT)

print(f"Gradient Shape: {g_grad.shape}")
print(f"Interleaves: {n_int}")

# Plot
plt.figure()
plt.plot(g_grad[:,0], label='Gx')
plt.plot(g_grad[:,1], label='Gy')
plt.title('Base Gradient Waveforms')
plt.legend()
plt.show()

# Calculate K-space trajectory of base arm
k_base = np.cumsum(g_grad, axis=0) * GRT * 42.58e6 # Approximate (gamma ignored mostly)
plt.figure()
plt.plot(k_base[:,0], k_base[:,1])
plt.title('Base K-Space Trajectory (No Rotation)')
plt.axis('equal')
plt.show()
