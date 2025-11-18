#!/usr/bin/env python
"""
Enhanced plotting for HyperSLICE trajectory analysis.
Shows correct gradient values and trajectory characteristics.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.insert(0, 'modules/libspiral/src')

from tensorflow_mri_trajectory import gen_spiral_traj_tfmri, convert_to_pypulseq_format
from libspiral import raster_to_grad, calcgradinfo

def plot_hyperslice_trajectory():
    """Generate and plot HyperSLICE trajectory with correct values."""

    # Generate trajectory
    print("Generating HyperSLICE trajectory...")
    traj = gen_spiral_traj_tfmri(
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
    )

    # Get parameters
    fov_m = 0.4  # 400 mm
    res_m = fov_m / 240
    k_nyquist = 0.5 / res_m  # 1/m

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('HyperSLICE Spiral Trajectory Analysis', fontsize=16, fontweight='bold')

    # ============ Plot 1: All spiral arms in k-space ============
    ax1 = plt.subplot(3, 4, 1)
    colors = plt.cm.rainbow(np.linspace(0, 1, traj['n_arms']))

    for arm in range(traj['n_arms']):
        kx = traj['kx'][arm, :] / (2 * np.pi)  # rad/m to 1/m
        ky = traj['ky'][arm, :] / (2 * np.pi)
        ax1.plot(kx, ky, color=colors[arm], alpha=0.7, linewidth=0.8)

    # Add circles for variable density regions
    circle_inner = Circle((0, 0), traj['vd_inner_cutoff'] * k_nyquist,
                          fill=False, edgecolor='black', linestyle='--', label=f"Inner {traj['vd_inner_cutoff']:.0%}")
    circle_outer = Circle((0, 0), traj['vd_outer_cutoff'] * k_nyquist,
                          fill=False, edgecolor='red', linestyle='--', label=f"Outer {traj['vd_outer_cutoff']:.0%}")
    circle_nyquist = Circle((0, 0), k_nyquist,
                           fill=False, edgecolor='green', linestyle='-', label="Nyquist")
    ax1.add_patch(circle_inner)
    ax1.add_patch(circle_outer)
    ax1.add_patch(circle_nyquist)

    ax1.set_xlabel('$k_x$ [1/m]')
    ax1.set_ylabel('$k_y$ [1/m]')
    ax1.set_title(f'K-space Coverage ({traj["n_arms"]} arms)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim([-k_nyquist*1.1, k_nyquist*1.1])
    ax1.set_ylim([-k_nyquist*1.1, k_nyquist*1.1])

    # ============ Plot 2: Single arm detail ============
    ax2 = plt.subplot(3, 4, 2)
    arm = 0
    kx = traj['kx'][arm, :] / (2 * np.pi)
    ky = traj['ky'][arm, :] / (2 * np.pi)

    # Color by time
    points = ax2.scatter(kx, ky, c=np.arange(len(kx)), cmap='viridis', s=2)
    plt.colorbar(points, ax=ax2, label='Sample index')

    ax2.set_xlabel('$k_x$ [1/m]')
    ax2.set_ylabel('$k_y$ [1/m]')
    ax2.set_title('Single Arm (colored by time)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # ============ Plot 3: Gradients vs Time ============
    ax3 = plt.subplot(3, 4, 3)
    gx = traj['gx'][0, :]  # First arm
    gy = traj['gy'][0, :]
    t_ms = np.arange(len(gx)) * traj['dwell_time'] * 1e3  # ms

    ax3.plot(t_ms, gx, 'b-', label='$G_x$', linewidth=1)
    ax3.plot(t_ms, gy, 'r-', label='$G_y$', linewidth=1)
    ax3.plot(t_ms, np.sqrt(gx**2 + gy**2), 'k--', label='|G|', linewidth=1.5)
    ax3.axhline(y=22, color='gray', linestyle=':', label='Max (22 mT/m)')

    ax3.set_xlabel('Time [ms]')
    ax3.set_ylabel('Gradient [mT/m]')
    ax3.set_title(f'Gradient Waveforms (max: {np.max(np.sqrt(gx**2 + gy**2)):.1f} mT/m)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # ============ Plot 4: K-space extent vs time ============
    ax4 = plt.subplot(3, 4, 4)
    k_mag = np.sqrt(kx**2 + ky**2)
    ax4.plot(t_ms, k_mag, 'g-', linewidth=2)
    ax4.axhline(y=k_nyquist, color='red', linestyle='--', label='Nyquist')
    ax4.fill_between(t_ms, 0, traj['vd_inner_cutoff'] * k_nyquist,
                      alpha=0.2, color='blue', label='Fully sampled')

    ax4.set_xlabel('Time [ms]')
    ax4.set_ylabel('|k| [1/m]')
    ax4.set_title('K-space Radius vs Time')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    # ============ Plot 5: Sampling density ============
    ax5 = plt.subplot(3, 4, 5)
    # Calculate local density as spacing between consecutive k-space points
    dk = np.sqrt(np.diff(kx)**2 + np.diff(ky)**2)
    density = 1 / (dk + 1e-10)  # Inverse of spacing
    density_norm = density / np.max(density)

    ax5.plot(t_ms[:-1], density_norm, 'purple', linewidth=1.5)
    ax5.set_xlabel('Time [ms]')
    ax5.set_ylabel('Relative Density')
    ax5.set_title('Sampling Density (variable)')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.1])

    # ============ Plot 6: Slew rates ============
    ax6 = plt.subplot(3, 4, 6)
    dt = traj['dwell_time']
    dgx_dt = np.gradient(gx, dt) * 1e-3  # mT/m/s to T/m/s
    dgy_dt = np.gradient(gy, dt) * 1e-3
    slew_mag = np.sqrt(dgx_dt**2 + dgy_dt**2)

    ax6.plot(t_ms, dgx_dt, 'b-', label='$S_x$', linewidth=1, alpha=0.7)
    ax6.plot(t_ms, dgy_dt, 'r-', label='$S_y$', linewidth=1, alpha=0.7)
    ax6.plot(t_ms, slew_mag, 'k--', label='|S|', linewidth=1.5)
    ax6.axhline(y=120, color='gray', linestyle=':', label='Max (120 T/m/s)')

    ax6.set_xlabel('Time [ms]')
    ax6.set_ylabel('Slew Rate [T/m/s]')
    ax6.set_title(f'Slew Rates (max: {np.max(slew_mag):.1f} T/m/s)')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)

    # ============ Plot 7: After raster_to_grad ============
    ax7 = plt.subplot(3, 4, 7)
    k, g, s, t = convert_to_pypulseq_format(traj)
    GRT = 10e-6
    t_grad, g_grad = raster_to_grad(g, traj['dwell_time'], GRT)

    t_grad_ms = t_grad * 1e3
    ax7.plot(t_grad_ms, g_grad[:, 0], 'b-', label='$G_x$ (rastered)', linewidth=1)
    ax7.plot(t_grad_ms, g_grad[:, 1], 'r-', label='$G_y$ (rastered)', linewidth=1)
    ax7.plot(t_grad_ms, np.sqrt(g_grad[:, 0]**2 + g_grad[:, 1]**2),
             'k--', label='|G| (rastered)', linewidth=1.5)

    ax7.set_xlabel('Time [ms]')
    ax7.set_ylabel('Gradient [mT/m]')
    ax7.set_title(f'After Rasterization (GRT={GRT*1e6:.0f}µs)')
    ax7.legend(loc='best')
    ax7.grid(True, alpha=0.3)

    # ============ Plot 8: Arm rotation angles ============
    ax8 = plt.subplot(3, 4, 8)
    # Calculate angle of first point for each arm
    angles = []
    for arm in range(traj['n_arms']):
        # Use a point away from origin to get angle
        idx = 50
        angle = np.arctan2(traj['ky'][arm, idx], traj['kx'][arm, idx]) * 180 / np.pi
        if angle < 0:
            angle += 360
        angles.append(angle)

    ax8.bar(range(traj['n_arms']), angles, color=colors)
    ax8.set_xlabel('Arm Index')
    ax8.set_ylabel('Starting Angle [degrees]')
    ax8.set_title(f'Arm Rotation (Δ={360/traj["n_arms"]:.1f}°)')
    ax8.grid(True, alpha=0.3, axis='y')

    # ============ Plot 9-12: Parameter Summary ============
    ax9 = plt.subplot(3, 4, (9, 12))
    ax9.axis('off')

    # Create parameter text
    param_text = f"""
    TRAJECTORY PARAMETERS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Base Resolution:     {traj['base_resolution']}
    FOV:                 {fov_m*1000:.0f} mm
    Pixel Size:          {res_m*1000:.2f} mm

    SPIRAL DESIGN:
    Number of Arms:      {traj['n_arms']}
    Samples per Arm:     {traj['n_samples']}
    Readout Time:        {traj['readout_time']:.2f} ms
    Dwell Time:          {traj['dwell_time']*1e6:.1f} µs

    VARIABLE DENSITY:
    Inner Cutoff:        {traj['vd_inner_cutoff']*100:.0f}% (fully sampled)
    Outer Cutoff:        {traj['vd_outer_cutoff']*100:.0f}% (transition end)
    Outer Density:       {traj['vd_outer_density']*100:.0f}% (at edge)

    GRADIENT PERFORMANCE:
    Max Gradient:        {np.max(np.sqrt(traj['gx']**2 + traj['gy']**2)):.1f} mT/m (of 22.0)
    Max Slew Rate:       {np.max(slew_mag):.1f} T/m/s (of 120.0)

    K-SPACE COVERAGE:
    Max |k|:             {np.max(k_mag):.1f} 1/m
    Nyquist:             {k_nyquist:.1f} 1/m
    Coverage:            {np.max(k_mag)/k_nyquist*100:.1f}%

    TEMPORAL RESOLUTION:
    TR (per arm):        {traj['readout_time'] + 2.0:.2f} ms
    Temporal Res (16 arms): {(traj['readout_time'] + 2.0) * 16:.1f} ms
    """

    ax9.text(0.05, 0.95, param_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

def plot_gradient_comparison():
    """Compare gradients before and after rasterization."""

    # Generate trajectory
    traj = gen_spiral_traj_tfmri(
        base_resolution=240,
        field_of_view=400,
        vd_spiral_arms=16,
        vd_inner_cutoff=0.15,
        pre_vd_outer_cutoff=0.41288,
        vd_outer_density=0.07,
        vd_type='hanning',
        max_grad_ampl=22.0,
        min_rise_time=10.0,
        dwell_time=1.4,
    )

    # Convert to PyPulseq format
    k, g, s, t = convert_to_pypulseq_format(traj)

    # Raster to gradient raster time
    GRT = 10e-6
    t_grad, g_grad = raster_to_grad(g, traj['dwell_time'], GRT)

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Gradient Rasterization Comparison', fontsize=14, fontweight='bold')

    # Original gradients
    t_orig_ms = t * 1e3
    axes[0, 0].plot(t_orig_ms, g[:, 0], 'b-', linewidth=1, label='Gx')
    axes[0, 0].plot(t_orig_ms, g[:, 1], 'r-', linewidth=1, label='Gy')
    axes[0, 0].set_title(f'Original ({len(t)} points, dt={traj["dwell_time"]*1e6:.1f}µs)')
    axes[0, 0].set_xlabel('Time [ms]')
    axes[0, 0].set_ylabel('Gradient [mT/m]')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Rasterized gradients
    t_rast_ms = t_grad * 1e3
    axes[0, 1].plot(t_rast_ms, g_grad[:, 0], 'b-', linewidth=1, label='Gx')
    axes[0, 1].plot(t_rast_ms, g_grad[:, 1], 'r-', linewidth=1, label='Gy')
    axes[0, 1].set_title(f'Rasterized ({len(t_grad)} points, dt={GRT*1e6:.0f}µs)')
    axes[0, 1].set_xlabel('Time [ms]')
    axes[0, 1].set_ylabel('Gradient [mT/m]')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Magnitude comparison
    axes[1, 0].plot(t_orig_ms, np.sqrt(g[:, 0]**2 + g[:, 1]**2), 'g-',
                    linewidth=1, label='Original', alpha=0.7)
    axes[1, 0].plot(t_rast_ms, np.sqrt(g_grad[:, 0]**2 + g_grad[:, 1]**2), 'k--',
                    linewidth=1.5, label='Rasterized')
    axes[1, 0].set_title('Gradient Magnitude Comparison')
    axes[1, 0].set_xlabel('Time [ms]')
    axes[1, 0].set_ylabel('|G| [mT/m]')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    STATISTICS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Original:
      Points:     {len(t)}
      Dwell:      {traj['dwell_time']*1e6:.1f} µs
      Max |G|:    {np.max(np.sqrt(g[:, 0]**2 + g[:, 1]**2)):.2f} mT/m
      Mean |G|:   {np.mean(np.sqrt(g[:, 0]**2 + g[:, 1]**2)):.2f} mT/m

    Rasterized:
      Points:     {len(t_grad)}
      Dwell:      {GRT*1e6:.0f} µs
      Max |G|:    {np.max(np.sqrt(g_grad[:, 0]**2 + g_grad[:, 1]**2)):.2f} mT/m
      Mean |G|:   {np.mean(np.sqrt(g_grad[:, 0]**2 + g_grad[:, 1]**2)):.2f} mT/m

    Reduction:    {len(t)/len(t_grad):.1f}x
    Max Error:    {np.abs(np.max(np.sqrt(g[:, 0]**2 + g[:, 1]**2)) - np.max(np.sqrt(g_grad[:, 0]**2 + g_grad[:, 1]**2))):.3f} mT/m
    """

    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Generating HyperSLICE trajectory plots...")

    # Generate main analysis plot
    fig1 = plot_hyperslice_trajectory()
    fig1.savefig('hyperslice_trajectory_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: hyperslice_trajectory_analysis.png")

    # Generate comparison plot
    fig2 = plot_gradient_comparison()
    fig2.savefig('gradient_rasterization_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: gradient_rasterization_comparison.png")

    # Show plots if running interactively
    try:
        plt.show()
    except:
        print("Plots saved successfully. Run with display to view interactively.")