"""
classical_scattering.py
=======================
Classical Scattering Simulation — Griffiths QM Ch. 10.1

Integrates trajectories of many particles under a central potential
using 4th-order Runge-Kutta and animates the full scattering event.

Panels
------
  Left (main)   : 2D scattering view — particle trajectories + potential field
  Top-right     : V(r) radial profile
  Bottom-right  : Live polar D(θ) histogram vs analytical curve
  Bottom strip  : Summary statistics (σ, b-θ relation)

Potentials available
--------------------
  'yukawa'      : V(r) = β e^{−μr}/r        (Griffiths Example 10.5)
  'soft_sphere' : V(r) = V0   (r ≤ a)        (Griffiths Example 10.4)
  'coulomb'     : V(r) = β/r  (Rutherford)   (Griffiths Example 10.6)
  'lennard_jones': V(r) = ε[(σ/r)^12 − (σ/r)^6]  (attractive + repulsive)

Output
------
  classical_scattering.mp4  (or .gif fallback)

Usage
-----
  python classical_scattering.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patheffects as pe
from scipy.special import spherical_jn, spherical_yn

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← tweak these
# ══════════════════════════════════════════════════════════════════════

POTENTIAL     = 'yukawa'   # 'yukawa' | 'soft_sphere' | 'coulomb' | 'lennard_jones'
N_PARTICLES   = 90         # number of particles (different impact parameters)
V0            = 1.0        # particle speed (natural units, 2m = ħ = 1)

# Potential parameters
BETA          = 3.0        # Yukawa/Coulomb strength β
MU            = 0.6        # Yukawa screening length μ
SPHERE_V0     = 8.0        # Soft-sphere height V0
SPHERE_A      = 1.2        # Sphere radius a
LJ_EPS        = 1.5        # Lennard-Jones well depth ε
LJ_SIGMA      = 1.0        # Lennard-Jones σ

# Domain and impact parameters
B_MIN         = 0.05       # minimum impact parameter (avoid singularity at b=0)
B_MAX         = 4.5        # maximum impact parameter
Z_START       = -9.0       # where particles begin (far left)
Z_END         =  9.0       # where we stop tracking (far right)
X_LIMS        = (-5.5, 5.5)
Z_LIMS        = (-9.5, 9.5)

# Integration
DT            = 0.015      # RK4 time step
N_STEPS       = 900        # total integration steps per particle
STEPS_PER_FRAME = 5        # steps shown per animation frame

# Animation output
FPS           = 28
DPI           = 130
OUTPUT_FILE   = 'classical_scattering.mp4'
TRAIL_LENGTH  = 60         # how many past positions to draw as a trail

# ══════════════════════════════════════════════════════════════════════
#  PHYSICS — Potentials and Forces
# ══════════════════════════════════════════════════════════════════════

def V(r, eps=1e-6):
    """Potential energy V(r) for the selected potential."""
    r = np.maximum(r, eps)
    if POTENTIAL == 'yukawa':
        return BETA * np.exp(-MU * r) / r
    elif POTENTIAL == 'soft_sphere':
        # Smooth step: V0 / (1 + exp((r-a)/ε))
        return SPHERE_V0 / (1.0 + np.exp(10.0 * (r - SPHERE_A)))
    elif POTENTIAL == 'coulomb':
        return BETA / r
    elif POTENTIAL == 'lennard_jones':
        sr = LJ_SIGMA / r
        return 4.0 * LJ_EPS * (sr**12 - sr**6)
    return 0.0


def grad_V(x, z):
    """
    Gradient of V in Cartesian coordinates: ∇V = (dV/dr)·r̂
    Computed via finite differences for generality.
    """
    r   = np.sqrt(x**2 + z**2)
    eps = 1e-5
    r_p = np.maximum(r + eps, 1e-8)
    r_m = np.maximum(r - eps, 1e-8)
    dVdr = (V(r_p) - V(r_m)) / (2 * eps)
    # r̂ = (x, z) / r
    r_safe = np.maximum(r, 1e-8)
    return dVdr * x / r_safe, dVdr * z / r_safe


def analytical_D_theta(theta):
    """
    Analytical differential cross-section D(θ) for chosen potential.
    Returns D(θ) at given angles (array).
    """
    theta = np.asarray(theta)
    eps   = 1e-8
    if POTENTIAL == 'yukawa':
        # No simple closed form; use numerical D(θ) via b(θ) Jacobian
        return None
    elif POTENTIAL == 'coulomb':
        # Rutherford: D = (β/4E)² / sin⁴(θ/2),  E = V0²/2 (kinetic energy)
        E = 0.5 * V0**2
        return (BETA / (4 * E))**2 / (np.sin(theta / 2 + eps))**4
    elif POTENTIAL == 'soft_sphere':
        return None
    return None


# ══════════════════════════════════════════════════════════════════════
#  INTEGRATION — 4th-Order Runge-Kutta
# ══════════════════════════════════════════════════════════════════════

def rk4_step(state, dt):
    """
    One RK4 step for a particle in a central potential.
    state = [x, z, vx, vz]
    Equations: dx/dt = vx, dz/dt = vz,
               dvx/dt = -∂V/∂x,  dvz/dt = -∂V/∂z
    """
    x, z, vx, vz = state

    def deriv(s):
        _x, _z, _vx, _vz = s
        gx, gz = grad_V(_x, _z)
        return np.array([_vx, _vz, -gx, -gz])

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state +       dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_trajectory(b, v0=V0, n_steps=N_STEPS, dt=DT):
    """
    Integrate one particle trajectory starting at (x=b, z=Z_START)
    with initial velocity (vx=0, vz=v0).

    Returns
    -------
    positions : (n_steps+1, 2) array of (x, z) positions
    final_angle: asymptotic scattering angle θ (in radians)
    """
    state     = np.array([b, Z_START, 0.0, v0], dtype=float)
    positions = np.empty((n_steps + 1, 2))
    positions[0] = [b, Z_START]

    for i in range(n_steps):
        state = rk4_step(state, dt)
        positions[i + 1] = [state[0], state[1]]

    # Asymptotic velocity direction → scattering angle
    vx_f, vz_f = state[2], state[3]
    theta_f = np.arctan2(abs(vx_f), vz_f)   # angle from +z axis
    if theta_f < 0:
        theta_f += np.pi
    return positions, theta_f


# ══════════════════════════════════════════════════════════════════════
#  PRE-COMPUTE ALL TRAJECTORIES
# ══════════════════════════════════════════════════════════════════════

print(f"[Classical] Potential: {POTENTIAL}  |  N={N_PARTICLES} particles")
print(f"[Classical] Integrating {N_PARTICLES} trajectories × {N_STEPS} steps …")

b_values    = np.linspace(B_MIN, B_MAX, N_PARTICLES)
trajectories = []   # list of (n_steps+1, 2) arrays
final_angles = []   # final scattering angle for each particle

for i, b in enumerate(b_values):
    pos, theta = integrate_trajectory(b)
    trajectories.append(pos)
    final_angles.append(theta)
    if (i + 1) % 20 == 0:
        print(f"  … {i+1}/{N_PARTICLES} done")

trajectories  = np.array(trajectories)   # shape: (N, n_steps+1, 2)
final_angles  = np.array(final_angles)
N_FRAMES      = N_STEPS // STEPS_PER_FRAME
print(f"[Classical] Done. {N_FRAMES} animation frames.")

# ══════════════════════════════════════════════════════════════════════
#  POTENTIAL FIELD — for background visualisation
# ══════════════════════════════════════════════════════════════════════

xg = np.linspace(*X_LIMS, 200)
zg = np.linspace(*Z_LIMS, 200)
XG, ZG = np.meshgrid(xg, zg)
RG = np.sqrt(XG**2 + ZG**2)
VG = V(RG)
# Clip for display
VG_disp = np.clip(VG, 0, np.percentile(VG[VG > 0], 90))

# Radial profile for side panel
r_1d = np.linspace(0.05, 6.0, 600)
V_1d = V(r_1d)

# ══════════════════════════════════════════════════════════════════════
#  COLOUR MAP — particles coloured by impact parameter b
# ══════════════════════════════════════════════════════════════════════

cmap_particles = plt.cm.plasma
norm_b = Normalize(vmin=B_MIN, vmax=B_MAX)
colors = cmap_particles(norm_b(b_values))   # RGBA for each particle

# ══════════════════════════════════════════════════════════════════════
#  FIGURE LAYOUT
# ══════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 8), facecolor='#0a0a0f')
gs  = gridspec.GridSpec(
    2, 2,
    figure=fig,
    width_ratios=[2.2, 1],
    height_ratios=[1, 1],
    hspace=0.45, wspace=0.35,
    left=0.05, right=0.97,
    top=0.91, bottom=0.10,
)

ax_main  = fig.add_subplot(gs[:, 0])      # main scattering view
ax_Vr    = fig.add_subplot(gs[0, 1])      # V(r) profile
ax_theta = fig.add_subplot(gs[1, 1], projection='polar')   # D(θ) histogram

# ── Styling helper ──────────────────────────────────────────────────
def style_ax(ax, xlabel='', ylabel='', title=''):
    ax.set_facecolor('#0d0d1a')
    ax.tick_params(colors='#8b8b9e', labelsize=7)
    ax.set_xlabel(xlabel, color='#8b8b9e', fontsize=8)
    ax.set_ylabel(ylabel, color='#8b8b9e', fontsize=8)
    ax.set_title(title, color='#c9cfe0', fontsize=8.5, fontweight='bold')
    ax.grid(True, color='#1e1e32', lw=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a45')

# ── Main scattering view ─────────────────────────────────────────────
ax_main.set_facecolor('#04040c')
ax_main.set_xlim(*Z_LIMS)
ax_main.set_ylim(*X_LIMS)
ax_main.set_aspect('equal')
ax_main.tick_params(colors='#6a6a80', labelsize=7)
ax_main.set_xlabel('z  (propagation axis)', color='#6a6a80', fontsize=9)
ax_main.set_ylabel('x  (transverse)', color='#6a6a80', fontsize=9)
for sp in ax_main.spines.values():
    sp.set_edgecolor('#1e1e35')

# Potential background
cf = ax_main.contourf(ZG, XG, VG_disp, levels=20,
                      cmap='YlOrRd', alpha=0.22, zorder=1)
ax_main.contour(ZG, XG, VG_disp, levels=6,
                colors='#ff8040', alpha=0.25, linewidths=0.5, zorder=2)

# Potential circle indicator
if POTENTIAL == 'soft_sphere':
    circ = Circle((0, 0), SPHERE_A, fill=False, edgecolor='#ffa040',
                  lw=1.5, linestyle='--', alpha=0.7, zorder=8)
    ax_main.add_patch(circ)

# Incoming beam arrows (left side)
for xi in np.linspace(-B_MAX * 0.8, B_MAX * 0.8, 7):
    ax_main.annotate('', xy=(Z_START + 1.5, xi),
                     xytext=(Z_START + 0.2, xi),
                     arrowprops=dict(arrowstyle='->', color='#4444aa',
                                     lw=0.9, alpha=0.5),
                     zorder=3)

# Target indicator at origin
ax_main.plot(0, 0, 'o', color='#ff6030', ms=6, zorder=10,
             markeredgecolor='white', markeredgewidth=0.5)

# Create artists: trail lines + particle dots
trail_lines  = []
particle_dots = []
for i in range(N_PARTICLES):
    col = colors[i]
    ln, = ax_main.plot([], [], '-', color=col, lw=0.9, alpha=0.55, zorder=4)
    dot, = ax_main.plot([], [], 'o', color=col, ms=3.5, zorder=6,
                        markeredgecolor='white', markeredgewidth=0.3)
    trail_lines.append(ln)
    particle_dots.append(dot)

# Time label
time_txt = ax_main.text(
    0.02, 0.97, '', transform=ax_main.transAxes,
    color='#c0c8e0', fontsize=8, va='top',
    fontfamily='monospace',
)
# Scattered count
scat_txt = ax_main.text(
    0.02, 0.92, '', transform=ax_main.transAxes,
    color='#80e0a0', fontsize=7.5, va='top',
    fontfamily='monospace',
)

# ── V(r) profile ─────────────────────────────────────────────────────
style_ax(ax_Vr,
         xlabel='r  (distance from target)',
         ylabel='V(r)',
         title=f'Potential Profile  [{POTENTIAL}]')
ax_Vr.plot(r_1d, V_1d, color='#ff8040', lw=1.8)
ax_Vr.fill_between(r_1d, V_1d, 0,
                    where=(V_1d > 0), alpha=0.18, color='#ff8040')
ax_Vr.fill_between(r_1d, V_1d, 0,
                    where=(V_1d < 0), alpha=0.18, color='#4488ff')
ax_Vr.axhline(0, color='#444466', lw=0.6)
ax_Vr.set_xlim(0, 6)
ylim_v = max(abs(V_1d[r_1d > 0.2]).max(), 0.1) * 1.15
ax_Vr.set_ylim(-ylim_v * 0.3, ylim_v)

# Annotate with formula
formulas = {
    'yukawa':       r'$V=\beta e^{-\mu r}/r$',
    'soft_sphere':  r'$V=V_0\;\Theta(a-r)$',
    'coulomb':      r'$V=\beta/r$',
    'lennard_jones':r'$V=4\varepsilon[(\sigma/r)^{12}-(\sigma/r)^6]$',
}
ax_Vr.text(0.97, 0.95, formulas.get(POTENTIAL, ''),
           transform=ax_Vr.transAxes, color='#ffb060',
           fontsize=9, ha='right', va='top')

# ── D(θ) polar histogram ─────────────────────────────────────────────
ax_theta.set_facecolor('#0d0d1a')
ax_theta.set_theta_zero_location('N')
ax_theta.set_theta_direction(-1)
ax_theta.tick_params(colors='#6a6a80', labelsize=6)
ax_theta.set_title(r'Scattering distribution  $D(\theta)$',
                   color='#c0c8e0', fontsize=8, pad=8)
ax_theta.grid(color='#1e1e32', lw=0.5)

# Analytical D(θ) curve (if available)
theta_arr = np.linspace(0.01, np.pi, 300)
D_analytic = analytical_D_theta(theta_arr)
if D_analytic is not None:
    D_norm = D_analytic / D_analytic.max()
    ax_theta.plot(theta_arr, D_norm, color='#40ddff',
                  lw=1.5, label='Analytical', alpha=0.8, zorder=5)
    ax_theta.fill(theta_arr, D_norm, color='#40ddff', alpha=0.08)

# Live histogram bars — create as a bar container
N_BINS     = 36
bin_edges  = np.linspace(0, np.pi, N_BINS + 1)
bin_width  = bin_edges[1] - bin_edges[0]
bin_centers= 0.5 * (bin_edges[:-1] + bin_edges[1:])
bar_heights= np.zeros(N_BINS)
bars       = ax_theta.bar(bin_centers, bar_heights, width=bin_width,
                           color='#c060ff', alpha=0.65,
                           edgecolor='#8030cc', linewidth=0.4,
                           bottom=0, zorder=4)

# b–θ scatter: small dots added as scatter is computed
btheta_scatter = ax_theta.scatter([], [], s=2, color='#ffcc40',
                                   alpha=0.4, zorder=3)
scattered_angles_list = []

# Colorbar for impact parameter
sm = plt.cm.ScalarMappable(cmap=cmap_particles, norm=norm_b)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_main, fraction=0.018, pad=0.02, shrink=0.6)
cbar.set_label('Impact parameter  b', color='#8b8b9e', fontsize=7)
cbar.ax.tick_params(colors='#8b8b9e', labelsize=6)
cbar.outline.set_edgecolor('#2a2a45')

# Main title
pot_labels = {
    'yukawa':       f'Yukawa  (β={BETA}, μ={MU})',
    'soft_sphere':  f'Soft Sphere  (V₀={SPHERE_V0}, a={SPHERE_A})',
    'coulomb':      f'Coulomb/Rutherford  (β={BETA})',
    'lennard_jones':f'Lennard-Jones  (ε={LJ_EPS}, σ={LJ_SIGMA})',
}
fig.suptitle(
    f'Classical Scattering  —  {pot_labels.get(POTENTIAL,"")}\n'
    f'{N_PARTICLES} particles  |  v₀ = {V0:.1f}  |  '
    r'$D(\theta) = \frac{b}{\sin\theta}\left|\frac{db}{d\theta}\right|$',
    color='white', fontsize=11, fontweight='bold', y=0.97,
)

# ══════════════════════════════════════════════════════════════════════
#  ANIMATION
# ══════════════════════════════════════════════════════════════════════

def init():
    for ln in trail_lines:
        ln.set_data([], [])
    for dot in particle_dots:
        dot.set_data([], [])
    time_txt.set_text('')
    scat_txt.set_text('')
    return trail_lines + particle_dots + [time_txt, scat_txt]


def update(frame_idx):
    step = frame_idx * STEPS_PER_FRAME
    t    = step * DT

    # How many particles have "escaped" (reached far right or far top/bottom)
    n_scattered_this_frame = 0

    for i in range(N_PARTICLES):
        traj = trajectories[i]                   # (N_STEPS+1, 2)
        current_step = min(step, N_STEPS)

        # Current position
        cx, cz = traj[current_step, 0], traj[current_step, 1]

        # Trail: last TRAIL_LENGTH positions
        t_start = max(0, current_step - TRAIL_LENGTH)
        trail_x = traj[t_start:current_step + 1, 0]
        trail_z = traj[t_start:current_step + 1, 1]

        # Clip to domain for display
        trail_lines[i].set_data(trail_z, trail_x)
        particle_dots[i].set_data([cz], [cx])

        # Check if particle has clearly scattered (far from origin)
        r_final = np.sqrt(cx**2 + cz**2)
        if r_final > 5.0 and cz > 0:  # outgoing
            n_scattered_this_frame += 1

    # Update D(θ) histogram with all particles that have already "finished"
    # (reached right boundary or escaped the domain)
    finished_angles = []
    for i in range(N_PARTICLES):
        traj = trajectories[i]
        cx, cz = traj[min(step, N_STEPS), 0], traj[min(step, N_STEPS), 1]
        r_now  = np.sqrt(cx**2 + cz**2)
        # Particle is "finished" if it's in the far zone and moving outward
        if r_now > 6.0:
            finished_angles.append(final_angles[i])

    if finished_angles:
        counts, _ = np.histogram(finished_angles, bins=bin_edges)
        max_c = max(counts.max(), 1)
        for bar, h in zip(bars, counts / max_c):
            bar.set_height(h)

        # Update b-θ scatter
        pairs = [(b_values[i], final_angles[i])
                 for i in range(N_PARTICLES)
                 if np.sqrt(trajectories[i][min(step, N_STEPS), 0]**2 +
                            trajectories[i][min(step, N_STEPS), 1]**2) > 6.0]
        if pairs:
            btheta_scatter.set_offsets(
                np.column_stack([[p[0] for p in pairs],
                                 [p[1] for p in pairs]])
            )

    # Labels
    time_txt.set_text(f't = {t:.2f}   step {step}/{N_STEPS}')
    n_done = sum(
        1 for i in range(N_PARTICLES)
        if np.sqrt(trajectories[i][min(step, N_STEPS), 0]**2 +
                   trajectories[i][min(step, N_STEPS), 1]**2) > 6.0
    )
    scat_txt.set_text(f'Scattered: {n_done}/{N_PARTICLES}')

    return (trail_lines + particle_dots +
            list(bars) + [time_txt, scat_txt, btheta_scatter])


print(f"[Classical] Building animation ({N_FRAMES} frames) …")
anim = animation.FuncAnimation(
    fig, update,
    frames   = N_FRAMES,
    init_func= init,
    interval = 1000 // FPS,
    blit     = False,    # blit=False for polar axis compatibility
)

# ══════════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════════

def save_animation(anim, path, fps, dpi):
    if path.endswith('.mp4') and animation.FFMpegWriter.isAvailable():
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=2200,
            metadata={'title': 'Classical Scattering', 'artist': 'QS Explorer'},
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'],
        )
        anim.save(path, writer=writer, dpi=dpi,
                  savefig_kwargs={'facecolor': '#0a0a0f'})
        print(f"[Classical] Saved: {path}")
    else:
        gif_path = path.replace('.mp4', '.gif')
        writer   = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=dpi,
                  savefig_kwargs={'facecolor': '#0a0a0f'})
        print(f"[Classical] Saved (GIF fallback): {gif_path}")


save_animation(anim, OUTPUT_FILE, FPS, DPI)
plt.close(fig)
print("[Classical] Done.")
