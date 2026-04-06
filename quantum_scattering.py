"""
quantum_scattering.py
=====================
Quantum Scattering Simulation — Griffiths QM Ch. 10.2–10.4

Solves the time-dependent Schrödinger equation in 2D using the
SPLIT-OPERATOR FFT method (Strang splitting, 2nd-order accurate):

    ψ(t + dt) = exp(−iV dt/2) · IFFT[exp(−ik²dt/2m) · FFT[exp(−iV dt/2) · ψ(t)]]

An initial Gaussian wave packet (modelling a quantum "particle") travels
toward a scattering potential and produces a scattered spherical wave —
exactly the structure in Griffiths Eq. 10.12.

Panels
------
  Top-left (large)  : |ψ(x,z)|²  — probability density  (main view)
  Top-right         : Re[ψ(x,z)] — wave pattern (shows interference)
  Bottom-left       : Phase angle ∠ψ(x,z)
  Bottom-right      : P_reflected(t), P_transmitted(t), P_total(t) — live plot

Potentials
----------
  'hard_sphere'  : Step wall V₀ ≫ 0  for r < a     (Griffiths Example 10.3)
  'soft_sphere'  : Gaussian bump V₀ exp(−r²/2σ²)   (Griffiths Example 10.4)
  'double_slit'  : Two openings in a hard wall       (quantum diffraction)
  'yukawa'       : V₀ exp(−μr)/r                    (Griffiths §10.4.2)

Natural units:  ħ = 1,  m = 1  →  E = k²/2,  v_group = k

Output
------
  quantum_scattering.mp4  (or .gif fallback)

Usage
-----
  python quantum_scattering.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.patheffects as pe

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← tweak these
# ══════════════════════════════════════════════════════════════════════

POTENTIAL       = 'hard_sphere'   # 'hard_sphere'|'soft_sphere'|'double_slit'|'yukawa'

# Grid
NX, NZ          = 384, 512        # grid points (power of 2 preferred for FFT)
LX, LZ          = 9.0, 13.0      # half-domain: x ∈ [−LX, LX], z ∈ [−LZ, LZ]

# Initial wave packet  (Griffiths: ψ_incident = A e^{ikz})
K0              = 5.0             # central wave number  k₀  (momentum)
SIGMA_X         = 0.9             # spatial width along x
SIGMA_Z         = 1.1             # spatial width along z (parallel to motion)
X0, Z0          = 0.0, -8.5      # initial packet centre

# Potential parameters
SPHERE_A        = 1.5             # hard/soft sphere radius
SPHERE_V0       = 600.0           # hard-sphere height (must be >> E = K0²/2)
SOFT_V0         = 30.0            # soft sphere / Yukawa height
SOFT_SIGMA      = 1.0             # Gaussian width for soft sphere
YUKAWA_BETA     = 8.0             # Yukawa strength β
YUKAWA_MU       = 0.6             # Yukawa screening μ
SLIT_WALL_Z     = 0.0             # z position of the double-slit wall
SLIT_HALF_GAP   = 0.55           # half-width of each slit opening
SLIT_SEPARATION = 2.0             # centre-to-centre slit separation
SLIT_THICKNESS  = 0.4             # wall thickness
SLIT_V0         = 800.0           # slit wall height

# Time evolution
DT              = 0.004           # time step (must satisfy k_max²·dt/2 < π for accuracy)
STEPS_PER_FRAME = 10              # split-operator steps between drawn frames
N_FRAMES        = 210             # total animation frames

# Absorbing boundary (prevents unphysical reflections from grid edges)
ABSORB_WIDTH    = 2.2             # width of absorbing strip at each edge
ABSORB_STRENGTH = 18.0            # Γ_max (larger = stronger absorption)

# Output
FPS             = 26
DPI             = 130
OUTPUT_FILE     = 'quantum_scattering.mp4'

# ══════════════════════════════════════════════════════════════════════
#  GRID SETUP
# ══════════════════════════════════════════════════════════════════════

# Position grids
x_1d   = np.linspace(-LX, LX, NX, endpoint=False)
z_1d   = np.linspace(-LZ, LZ, NZ, endpoint=False)
dx, dz = x_1d[1] - x_1d[0], z_1d[1] - z_1d[0]
X, Z   = np.meshgrid(x_1d, z_1d, indexing='ij')   # shape: (NX, NZ)
R      = np.sqrt(X**2 + Z**2)

# Wave-number grids (FFT convention)
# kx ∈ [−π/dx, π/dx),  kz ∈ [−π/dz, π/dz)
kx_1d  = 2 * np.pi * np.fft.fftfreq(NX, d=dx)
kz_1d  = 2 * np.pi * np.fft.fftfreq(NZ, d=dz)
KX, KZ = np.meshgrid(kx_1d, kz_1d, indexing='ij')
K2     = KX**2 + KZ**2     # |k|² (used in kinetic phase)

print(f"[Quantum]  Grid: {NX}×{NZ}  |  dx={dx:.3f}, dz={dz:.3f}")
print(f"[Quantum]  k_max = {np.sqrt(K2.max()):.1f}  |  "
      f"k₀ = {K0}  |  E = {K0**2/2:.2f}")
print(f"[Quantum]  dt={DT}  |  steps/frame={STEPS_PER_FRAME}  |  "
      f"frames={N_FRAMES}")

# ══════════════════════════════════════════════════════════════════════
#  POTENTIAL V(x, z)
# ══════════════════════════════════════════════════════════════════════

def build_potential():
    """Build the scattering potential V(x,z) on the 2D grid."""
    V = np.zeros((NX, NZ), dtype=float)

    if POTENTIAL == 'hard_sphere':
        # Step potential: V = V0 inside sphere, 0 outside
        # Smoothed slightly with tanh to reduce Gibbs ringing
        V = SPHERE_V0 / (1.0 + np.exp(12.0 * (R - SPHERE_A)))

    elif POTENTIAL == 'soft_sphere':
        # Gaussian repulsive bump  V = V0 exp(−r²/2σ²)
        V = SOFT_V0 * np.exp(-R**2 / (2 * SOFT_SIGMA**2))

    elif POTENTIAL == 'double_slit':
        # Hard wall at z = SLIT_WALL_Z with two openings
        wall_mask = np.abs(Z - SLIT_WALL_Z) < SLIT_THICKNESS / 2
        # Slit openings: two gaps centred at ±SLIT_SEPARATION/2
        open1 = np.abs(X - SLIT_SEPARATION / 2) < SLIT_HALF_GAP
        open2 = np.abs(X + SLIT_SEPARATION / 2) < SLIT_HALF_GAP
        slit_mask = wall_mask & ~(open1 | open2)
        V[slit_mask] = SLIT_V0

    elif POTENTIAL == 'yukawa':
        # Yukawa: V = β exp(−μr)/r  (regularised at origin)
        r_safe = np.maximum(R, 0.05)
        V = YUKAWA_BETA * np.exp(-YUKAWA_MU * r_safe) / r_safe

    return V


# ══════════════════════════════════════════════════════════════════════
#  ABSORBING BOUNDARY LAYER   (complex absorbing potential)
# ══════════════════════════════════════════════════════════════════════

def build_absorber():
    """
    Γ(x,z) ≥ 0 — imaginary part of the complex potential.
    Applied as an additional damping:  ψ → ψ · exp(−Γ dt/2)
    Large Γ near boundaries absorbs outgoing waves without reflection.
    """
    Gamma = np.zeros((NX, NZ), dtype=float)

    def _ramp(coord, L, d):
        """Smooth quadratic ramp from 0 to Γ_max within distance d of edge."""
        dist = np.maximum(np.abs(coord) - (L - d), 0.0) / d
        return ABSORB_STRENGTH * dist**2

    Gamma += _ramp(X, LX, ABSORB_WIDTH)
    Gamma += _ramp(Z, LZ, ABSORB_WIDTH)
    return Gamma


# ══════════════════════════════════════════════════════════════════════
#  PRECOMPUTE PHASE FACTORS  (done once, reused every step)
# ══════════════════════════════════════════════════════════════════════

V_pot  = build_potential()
Gamma  = build_absorber()

# Half-step potential phase:  exp(−iV·dt/2) · exp(−Γ·dt/2)
half_V_phase = np.exp(-1j * V_pot * DT / 2) * np.exp(-Gamma * DT / 2)

# Full kinetic phase in k-space:  exp(−ik²/(2m) · dt)  [m=1, ħ=1]
kin_phase = np.exp(-1j * (K2 / 2) * DT)

print(f"[Quantum]  Potential '{POTENTIAL}' built.  "
      f"max V = {V_pot.max():.1f},  max Γ = {Gamma.max():.1f}")

# ══════════════════════════════════════════════════════════════════════
#  INITIAL WAVE PACKET   ψ₀ = A exp(−x²/4σx² − (z−z₀)²/4σz²) exp(ik₀z)
# ══════════════════════════════════════════════════════════════════════

psi = (np.exp(-((X - X0)**2 / (4 * SIGMA_X**2))
              -((Z - Z0)**2 / (4 * SIGMA_Z**2)))
       * np.exp(1j * K0 * Z))

# Normalise  ∫|ψ|² dx dz = 1
norm = np.sqrt((np.abs(psi)**2).sum() * dx * dz)
psi /= norm

print(f"[Quantum]  Wave packet: k₀={K0}, σx={SIGMA_X}, σz={SIGMA_Z}, "
      f"z₀={Z0}")
print(f"[Quantum]  Group velocity v_g = k₀/m = {K0:.1f}  →  "
      f"arrival at origin ≈ t = {abs(Z0)/K0:.2f}")

# ══════════════════════════════════════════════════════════════════════
#  SPLIT-OPERATOR STEP
# ══════════════════════════════════════════════════════════════════════

def split_operator_step(psi):
    """
    One Strang-split time step (2nd-order accurate):
      1.  Half V-step  : ψ₁ = exp(−iV dt/2) · ψ
      2.  Full T-step  : ψ₂ = IFFT[exp(−ik²dt/2) · FFT(ψ₁)]
      3.  Half V-step  : ψ₃ = exp(−iV dt/2) · ψ₂

    The imaginary part of V (absorber Γ) damps near boundaries.
    """
    psi1 = half_V_phase * psi                      # step 1
    psi2 = np.fft.ifftn(kin_phase * np.fft.fftn(psi1))   # step 2
    psi3 = half_V_phase * psi2                     # step 3
    return psi3


# ══════════════════════════════════════════════════════════════════════
#  OBSERVABLES
# ══════════════════════════════════════════════════════════════════════

def probability_current(psi):
    """
    J(x,z) = (ħ/m) Im[ψ* ∇ψ]  =  Im[ψ* ∇ψ]  (natural units ħ=m=1)
    Computed with central differences.
    """
    dpsi_dz = np.gradient(psi, dz, axis=1)
    dpsi_dx = np.gradient(psi, dx, axis=0)
    Jx = np.imag(np.conj(psi) * dpsi_dx)
    Jz = np.imag(np.conj(psi) * dpsi_dz)
    return Jx, Jz


def compute_probabilities(psi):
    """
    Returns:
      P_total      : total probability (should be ~1 early, decreases with absorber)
      P_left       : probability in z < 0 half-space  (incident + reflected)
      P_right      : probability in z > 0 half-space  (transmitted + forward)
      P_scattered  : probability outside the beam axis |x| > 1
    """
    prob     = np.abs(psi)**2 * dx * dz
    P_total  = prob.sum() * dx * dz
    P_left   = prob[:, z_1d < 0].sum() * dx * dz
    P_right  = prob[:, z_1d >= 0].sum() * dx * dz
    P_scat   = prob[np.abs(x_1d) > 1.5, :].sum() * dx * dz
    return P_total, P_left, P_right, P_scat


# ══════════════════════════════════════════════════════════════════════
#  FIGURE LAYOUT
# ══════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 9), facecolor='#04040c')
gs  = gridspec.GridSpec(
    2, 3,
    figure=fig,
    width_ratios=[2.2, 1.0, 1.0],
    height_ratios=[1.0, 1.0],
    hspace=0.42, wspace=0.38,
    left=0.04, right=0.97,
    top=0.91, bottom=0.09,
)

ax_psi2  = fig.add_subplot(gs[:, 0])      # |ψ|²  main panel
ax_repsi = fig.add_subplot(gs[0, 1])      # Re[ψ]
ax_phase = fig.add_subplot(gs[0, 2])      # phase angle ∠ψ
ax_prob  = fig.add_subplot(gs[1, 1:])     # probability vs time

# ── Axis styling ─────────────────────────────────────────────────────

def style_img_ax(ax, title):
    ax.set_facecolor('#04040c')
    ax.tick_params(colors='#5a5a72', labelsize=6.5)
    ax.set_xlabel('z', color='#5a5a72', fontsize=8)
    ax.set_ylabel('x', color='#5a5a72', fontsize=8)
    ax.set_title(title, color='#c0c8e0', fontsize=9, fontweight='bold')

def style_ax(ax, xlabel='', ylabel='', title=''):
    ax.set_facecolor('#0a0a18')
    ax.tick_params(colors='#5a5a72', labelsize=7)
    ax.set_xlabel(xlabel, color='#5a5a72', fontsize=8)
    ax.set_ylabel(ylabel, color='#5a5a72', fontsize=8)
    ax.set_title(title, color='#c0c8e0', fontsize=9, fontweight='bold')
    ax.grid(True, color='#131326', lw=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#1a1a30')

style_img_ax(ax_psi2,  r'$|\psi(x,z,t)|^2$  — Probability Density')
style_img_ax(ax_repsi, r'Re$[\psi(x,z,t)]$  — Wave Pattern')
style_img_ax(ax_phase, r'Phase  $\angle\psi(x,z,t)$')
style_ax(ax_prob,
         xlabel='Time  t',
         ylabel='Probability',
         title='Probability Tracking')

# Aspect and extent
extent = [-LZ, LZ, -LX, LX]   # [z_min, z_max, x_min, x_max]

# ── |ψ|² image ───────────────────────────────────────────────────────
prob0  = np.abs(psi)**2
vmax0  = prob0.max() * 0.9
im_psi = ax_psi2.imshow(
    prob0, extent=extent, origin='lower',
    cmap='inferno', vmin=0, vmax=vmax0,
    aspect='auto', interpolation='bilinear',
)
cb_psi = fig.colorbar(im_psi, ax=ax_psi2, fraction=0.020, pad=0.02)
cb_psi.set_label(r'$|\psi|^2$', color='#8b8b9e', fontsize=7)
cb_psi.ax.tick_params(colors='#8b8b9e', labelsize=6)
cb_psi.outline.set_edgecolor('#2a2a45')

# Potential boundary contour (will be redrawn — create a placeholder line)
if POTENTIAL in ('hard_sphere',):
    th  = np.linspace(0, 2*np.pi, 200)
    ax_psi2.plot(SPHERE_A * np.cos(th), SPHERE_A * np.sin(th),
                 color='white', lw=1.0, alpha=0.5, zorder=5)
    ax_repsi.plot(SPHERE_A * np.cos(th), SPHERE_A * np.sin(th),
                  color='white', lw=0.8, alpha=0.4, zorder=5)

# Probability current arrows (sparse grid)
step_x = NX // 18
step_z = NZ // 22
Xq = X[::step_x, ::step_z]
Zq = Z[::step_x, ::step_z]
Jx0, Jz0 = probability_current(psi)
Jxq = Jx0[::step_x, ::step_z]
Jzq = Jz0[::step_x, ::step_z]
J_mag = np.sqrt(Jxq**2 + Jzq**2) + 1e-30
J_scale = np.percentile(J_mag, 90)
quiv = ax_psi2.quiver(
    Zq, Xq, Jzq / J_scale, Jxq / J_scale,
    alpha=0.45, color='#00ccff', scale=22, width=0.0025,
    headwidth=4, headlength=5, zorder=6,
)

# Time annotation
time_txt = ax_psi2.text(
    0.02, 0.97, 't = 0.00', transform=ax_psi2.transAxes,
    color='white', fontsize=8.5, va='top',
    fontfamily='monospace', zorder=8,
)
norm_txt = ax_psi2.text(
    0.02, 0.92, 'P_total = 1.000', transform=ax_psi2.transAxes,
    color='#80e0a0', fontsize=7.5, va='top',
    fontfamily='monospace', zorder=8,
)

# ── Re[ψ] image ──────────────────────────────────────────────────────
re0   = psi.real
vrmax = np.percentile(np.abs(re0), 99) + 1e-12
im_re = ax_repsi.imshow(
    re0, extent=extent, origin='lower',
    cmap='RdBu_r', vmin=-vrmax, vmax=vrmax,
    aspect='auto', interpolation='bilinear',
)

# ── Phase image ───────────────────────────────────────────────────────
ph0    = np.angle(psi)
im_ph  = ax_phase.imshow(
    ph0, extent=extent, origin='lower',
    cmap='hsv', vmin=-np.pi, vmax=np.pi,
    aspect='auto', interpolation='bilinear',
)
cb_ph = fig.colorbar(im_ph, ax=ax_phase, fraction=0.046, pad=0.04,
                     ticks=[-np.pi, 0, np.pi])
cb_ph.ax.set_yticklabels(['-π', '0', '+π'], fontsize=6, color='#8b8b9e')
cb_ph.outline.set_edgecolor('#2a2a45')

# ── Probability time-series ───────────────────────────────────────────
ax_prob.set_facecolor('#0a0a18')
times_list = [0.0]
Ptot_list  = [1.0]
Pleft_list = [0.0]
Prigh_list = [0.0]
Pscat_list = [0.0]

line_tot,  = ax_prob.plot([], [], color='white',    lw=1.5, label='$P_{total}$')
line_left, = ax_prob.plot([], [], color='#ff6040',  lw=1.4,
                           label='$P_{z<0}$  (incident + reflected)',
                           linestyle='--')
line_righ, = ax_prob.plot([], [], color='#40ddaa',  lw=1.4,
                           label='$P_{z>0}$  (transmitted)',
                           linestyle='-.')
line_scat, = ax_prob.plot([], [], color='#c060ff',  lw=1.2,
                           label=r'$P_{|x|>1.5}$  (scattered)',
                           linestyle=':')

ax_prob.set_xlim(0, N_FRAMES * STEPS_PER_FRAME * DT)
ax_prob.set_ylim(-0.05, 1.15)
ax_prob.legend(fontsize=6.5, facecolor='#0a0a18', edgecolor='#1a1a30',
               labelcolor='#c0c8e0', loc='upper right')
style_ax(ax_prob, xlabel='Time  t', ylabel='Probability',
         title='Probability Tracking')

# Annotate the optical theorem result in the prob panel
ax_prob.text(0.02, 0.97,
             r'$\psi\approx A\left[e^{ikz}+f(\theta)\frac{e^{ikr}}{r}\right]$'
             '\n'
             r'$D(\theta)=|f(\theta)|^2$',
             transform=ax_prob.transAxes,
             color='#60a0ff', fontsize=7.5, va='top',
             bbox=dict(fc='#0a0a18', ec='#1a1a30', boxstyle='round,pad=0.4',
                       alpha=0.8))

# Main title
pot_labels = {
    'hard_sphere':  f'Hard Sphere  (a={SPHERE_A}, V₀={SPHERE_V0:.0f})',
    'soft_sphere':  f'Soft Sphere  (σ={SOFT_SIGMA}, V₀={SOFT_V0})',
    'double_slit':  f'Double Slit  (d={2*SLIT_HALF_GAP:.2f}, sep={SLIT_SEPARATION})',
    'yukawa':       f'Yukawa  (β={YUKAWA_BETA}, μ={YUKAWA_MU})',
}
fig.suptitle(
    f'Quantum Scattering  —  {pot_labels.get(POTENTIAL,"")}\n'
    r'$i\hbar\partial_t\psi = \left[-\frac{\hbar^2}{2m}\nabla^2 + V\right]\psi$'
    f'   |   $k_0={K0}$,  $E=k_0^2/2={K0**2/2:.1f}$,  '
    r'$\lambda=2\pi/k_0=$'
    f'{2*np.pi/K0:.2f}',
    color='white', fontsize=10.5, fontweight='bold', y=0.975,
)

# ══════════════════════════════════════════════════════════════════════
#  ANIMATION
# ══════════════════════════════════════════════════════════════════════

# We keep a mutable reference to psi
state = {'psi': psi.copy(), 'frame': 0}

# Find a good normalisation for |ψ|² from early frames
_psi_ref = psi.copy()
for _ in range(30):
    _psi_ref = split_operator_step(_psi_ref)
_vmax_prob = (np.abs(_psi_ref)**2).max() * 1.3
del _psi_ref


def update(frame_idx):
    psi = state['psi']

    # Advance STEPS_PER_FRAME split-operator steps
    for _ in range(STEPS_PER_FRAME):
        psi = split_operator_step(psi)

    state['psi']  = psi
    state['frame']= frame_idx + 1

    t = (frame_idx + 1) * STEPS_PER_FRAME * DT

    # ── Update images ────────────────────────────────────────────────
    prob  = np.abs(psi)**2
    re_   = psi.real
    phase_= np.angle(psi)

    # Dynamically adapt vmax to always show structure clearly
    vmax_p = max(prob.max() * 0.92, 1e-10)
    im_psi.set_data(prob)
    im_psi.set_clim(0, vmax_p)

    vmax_r = max(np.percentile(np.abs(re_), 99.0), 1e-10)
    im_re.set_data(re_)
    im_re.set_clim(-vmax_r, vmax_r)

    im_ph.set_data(phase_)

    # ── Probability current arrows ────────────────────────────────────
    Jx, Jz = probability_current(psi)
    Jxq = Jx[::step_x, ::step_z]
    Jzq = Jz[::step_x, ::step_z]
    J_m = np.sqrt(Jxq**2 + Jzq**2) + 1e-30
    J_s = max(np.percentile(J_m, 93), 1e-12)
    quiv.set_UVC(Jzq / J_s, Jxq / J_s)

    # ── Probabilities ─────────────────────────────────────────────────
    P_tot, P_left, P_right, P_scat = compute_probabilities(psi)
    times_list.append(t)
    Ptot_list.append(float(P_tot))
    Pleft_list.append(float(P_left))
    Prigh_list.append(float(P_right))
    Pscat_list.append(float(P_scat))

    line_tot.set_data(times_list,  Ptot_list)
    line_left.set_data(times_list, Pleft_list)
    line_righ.set_data(times_list, Prigh_list)
    line_scat.set_data(times_list, Pscat_list)
    ax_prob.set_xlim(0, max(times_list[-1] * 1.02, 0.1))

    # ── Labels ────────────────────────────────────────────────────────
    time_txt.set_text(f't = {t:.2f}  |  frame {frame_idx+1}/{N_FRAMES}')
    norm_txt.set_text(f'P_total = {P_tot:.3f}')

    return [im_psi, im_re, im_ph, quiv,
            line_tot, line_left, line_righ, line_scat,
            time_txt, norm_txt]


print(f"\n[Quantum]  Building animation ({N_FRAMES} frames, "
      f"{STEPS_PER_FRAME} steps/frame) …")
print(f"[Quantum]  Total simulated time: "
      f"{N_FRAMES * STEPS_PER_FRAME * DT:.2f} time units")

anim = animation.FuncAnimation(
    fig, update,
    frames   = N_FRAMES,
    interval = 1000 // FPS,
    blit     = False,
)

# ══════════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════════

def save_animation(anim, path, fps, dpi):
    if path.endswith('.mp4') and animation.FFMpegWriter.isAvailable():
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=2800,
            metadata={'title': 'Quantum Scattering', 'artist': 'QS Explorer'},
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'],
        )
        anim.save(path, writer=writer, dpi=dpi,
                  savefig_kwargs={'facecolor': '#04040c'})
        print(f"[Quantum]  Saved: {path}")
    else:
        gif_path = path.replace('.mp4', '.gif')
        writer   = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=max(dpi // 2, 72),
                  savefig_kwargs={'facecolor': '#04040c'})
        print(f"[Quantum]  Saved (GIF fallback): {gif_path}")


save_animation(anim, OUTPUT_FILE, FPS, DPI)
plt.close(fig)
print("[Quantum]  Done.")
