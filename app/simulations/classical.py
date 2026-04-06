from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from matplotlib.colors import Normalize
from matplotlib.patches import Circle


ProgressCallback = Optional[Callable[[int, str], None]]


def _configure_ffmpeg() -> None:
    try:
        matplotlib.rcParams["animation.ffmpeg_path"] = get_ffmpeg_exe()
    except Exception:
        # Keep matplotlib defaults if imageio-ffmpeg is unavailable.
        pass


@dataclass
class ClassicalParams:
    potential: str = "yukawa"
    n_particles: int = 90
    v0: float = 1.0
    beta: float = 3.0
    mu: float = 0.6
    sphere_v0: float = 8.0
    sphere_a: float = 1.2
    lj_eps: float = 1.5
    lj_sigma: float = 1.0
    b_min: float = 0.05
    b_max: float = 4.5
    z_start: float = -9.0
    x_lims: tuple[float, float] = (-5.5, 5.5)
    z_lims: tuple[float, float] = (-9.5, 9.5)
    dt: float = 0.015
    total_time: float = 13.5
    steps_per_frame: int = 5
    fps: int = 28
    dpi: int = 180
    trail_length: int = 60
    mass: float = 1.0


def _emit(progress_cb: ProgressCallback, percent: int, message: str) -> None:
    if progress_cb is not None:
        progress_cb(max(0, min(100, int(percent))), message)


def run_classical(
    params: Dict[str, float | int | str],
    output_path: str,
    progress_cb: ProgressCallback = None,
) -> str:
    _configure_ffmpeg()
    p = ClassicalParams(**params)
    n_steps = max(1, int(p.total_time / p.dt))
    n_frames = max(1, n_steps // p.steps_per_frame)

    def V(r: np.ndarray | float, eps: float = 1e-6) -> np.ndarray | float:
        r = np.maximum(r, eps)
        if p.potential == "yukawa":
            return p.beta * np.exp(-p.mu * r) / r
        if p.potential == "soft_sphere":
            return p.sphere_v0 / (1.0 + np.exp(10.0 * (r - p.sphere_a)))
        if p.potential == "coulomb":
            return p.beta / r
        if p.potential == "lennard_jones":
            sr = p.lj_sigma / r
            return 4.0 * p.lj_eps * (sr**12 - sr**6)
        return np.zeros_like(r)

    def grad_V(x: float, z: float) -> tuple[float, float]:
        r = np.sqrt(x**2 + z**2)
        eps = 1e-5
        r_p = max(r + eps, 1e-8)
        r_m = max(r - eps, 1e-8)
        dVdr = (V(r_p) - V(r_m)) / (2 * eps)
        r_safe = max(r, 1e-8)
        return float(dVdr * x / r_safe), float(dVdr * z / r_safe)

    def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
        def deriv(s: np.ndarray) -> np.ndarray:
            _x, _z, _vx, _vz = s
            gx, gz = grad_V(float(_x), float(_z))
            return np.array([_vx, _vz, -gx / p.mass, -gz / p.mass], dtype=float)

        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def integrate_trajectory(b: float) -> tuple[np.ndarray, float]:
        state = np.array([b, p.z_start, 0.0, p.v0], dtype=float)
        positions = np.empty((n_steps + 1, 2), dtype=float)
        positions[0] = [b, p.z_start]
        for i in range(n_steps):
            state = rk4_step(state, p.dt)
            positions[i + 1] = [state[0], state[1]]
        theta_f = np.arctan2(abs(state[2]), state[3])
        if theta_f < 0:
            theta_f += np.pi
        return positions, float(theta_f)

    _emit(progress_cb, 1, "Integrating classical trajectories...")
    b_values = np.linspace(p.b_min, p.b_max, p.n_particles)
    trajectories = []
    final_angles = []
    for idx, b in enumerate(b_values):
        pos, theta = integrate_trajectory(float(b))
        trajectories.append(pos)
        final_angles.append(theta)
        pct = 1 + int(54 * (idx + 1) / p.n_particles)
        _emit(progress_cb, pct, f"Integrated {idx + 1}/{p.n_particles} trajectories")

    trajectories = np.array(trajectories)
    final_angles = np.array(final_angles)

    _emit(progress_cb, 60, "Building classical animation canvas...")
    fig = plt.figure(figsize=(14, 8), facecolor="#0a0a0f")
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[2.2, 1],
        height_ratios=[1, 1],
        hspace=0.45,
        wspace=0.35,
        left=0.05,
        right=0.97,
        top=0.91,
        bottom=0.10,
    )

    ax_main = fig.add_subplot(gs[:, 0])
    ax_Vr = fig.add_subplot(gs[0, 1])
    ax_theta = fig.add_subplot(gs[1, 1], projection="polar")

    xg = np.linspace(*p.x_lims, 200)
    zg = np.linspace(*p.z_lims, 200)
    XG, ZG = np.meshgrid(xg, zg)
    RG = np.sqrt(XG**2 + ZG**2)
    VG = V(RG)
    positive = VG[VG > 0]
    clip_hi = np.percentile(positive, 90) if positive.size else max(float(np.max(VG)), 1.0)
    VG_disp = np.clip(VG, np.min(VG), clip_hi)

    r_1d = np.linspace(0.05, 6.0, 600)
    V_1d = V(r_1d)

    cmap_particles = plt.cm.plasma
    norm_b = Normalize(vmin=p.b_min, vmax=p.b_max)
    colors = cmap_particles(norm_b(b_values))

    ax_main.set_facecolor("#04040c")
    ax_main.set_xlim(*p.z_lims)
    ax_main.set_ylim(*p.x_lims)
    ax_main.set_aspect("equal")
    ax_main.tick_params(colors="#6a6a80", labelsize=7)
    ax_main.set_xlabel("z  (propagation axis)", color="#6a6a80", fontsize=9)
    ax_main.set_ylabel("x  (transverse)", color="#6a6a80", fontsize=9)

    ax_main.contourf(ZG, XG, VG_disp, levels=20, cmap="YlOrRd", alpha=0.22, zorder=1)
    ax_main.contour(ZG, XG, VG_disp, levels=6, colors="#ff8040", alpha=0.25, linewidths=0.5, zorder=2)

    if p.potential == "soft_sphere":
        ax_main.add_patch(
            Circle((0, 0), p.sphere_a, fill=False, edgecolor="#ffa040", lw=1.5, linestyle="--", alpha=0.7, zorder=8)
        )

    for xi in np.linspace(-p.b_max * 0.8, p.b_max * 0.8, 7):
        ax_main.annotate(
            "",
            xy=(p.z_start + 1.5, xi),
            xytext=(p.z_start + 0.2, xi),
            arrowprops=dict(arrowstyle="->", color="#4444aa", lw=0.9, alpha=0.5),
            zorder=3,
        )

    ax_main.plot(0, 0, "o", color="#ff6030", ms=6, zorder=10, markeredgecolor="white", markeredgewidth=0.5)

    trail_lines = []
    particle_dots = []
    for i in range(p.n_particles):
        col = colors[i]
        ln, = ax_main.plot([], [], "-", color=col, lw=0.9, alpha=0.55, zorder=4)
        dot, = ax_main.plot([], [], "o", color=col, ms=3.5, zorder=6, markeredgecolor="white", markeredgewidth=0.3)
        trail_lines.append(ln)
        particle_dots.append(dot)

    time_txt = ax_main.text(0.02, 0.97, "", transform=ax_main.transAxes, color="#c0c8e0", fontsize=8, va="top", fontfamily="monospace")
    scat_txt = ax_main.text(0.02, 0.92, "", transform=ax_main.transAxes, color="#80e0a0", fontsize=7.5, va="top", fontfamily="monospace")

    ax_Vr.set_facecolor("#0d0d1a")
    ax_Vr.tick_params(colors="#8b8b9e", labelsize=7)
    ax_Vr.set_xlabel("r  (distance from target)", color="#8b8b9e", fontsize=8)
    ax_Vr.set_ylabel("V(r)", color="#8b8b9e", fontsize=8)
    ax_Vr.set_title(f"Potential Profile  [{p.potential}]", color="#c9cfe0", fontsize=8.5, fontweight="bold")
    ax_Vr.grid(True, color="#1e1e32", lw=0.5)
    ax_Vr.plot(r_1d, V_1d, color="#ff8040", lw=1.8)
    ax_Vr.fill_between(r_1d, V_1d, 0, where=(V_1d > 0), alpha=0.18, color="#ff8040")
    ax_Vr.fill_between(r_1d, V_1d, 0, where=(V_1d < 0), alpha=0.18, color="#4488ff")
    ax_Vr.axhline(0, color="#444466", lw=0.6)
    ax_Vr.set_xlim(0, 6)

    ax_theta.set_facecolor("#0d0d1a")
    ax_theta.set_theta_zero_location("N")
    ax_theta.set_theta_direction(-1)
    ax_theta.tick_params(colors="#6a6a80", labelsize=6)
    ax_theta.set_title(r"Scattering distribution  $D(\theta)$", color="#c0c8e0", fontsize=8, pad=8)
    ax_theta.grid(color="#1e1e32", lw=0.5)

    n_bins = 36
    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bars = ax_theta.bar(
        bin_centers,
        np.zeros(n_bins),
        width=bin_width,
        color="#c060ff",
        alpha=0.65,
        edgecolor="#8030cc",
        linewidth=0.4,
        bottom=0,
        zorder=4,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap_particles, norm=norm_b)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_main, fraction=0.018, pad=0.02, shrink=0.6)
    cbar.set_label("Impact parameter  b", color="#8b8b9e", fontsize=7)
    cbar.ax.tick_params(colors="#8b8b9e", labelsize=6)

    fig.suptitle(
        f"Classical Scattering  —  potential={p.potential}, v0={p.v0:.2f}, N={p.n_particles}",
        color="white",
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )

    def init():
        for ln in trail_lines:
            ln.set_data([], [])
        for dot in particle_dots:
            dot.set_data([], [])
        time_txt.set_text("")
        scat_txt.set_text("")
        return trail_lines + particle_dots + [time_txt, scat_txt]

    def update(frame_idx: int):
        step = min(frame_idx * p.steps_per_frame, n_steps)
        t = step * p.dt

        for i in range(p.n_particles):
            traj = trajectories[i]
            cx, cz = traj[step, 0], traj[step, 1]
            t_start = max(0, step - p.trail_length)
            trail_x = traj[t_start : step + 1, 0]
            trail_z = traj[t_start : step + 1, 1]
            trail_lines[i].set_data(trail_z, trail_x)
            particle_dots[i].set_data([cz], [cx])

        finished_angles = []
        for i in range(p.n_particles):
            cx, cz = trajectories[i][step, 0], trajectories[i][step, 1]
            if np.sqrt(cx**2 + cz**2) > 6.0:
                finished_angles.append(final_angles[i])

        if finished_angles:
            counts, _ = np.histogram(finished_angles, bins=bin_edges)
            max_c = max(int(counts.max()), 1)
            for bar, h in zip(bars, counts / max_c):
                bar.set_height(float(h))

        time_txt.set_text(f"t = {t:.2f}   step {step}/{n_steps}")
        scat_txt.set_text(f"Scattered: {len(finished_angles)}/{p.n_particles}")
        return trail_lines + particle_dots + list(bars) + [time_txt, scat_txt]

    _emit(progress_cb, 70, "Rendering and encoding MP4 for classical simulation...")
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=1000 // p.fps,
        blit=False,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=p.fps,
        bitrate=-1,
        metadata={"title": "Classical Scattering", "artist": "Unified Scattering Lab"},
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "17"],
    )
    anim.save(output_path, writer=writer, dpi=p.dpi, savefig_kwargs={"facecolor": "#0a0a0f"})

    _emit(progress_cb, 95, "Building scattering-only preview animation...")
    preview_fig, preview_ax = plt.subplots(figsize=(10.5, 6.0), facecolor="#04040c")
    preview_ax.set_facecolor("#04040c")
    preview_ax.set_xlim(*p.z_lims)
    preview_ax.set_ylim(*p.x_lims)
    preview_ax.set_aspect("equal", adjustable="box")
    preview_ax.tick_params(colors="#6a6a80", labelsize=8)
    preview_ax.set_xlabel("z  (propagation axis)", color="#6a6a80", fontsize=9)
    preview_ax.set_ylabel("x  (transverse)", color="#6a6a80", fontsize=9)
    for sp in preview_ax.spines.values():
        sp.set_edgecolor("#1e2f3a")

    # Restore the scattering site context in preview (potential field + target marker).
    preview_ax.contourf(ZG, XG, VG_disp, levels=20, cmap="YlOrRd", alpha=0.22, zorder=1)
    preview_ax.contour(ZG, XG, VG_disp, levels=6, colors="#ff8040", alpha=0.25, linewidths=0.5, zorder=2)
    if p.potential == "soft_sphere":
        preview_ax.add_patch(
            Circle((0, 0), p.sphere_a, fill=False, edgecolor="#ffa040", lw=1.5, linestyle="--", alpha=0.7, zorder=8)
        )
    preview_ax.plot(0, 0, "o", color="#ff6030", ms=6, zorder=10, markeredgecolor="white", markeredgewidth=0.5)
    for xi in np.linspace(-p.b_max * 0.8, p.b_max * 0.8, 7):
        preview_ax.annotate(
            "",
            xy=(p.z_start + 1.5, xi),
            xytext=(p.z_start + 0.2, xi),
            arrowprops=dict(arrowstyle="->", color="#4444aa", lw=0.9, alpha=0.5),
            zorder=3,
        )

    preview_lines = []
    preview_dots = []
    for i in range(p.n_particles):
        col = colors[i]
        ln, = preview_ax.plot([], [], "-", color=col, lw=0.85, alpha=0.7)
        dot, = preview_ax.plot([], [], "o", color=col, ms=2.8, markeredgecolor="white", markeredgewidth=0.2)
        preview_lines.append(ln)
        preview_dots.append(dot)

    preview_time = preview_ax.text(
        0.02,
        0.96,
        "",
        transform=preview_ax.transAxes,
        color="#c0c8e0",
        fontsize=8,
        va="top",
        fontfamily="monospace",
    )

    def preview_init():
        for ln in preview_lines:
            ln.set_data([], [])
        for dot in preview_dots:
            dot.set_data([], [])
        preview_time.set_text("")
        return preview_lines + preview_dots + [preview_time]

    def preview_update(frame_idx: int):
        step = min(frame_idx * p.steps_per_frame, n_steps)
        for i in range(p.n_particles):
            traj = trajectories[i]
            t_start = max(0, step - p.trail_length)
            trail_x = traj[t_start : step + 1, 0]
            trail_z = traj[t_start : step + 1, 1]
            preview_lines[i].set_data(trail_z, trail_x)
            preview_dots[i].set_data([traj[step, 1]], [traj[step, 0]])
        preview_time.set_text(f"t = {step * p.dt:.2f}")
        return preview_lines + preview_dots + [preview_time]

    preview_anim = animation.FuncAnimation(
        preview_fig,
        preview_update,
        frames=n_frames,
        init_func=preview_init,
        interval=1000 // max(1, p.fps),
        blit=False,
    )
    preview_gif_path = os.path.splitext(output_path)[0] + "_preview.gif"
    preview_writer = animation.PillowWriter(fps=max(12, min(p.fps, 20)))
    preview_anim.save(
        preview_gif_path,
        writer=preview_writer,
        dpi=min(p.dpi, 120),
        savefig_kwargs={"facecolor": "#04040c"},
    )
    plt.close(preview_fig)
    plt.close(fig)
    _emit(progress_cb, 100, "Classical simulation completed")
    return output_path
