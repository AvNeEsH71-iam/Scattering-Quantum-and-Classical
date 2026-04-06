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


ProgressCallback = Optional[Callable[[int, str], None]]


def _configure_ffmpeg() -> None:
    try:
        matplotlib.rcParams["animation.ffmpeg_path"] = get_ffmpeg_exe()
    except Exception:
        # Keep matplotlib defaults if imageio-ffmpeg is unavailable.
        pass


@dataclass
class QuantumParams:
    potential_type: str = "barrier"  # barrier|well
    nx: int = 384
    nz: int = 512
    lx: float = 9.0
    lz: float = 13.0
    k0: float = 5.0
    sigma_x: float = 0.9
    sigma_z: float = 1.1
    x0: float = 0.0
    z0: float = -8.5
    sphere_a: float = 1.5
    sphere_v0: float = 600.0
    soft_v0: float = 30.0
    soft_sigma: float = 1.0
    yukawa_beta: float = 8.0
    yukawa_mu: float = 0.6
    dt: float = 0.004
    steps_per_frame: int = 10
    total_time: float = 8.4
    absorb_width: float = 2.2
    absorb_strength: float = 18.0
    fps: int = 26
    dpi: int = 180


def _emit(progress_cb: ProgressCallback, percent: int, message: str) -> None:
    if progress_cb is not None:
        progress_cb(max(0, min(100, int(percent))), message)


def run_quantum(
    params: Dict[str, float | int | str],
    output_path: str,
    progress_cb: ProgressCallback = None,
) -> str:
    _configure_ffmpeg()
    p = QuantumParams(**params)
    n_frames = max(1, int(p.total_time / (p.dt * p.steps_per_frame)))

    nx = int(p.nx)
    nz = int(p.nz)

    x_1d = np.linspace(-p.lx, p.lx, nx, endpoint=False)
    z_1d = np.linspace(-p.lz, p.lz, nz, endpoint=False)
    dx, dz = x_1d[1] - x_1d[0], z_1d[1] - z_1d[0]
    X, Z = np.meshgrid(x_1d, z_1d, indexing="ij")
    R = np.sqrt(X**2 + Z**2)

    kx_1d = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    kz_1d = 2 * np.pi * np.fft.fftfreq(nz, d=dz)
    KX, KZ = np.meshgrid(kx_1d, kz_1d, indexing="ij")
    K2 = KX**2 + KZ**2

    _emit(progress_cb, 2, "Building quantum potential and absorber...")

    def build_potential() -> np.ndarray:
        V = np.zeros((nx, nz), dtype=float)
        if p.potential_type == "barrier":
            V = p.soft_v0 * np.exp(-R**2 / (2 * p.soft_sigma**2))
        elif p.potential_type == "well":
            V = -abs(p.soft_v0) * np.exp(-R**2 / (2 * p.soft_sigma**2))
        return V

    def build_absorber() -> np.ndarray:
        Gamma = np.zeros((nx, nz), dtype=float)

        def ramp(coord: np.ndarray, L: float, d: float) -> np.ndarray:
            dist = np.maximum(np.abs(coord) - (L - d), 0.0) / d
            return p.absorb_strength * dist**2

        Gamma += ramp(X, p.lx, p.absorb_width)
        Gamma += ramp(Z, p.lz, p.absorb_width)
        return Gamma

    V_pot = build_potential()
    Gamma = build_absorber()
    half_V_phase = np.exp(-1j * V_pot * p.dt / 2) * np.exp(-Gamma * p.dt / 2)
    kin_phase = np.exp(-1j * (K2 / 2) * p.dt)

    _emit(progress_cb, 8, "Initializing quantum wave packet...")
    psi = (
        np.exp(-((X - p.x0) ** 2 / (4 * p.sigma_x**2)) - ((Z - p.z0) ** 2 / (4 * p.sigma_z**2)))
        * np.exp(1j * p.k0 * Z)
    )
    norm = np.sqrt((np.abs(psi) ** 2).sum() * dx * dz)
    psi /= norm
    psi0 = psi.copy()

    def split_operator_step(psi_local: np.ndarray) -> np.ndarray:
        psi1 = half_V_phase * psi_local
        psi2 = np.fft.ifftn(kin_phase * np.fft.fftn(psi1))
        psi3 = half_V_phase * psi2
        return psi3

    def probability_current(psi_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dpsi_dz = np.gradient(psi_local, dz, axis=1)
        dpsi_dx = np.gradient(psi_local, dx, axis=0)
        Jx = np.imag(np.conj(psi_local) * dpsi_dx)
        Jz = np.imag(np.conj(psi_local) * dpsi_dz)
        return Jx, Jz

    def compute_probabilities(psi_local: np.ndarray) -> tuple[float, float, float, float]:
        prob = np.abs(psi_local) ** 2
        p_total = float(prob.sum() * dx * dz)
        p_left = float(prob[:, z_1d < 0].sum() * dx * dz)
        p_right = float(prob[:, z_1d >= 0].sum() * dx * dz)
        p_scat = float(prob[np.abs(x_1d) > 1.5, :].sum() * dx * dz)
        return p_total, p_left, p_right, p_scat

    _emit(progress_cb, 15, "Building quantum animation canvas...")
    fig = plt.figure(figsize=(14, 9), facecolor="#04040c")
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[2.2, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.42,
        wspace=0.38,
        left=0.04,
        right=0.97,
        top=0.91,
        bottom=0.09,
    )

    ax_psi2 = fig.add_subplot(gs[:, 0])
    ax_repsi = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[0, 2])
    ax_prob = fig.add_subplot(gs[1, 1:])

    extent = [-p.lz, p.lz, -p.lx, p.lx]

    prob0 = np.abs(psi) ** 2
    im_psi = ax_psi2.imshow(prob0, extent=extent, origin="lower", cmap="inferno", aspect="auto", interpolation="bilinear")
    im_re = ax_repsi.imshow(psi.real, extent=extent, origin="lower", cmap="RdBu_r", aspect="auto", interpolation="bilinear")
    im_ph = ax_phase.imshow(np.angle(psi), extent=extent, origin="lower", cmap="hsv", vmin=-np.pi, vmax=np.pi, aspect="auto")

    step_x = max(1, nx // 18)
    step_z = max(1, nz // 22)
    Xq = X[::step_x, ::step_z]
    Zq = Z[::step_x, ::step_z]
    Jx0, Jz0 = probability_current(psi)
    Jxq = Jx0[::step_x, ::step_z]
    Jzq = Jz0[::step_x, ::step_z]
    J_scale = np.percentile(np.sqrt(Jxq**2 + Jzq**2), 90) + 1e-12
    quiv = ax_psi2.quiver(Zq, Xq, Jzq / J_scale, Jxq / J_scale, alpha=0.45, color="#00ccff", scale=22, width=0.0025)

    for ax in (ax_psi2, ax_repsi, ax_phase):
        ax.set_facecolor("#04040c")
        ax.tick_params(colors="#5a5a72", labelsize=6.5)
        ax.set_xlabel("z", color="#5a5a72", fontsize=8)
        ax.set_ylabel("x", color="#5a5a72", fontsize=8)

    ax_psi2.set_title(r"$|\psi(x,z,t)|^2$", color="#c0c8e0", fontsize=9, fontweight="bold")
    ax_repsi.set_title(r"Re$[\psi(x,z,t)]$", color="#c0c8e0", fontsize=9, fontweight="bold")
    ax_phase.set_title(r"Phase $\angle\psi(x,z,t)$", color="#c0c8e0", fontsize=9, fontweight="bold")

    ax_prob.set_facecolor("#0a0a18")
    ax_prob.tick_params(colors="#5a5a72", labelsize=7)
    ax_prob.grid(True, color="#131326", lw=0.5)
    ax_prob.set_xlabel("Time t", color="#5a5a72", fontsize=8)
    ax_prob.set_ylabel("Probability", color="#5a5a72", fontsize=8)
    ax_prob.set_title("Probability Tracking", color="#c0c8e0", fontsize=9, fontweight="bold")

    times_list = [0.0]
    Ptot_list = [1.0]
    Pleft_list = [0.0]
    Pright_list = [0.0]
    Pscat_list = [0.0]
    line_tot, = ax_prob.plot([], [], color="white", lw=1.5, label="$P_{total}$")
    line_left, = ax_prob.plot([], [], color="#ff6040", lw=1.4, linestyle="--", label="$P_{z<0}$")
    line_right, = ax_prob.plot([], [], color="#40ddaa", lw=1.4, linestyle="-.", label="$P_{z>0}$")
    line_scat, = ax_prob.plot([], [], color="#c060ff", lw=1.2, linestyle=":", label="$P_{|x|>1.5}$")
    ax_prob.legend(fontsize=6.5, facecolor="#0a0a18", edgecolor="#1a1a30", labelcolor="#c0c8e0", loc="upper right")

    time_txt = ax_psi2.text(0.02, 0.97, "t = 0.00", transform=ax_psi2.transAxes, color="white", fontsize=8.5, va="top", fontfamily="monospace")
    norm_txt = ax_psi2.text(0.02, 0.92, "P_total = 1.000", transform=ax_psi2.transAxes, color="#80e0a0", fontsize=7.5, va="top", fontfamily="monospace")

    state = {"psi": psi0.copy()}

    _emit(progress_cb, 20, "Running split-operator evolution and encoding MP4...")

    def update(frame_idx: int):
        psi_local = state["psi"]
        for _ in range(p.steps_per_frame):
            psi_local = split_operator_step(psi_local)
        state["psi"] = psi_local

        t = (frame_idx + 1) * p.steps_per_frame * p.dt
        prob = np.abs(psi_local) ** 2
        re_ = psi_local.real
        phase = np.angle(psi_local)

        im_psi.set_data(prob)
        im_psi.set_clim(0, max(prob.max() * 0.92, 1e-10))

        vmax_r = max(np.percentile(np.abs(re_), 99.0), 1e-10)
        im_re.set_data(re_)
        im_re.set_clim(-vmax_r, vmax_r)

        im_ph.set_data(phase)

        Jx, Jz = probability_current(psi_local)
        Jxq = Jx[::step_x, ::step_z]
        Jzq = Jz[::step_x, ::step_z]
        Js = max(np.percentile(np.sqrt(Jxq**2 + Jzq**2), 93), 1e-12)
        quiv.set_UVC(Jzq / Js, Jxq / Js)

        P_tot, P_left, P_right, P_scat = compute_probabilities(psi_local)
        times_list.append(t)
        Ptot_list.append(P_tot)
        Pleft_list.append(P_left)
        Pright_list.append(P_right)
        Pscat_list.append(P_scat)

        line_tot.set_data(times_list, Ptot_list)
        line_left.set_data(times_list, Pleft_list)
        line_right.set_data(times_list, Pright_list)
        line_scat.set_data(times_list, Pscat_list)
        ax_prob.set_xlim(0, max(times_list[-1] * 1.02, 0.1))
        ax_prob.set_ylim(-0.05, max(1.05, max(Ptot_list) * 1.05))

        time_txt.set_text(f"t = {t:.2f} | frame {frame_idx + 1}/{n_frames}")
        norm_txt.set_text(f"P_total = {P_tot:.3f}")

        if (frame_idx + 1) % max(1, n_frames // 10) == 0:
            _emit(progress_cb, 20 + int(70 * (frame_idx + 1) / n_frames), f"Rendered frame {frame_idx + 1}/{n_frames}")

        return [im_psi, im_re, im_ph, quiv, line_tot, line_left, line_right, line_scat, time_txt, norm_txt]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // p.fps, blit=False)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=p.fps,
        bitrate=-1,
        metadata={"title": "Quantum Scattering", "artist": "Unified Scattering Lab"},
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "17"],
    )
    anim.save(output_path, writer=writer, dpi=p.dpi, savefig_kwargs={"facecolor": "#04040c"})

    _emit(progress_cb, 95, "Building scattering-only preview animation...")
    preview_fig, preview_ax = plt.subplots(figsize=(10.5, 6.0), facecolor="#04040c")
    preview_ax.set_facecolor("#04040c")
    preview_ax.tick_params(colors="#5a5a72", labelsize=8)
    preview_ax.set_xlabel("z", color="#8fa6ba", fontsize=9)
    preview_ax.set_ylabel("x", color="#8fa6ba", fontsize=9)
    for sp in preview_ax.spines.values():
        sp.set_edgecolor("#1e2f3a")

    preview_extent = [-p.lz, p.lz, -p.lx, p.lx]
    preview_state = {"psi": psi0.copy()}
    preview_prob = np.abs(preview_state["psi"]) ** 2
    preview_im = preview_ax.imshow(
        preview_prob,
        extent=preview_extent,
        origin="lower",
        cmap="inferno",
        aspect="auto",
        interpolation="bilinear",
    )
    preview_time = preview_ax.text(
        0.02,
        0.96,
        "t = 0.00",
        transform=preview_ax.transAxes,
        color="#c0c8e0",
        fontsize=8,
        va="top",
        fontfamily="monospace",
    )

    def preview_update(frame_idx: int):
        psi_local = preview_state["psi"]
        for _ in range(p.steps_per_frame):
            psi_local = split_operator_step(psi_local)
        preview_state["psi"] = psi_local
        prob_local = np.abs(psi_local) ** 2
        preview_im.set_data(prob_local)
        preview_im.set_clim(0, max(prob_local.max() * 0.92, 1e-10))
        preview_time.set_text(f"t = {(frame_idx + 1) * p.steps_per_frame * p.dt:.2f}")
        return [preview_im, preview_time]

    preview_anim = animation.FuncAnimation(
        preview_fig,
        preview_update,
        frames=n_frames,
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
    _emit(progress_cb, 100, "Quantum simulation completed")
    return output_path
