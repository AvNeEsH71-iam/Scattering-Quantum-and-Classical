from __future__ import annotations

import os
import traceback
from typing import Any, Dict

from PyQt5.QtCore import QThread, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QStackedLayout,
)

from simulations import run_classical, run_quantum
from utils import ParamSpec, build_temp_output_path, copy_video, get_default_output_dir, load_param_specs


class SimulationWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, mode: str, params: Dict[str, Any], output_path: str):
        super().__init__()
        self.mode = mode
        self.params = params
        self.output_path = output_path

    def _on_progress(self, value: int, message: str) -> None:
        self.progress.emit(value)
        self.status.emit(message)

    def run(self) -> None:
        try:
            self.status.emit(f"Running {self.mode}...")
            if self.mode == "Classical Scattering":
                path = run_classical(self.params, self.output_path, self._on_progress)
            else:
                path = run_quantum(self.params, self.output_path, self._on_progress)
            self.finished_ok.emit(path)
        except Exception as exc:
            details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.failed.emit(details)


class MainWindow(QMainWindow):
    def __init__(self, classical_file: str, quantum_file: str):
        super().__init__()
        self.setWindowTitle("Unified Scattering Lab")
        self.resize(1520, 900)

        self.param_specs = load_param_specs(classical_file, quantum_file)
        self.param_widgets: Dict[str, QWidget] = {}
        self.last_output: str | None = None
        self.worker: SimulationWorker | None = None
        self.preview_mode: str = "video"
        self.preview_gif_path: str | None = None

        self.media_player = QMediaPlayer(self)
        self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.media_player.stateChanged.connect(self._on_media_state_changed)
        self.media_player.error.connect(self._on_media_error)
        self.preview_movie: QMovie | None = None

        self._build_ui()
        self._apply_styles()
        self._rebuild_parameter_form()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_gif_frame()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QGridLayout(root)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 0)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        left_panel = self._build_left_panel()
        center_panel = self._build_center_panel()
        right_panel = self._build_right_panel()

        layout.addWidget(left_panel, 0, 0)
        layout.addWidget(center_panel, 0, 1)
        layout.addWidget(right_panel, 0, 2)

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("leftPanel")
        panel.setMinimumWidth(390)
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        title = QLabel("Simulation Setup")
        title.setObjectName("panelTitle")
        v.addWidget(title)

        mode_group = QGroupBox("Simulation Mode")
        mode_layout = QFormLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Classical Scattering", "Quantum Scattering"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addRow("Select", self.mode_combo)
        v.addWidget(mode_group)

        self.param_group = QGroupBox("Parameters")
        self.param_form = QFormLayout()
        self.param_form.setLabelAlignment(Qt.AlignLeft)
        self.param_group.setLayout(self.param_form)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.param_group)
        v.addWidget(scroll, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        v.addWidget(self.progress)

        button_bar = QHBoxLayout()
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self._run_simulation)
        self.save_btn = QPushButton("Save MP4")
        self.save_btn.clicked.connect(self._save_video)
        self.save_btn.setEnabled(False)
        button_bar.addWidget(self.run_btn)
        button_bar.addWidget(self.save_btn)
        v.addLayout(button_bar)

        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("centerPanel")
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        title = QLabel("Animation Preview")
        title.setObjectName("panelTitle")
        v.addWidget(title)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(720, 480)
        self.media_player.setVideoOutput(self.video_widget)

        self.gif_label = QLabel("Preview will appear here after simulation finishes")
        self.gif_label.setAlignment(Qt.AlignCenter)
        self.gif_label.setMinimumSize(720, 480)
        self.gif_label.setStyleSheet("background-color: #05070a; color: #8fa6ba;")

        self.preview_container = QWidget()
        self.preview_stack = QStackedLayout(self.preview_container)
        self.preview_stack.addWidget(self.video_widget)
        self.preview_stack.addWidget(self.gif_label)
        self.preview_stack.setCurrentIndex(0)

        v.addWidget(self.preview_container, 1)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.restart_btn = QPushButton("Restart")
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.restart_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._play_preview)
        self.pause_btn.clicked.connect(self._pause_preview)
        self.restart_btn.clicked.connect(self._restart_video)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.restart_btn)
        controls.addStretch(1)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.media_player.setPosition)
        self.media_player.durationChanged.connect(self._on_duration_changed)
        self.media_player.positionChanged.connect(self._on_position_changed)

        v.addLayout(controls)
        v.addWidget(self.position_slider)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("rightPanel")
        panel.setMinimumWidth(380)
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        title = QLabel("Simulation Log")
        title.setObjectName("panelTitle")
        v.addWidget(title)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        v.addWidget(self.log_text, 1)

        return panel

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #ecf1f5; }
            QFrame#leftPanel, QFrame#centerPanel, QFrame#rightPanel {
                background-color: #ffffff;
                border: 1px solid #c8d3de;
                border-radius: 10px;
            }
            QLabel#panelTitle {
                color: #12324a;
                font-size: 18px;
                font-weight: 700;
            }
            QGroupBox {
                border: 1px solid #d6e0e9;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: 600;
                color: #18415f;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #0f6aa8;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:disabled {
                background-color: #8fa6ba;
            }
            QProgressBar {
                border: 1px solid #bfd0de;
                border-radius: 5px;
                text-align: center;
                background-color: #edf3f8;
            }
            QProgressBar::chunk {
                background-color: #2a8ccf;
                border-radius: 5px;
            }
            """
        )

    def _clear_form(self) -> None:
        while self.param_form.rowCount() > 0:
            self.param_form.removeRow(0)
        self.param_widgets.clear()

    def _rebuild_parameter_form(self) -> None:
        self._clear_form()
        specs = self.param_specs[self.mode_combo.currentText()]
        for spec in specs:
            if spec.kind == "choice":
                editor = QComboBox()
                for item in spec.choices or []:
                    editor.addItem(item)
                idx = editor.findText(str(spec.default))
                if idx >= 0:
                    editor.setCurrentIndex(idx)
            elif spec.kind == "int":
                editor = QSpinBox()
                editor.setRange(int(spec.minimum), int(spec.maximum))
                editor.setSingleStep(int(spec.step))
                editor.setValue(int(spec.default))
            else:
                editor = QDoubleSpinBox()
                editor.setRange(float(spec.minimum), float(spec.maximum))
                editor.setSingleStep(float(spec.step))
                editor.setDecimals(6)
                editor.setValue(float(spec.default))

            editor.setObjectName(spec.key)
            self.param_form.addRow(spec.label, editor)
            self.param_widgets[spec.key] = editor

    def _get_params(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, widget in self.param_widgets.items():
            if isinstance(widget, QComboBox):
                out[key] = widget.currentText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                out[key] = widget.value()
            else:
                raise ValueError(f"Unsupported widget for parameter '{key}'")

        if self.mode_combo.currentText() == "Classical Scattering":
            out.setdefault("b_min", 0.05)
            out.setdefault("z_start", -9.0)
            out.setdefault("x_lims", (-5.5, 5.5))
            out.setdefault("z_lims", (-9.5, 9.5))
            out.setdefault("steps_per_frame", 5)
            out.setdefault("trail_length", 60)
            out.setdefault("fps", 28)
            out.setdefault("dpi", 180)
        else:
            out.setdefault("steps_per_frame", 10)
            out.setdefault("fps", 26)
            out.setdefault("dpi", 180)
            out.setdefault("lx", 9.0)
            out.setdefault("lz", 13.0)
            out.setdefault("x0", 0.0)
            out.setdefault("z0", -8.5)
            out.setdefault("soft_sigma", 1.0)
            out.setdefault("absorb_width", 2.2)
            out.setdefault("absorb_strength", 18.0)
            out.setdefault("sphere_a", 1.5)
            out.setdefault("sphere_v0", 600.0)
            out.setdefault("yukawa_beta", 8.0)
            out.setdefault("yukawa_mu", 0.6)
        return out

    def _append_log(self, text: str) -> None:
        self.log_text.append(text)

    def _on_mode_changed(self) -> None:
        self._rebuild_parameter_form()
        self._append_log(f"Selected simulation type: {self.mode_combo.currentText()}")

    def _run_simulation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Simulation Running", "A simulation is already running.")
            return

        try:
            params = self._get_params()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Parameters", str(exc))
            return

        mode = self.mode_combo.currentText()
        output_dir = get_default_output_dir()
        out_file = build_temp_output_path(mode.replace(" ", "_"), output_dir)

        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.restart_btn.setEnabled(False)
        self._append_log(f"Selected simulation type: {mode}")
        self._append_log(f"Parameter values: {params}")
        self._append_log("Status: Running...")

        self.worker = SimulationWorker(mode, params, out_file)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(lambda m: self._append_log(f"Status: {m}"))
        self.worker.finished_ok.connect(self._on_simulation_finished)
        self.worker.failed.connect(self._on_simulation_failed)
        self.worker.start()

    def _on_simulation_finished(self, video_path: str) -> None:
        self.last_output = video_path
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.restart_btn.setEnabled(True)
        self._append_log("Status: Completed")
        self._append_log(f"Generated MP4: {video_path}")

        # Prefer scattering-only preview asset, fallback to generic GIF if present.
        preview_primary = os.path.splitext(video_path)[0] + "_preview.gif"
        preview_fallback = os.path.splitext(video_path)[0] + ".gif"
        if os.path.exists(preview_primary):
            self.preview_gif_path = preview_primary
        elif os.path.exists(preview_fallback):
            self.preview_gif_path = preview_fallback
        else:
            self.preview_gif_path = None

        if self.preview_gif_path:
            self._append_log(f"Generated preview: {self.preview_gif_path}")
            self._show_gif_preview(self.preview_gif_path, autoplay=True)
        else:
            self.preview_mode = "video"
            self.preview_stack.setCurrentIndex(0)
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
            # Start preview automatically once media is loaded.
            QTimer.singleShot(300, self.media_player.play)

    def _on_simulation_failed(self, details: str) -> None:
        self.run_btn.setEnabled(True)
        self._append_log("Status: Error")
        self._append_log(details)
        QMessageBox.critical(self, "Simulation Error", details.splitlines()[-1] if details.strip() else "Unknown error")

    def _save_video(self) -> None:
        if not self.last_output or not os.path.exists(self.last_output):
            QMessageBox.warning(self, "No Output", "No generated MP4 is available to save.")
            return

        target_path, _ = QFileDialog.getSaveFileName(self, "Save MP4", "scattering_output.mp4", "MP4 Video (*.mp4)")
        if not target_path:
            return

        try:
            copy_video(self.last_output, target_path)
            self._append_log(f"Saved MP4 to: {target_path}")
        except Exception as exc:
            self._append_log(f"Error while saving MP4: {exc}")
            QMessageBox.critical(self, "Save Error", str(exc))

    def _restart_video(self) -> None:
        if self.preview_mode == "video":
            self.media_player.setPosition(0)
            self.media_player.play()
        else:
            if self.preview_movie is not None:
                self.preview_movie.stop()
                self.preview_movie.start()

    def _play_preview(self) -> None:
        if self.preview_mode == "video":
            self.media_player.play()
        else:
            if self.preview_movie is not None:
                self.preview_movie.setPaused(False)
                if self.preview_movie.state() != QMovie.Running:
                    self.preview_movie.start()

    def _pause_preview(self) -> None:
        if self.preview_mode == "video":
            self.media_player.pause()
        else:
            if self.preview_movie is not None:
                self.preview_movie.setPaused(True)

    def _switch_to_gif_preview(self) -> None:
        if not self.preview_gif_path or not os.path.exists(self.preview_gif_path):
            self._append_log("Preview fallback unavailable: GIF preview file not found")
            return

        self._show_gif_preview(self.preview_gif_path, autoplay=True)
        self._append_log("Preview fallback: switched to GIF playback")

    def _show_gif_preview(self, gif_path: str, autoplay: bool) -> None:
        self.preview_mode = "gif"
        if self.preview_movie is not None:
            try:
                self.preview_movie.frameChanged.disconnect(self._on_movie_frame_changed)
            except Exception:
                pass
            self.preview_movie.stop()
        self.preview_movie = QMovie(gif_path)
        self.preview_movie.setCacheMode(QMovie.CacheAll)
        self.preview_movie.frameChanged.connect(self._on_movie_frame_changed)
        self.preview_stack.setCurrentIndex(1)
        if autoplay:
            self.preview_movie.start()
        self._refresh_gif_frame()

    def _on_movie_frame_changed(self, _frame_idx: int) -> None:
        self._refresh_gif_frame()

    def _refresh_gif_frame(self) -> None:
        if self.preview_movie is None:
            return
        pix = self.preview_movie.currentPixmap()
        if pix.isNull():
            return
        scaled = pix.scaled(self.gif_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.gif_label.setPixmap(scaled)

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.LoadedMedia:
            self._append_log("Preview status: media loaded")
        elif status == QMediaPlayer.InvalidMedia:
            self._append_log("Preview status: invalid media")
            self._switch_to_gif_preview()
        elif status == QMediaPlayer.EndOfMedia:
            self._append_log("Preview status: playback finished")

    def _on_media_state_changed(self, state: QMediaPlayer.State) -> None:
        if state == QMediaPlayer.PlayingState:
            self._append_log("Preview: playing")
        elif state == QMediaPlayer.PausedState:
            self._append_log("Preview: paused")
        elif state == QMediaPlayer.StoppedState:
            self._append_log("Preview: stopped")

    def _on_media_error(self, _error: QMediaPlayer.Error) -> None:
        err_msg = self.media_player.errorString() or "Unknown media playback error"
        self._append_log(f"Preview error: {err_msg}")
        self._switch_to_gif_preview()

    def _on_duration_changed(self, duration: int) -> None:
        self.position_slider.setRange(0, duration)

    def _on_position_changed(self, position: int) -> None:
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)
