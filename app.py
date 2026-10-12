"""Desktop app: analyse a recording, or listen to the microphone in real time.

    python app.py                       # default model (models/checkpoints/best.pt)
    python app.py --checkpoint models/checkpoints/v4/best.pt --device cpu
    python app.py --simulate some.wav   # feed a file into the live tab as if it were the mic

Live mode keeps the last ~4 s of microphone audio and scores it every 0.5 s,
exactly like detect.py scores a file: Silero VAD first (no speech -> no verdict),
then AASIST on the standard 16 kHz / 4.04 s / -26 dBFS window. The verdict is the
average P(AI) of the last few speech windows, so one odd window doesn't flip it.

Based on the GUI from the earlier SIH project (src/gui.py).
"""

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QThread, QTimer, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame, QGraphicsDropShadowEffect,
                               QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton, QSlider,
                               QTabWidget, QVBoxLayout, QWidget)

from detector.audio_utils import load_file, to_standard_rate
from detector.config import NB_SAMP, SAMPLE_RATE
from detector.model import Detector

HOP_SECONDS = 0.5          # how often the live window is scored
MIN_LIVE_SECONDS = 1.5     # start scoring once this much audio has arrived
SMOOTH_WINDOWS = 4         # verdict = mean P(AI) of this many recent speech windows (~2 s of new audio)
HISTORY_SECONDS = 30       # length of the live P(AI) graph
MEDIA = "Audio / video (*.wav *.flac *.mp3 *.m4a *.ogg *.opus *.webm *.mp4 *.mkv);;All files (*)"

RED, GREEN, BLUE, GREY = "#ff7b72", "#7ee787", "#58a6ff", "#8b949e"


# ----------------------------- widgets -----------------------------

class GlowButton(QPushButton):
    """Button with a restrained hover glow."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._glow = QGraphicsDropShadowEffect(self)
        self._glow.setOffset(0, 1)
        self._glow.setBlurRadius(0)
        self._glow.setColor(QColor(BLUE))
        self.setGraphicsEffect(self._glow)
        self._anim = QPropertyAnimation(self._glow, b"blurRadius", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _set_glow(self, blur):
        self._anim.stop()
        self._anim.setStartValue(self._glow.blurRadius())
        self._anim.setEndValue(blur)
        self._anim.start()

    def enterEvent(self, event):
        super().enterEvent(event)
        if self.isEnabled():
            self._set_glow(18)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._set_glow(0)


class ConfidenceMeter(QWidget):
    """Animated HUMAN / AI bars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ai = self._start = self._target = 0.5
        self._visible = False
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(300)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._step)
        self.setMinimumHeight(72)

    def set_ai(self, ai):
        if ai is None or np.isnan(ai):
            self._visible = False
            self.update()
            return
        self._visible = True
        self._anim.stop()
        self._start, self._target = self._ai, float(np.clip(ai, 0, 1))
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def _step(self, t):
        self._ai = self._start + (self._target - self._start) * float(t)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(2, 4, -2, -4)
        bar_left, value_w = r.left() + 70, 60
        for i, (label, value, color) in enumerate((("HUMAN", 1 - self._ai, GREEN), ("AI", self._ai, RED))):
            y = r.top() + 6 + i * 30
            p.setFont(QFont("Sans", 9, QFont.Weight.Bold))
            p.setPen(QColor(color))
            p.drawText(r.left(), y + 12, label)
            track = self.rect().adjusted(bar_left, y, -value_w - 2, 0)
            track.setHeight(14)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#0d1117"))
            p.drawRoundedRect(track, 7, 7)
            if self._visible:
                fill = track.adjusted(0, 0, 0, 0)
                fill.setWidth(max(3, int(track.width() * value)))
                p.setBrush(QColor(color))
                p.drawRoundedRect(fill, 7, 7)
            p.setPen(QColor("#c9d1d9"))
            p.drawText(r.right() - value_w + 10, y + 12, f"{value * 100:.1f}%" if self._visible else "—")


class WaveformWidget(QWidget):
    """Oscilloscope-style view of the recent audio."""

    def __init__(self, caption="AUDIO PREVIEW"):
        super().__init__()
        self.samples = np.zeros(1000, np.float32)
        self.caption, self.live = caption, False
        self.setMinimumHeight(120)

    def set_samples(self, samples):
        samples = np.asarray(samples, np.float32)
        if len(samples) < 2:
            return
        target = max(400, self.width() * 2)
        if len(samples) > target:
            # min/max per bucket keeps the envelope visible
            b = len(samples) // (target // 2)
            s = samples[:b * (target // 2)].reshape(-1, b)
            samples = np.stack([s.min(1), s.max(1)], 1).ravel()
        peak = float(np.abs(samples).max())
        self.samples = samples / max(peak, 0.02)       # quiet input stays small
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0b0f14"))
        r = self.rect().adjusted(12, 12, -12, -12)
        mid = r.center().y()
        p.setFont(QFont("Sans", 8, QFont.Weight.Bold))
        p.setPen(QColor(GREEN if self.live else "#6e7681"))
        p.drawText(r.left(), r.top() + 10, self.caption)
        p.setPen(QPen(QColor("#202938"), 1))
        p.drawLine(r.left(), mid, r.right(), mid)
        n = len(self.samples)
        half = r.height() * 0.4
        p.setPen(QPen(QColor(GREEN if self.live else BLUE), 1.3))
        prev = None
        for i, v in enumerate(self.samples):
            pt = (int(r.left() + i / (n - 1) * r.width()), int(mid - float(v) * half))
            if prev:
                p.drawLine(prev[0], prev[1], pt[0], pt[1])
            prev = pt


class ProbabilityGraph(QWidget):
    """P(AI) over time, threshold as a dashed line; gaps where there was no speech."""

    def __init__(self, span_seconds=None):
        super().__init__()
        self.points = []            # (time_s, p_ai or nan)
        self.span = span_seconds    # None = fit everything (file mode)
        self.threshold = 0.5
        self.setMinimumHeight(140)

    def set_points(self, points, threshold):
        self.points, self.threshold = list(points), threshold
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0b0f14"))
        r = self.rect().adjusted(40, 22, -12, -20)
        p.setFont(QFont("Sans", 8, QFont.Weight.Bold))
        p.setPen(QColor("#6e7681"))
        p.drawText(12, 14, "P(AI) OVER TIME")
        for v in (0.0, 0.5, 1.0):
            y = int(r.bottom() - v * r.height())
            p.setPen(QPen(QColor("#202938"), 1))
            p.drawLine(r.left(), y, r.right(), y)
            p.setPen(QColor("#6e7681"))
            p.drawText(8, y + 4, f"{v:.1f}")
        ty = int(r.bottom() - self.threshold * r.height())
        p.setPen(QPen(QColor("#d29922"), 1, Qt.PenStyle.DashLine))
        p.drawLine(r.left(), ty, r.right(), ty)
        if not self.points:
            return
        t_end = self.points[-1][0]
        t0 = t_end - self.span if self.span else self.points[0][0]
        width = max(t_end - t0, 1e-6)
        prev = None
        for t, v in self.points:
            if t < t0:
                continue
            x = int(r.left() + (t - t0) / width * r.width())
            if np.isnan(v):
                prev = None
                continue
            y = int(r.bottom() - v * r.height())
            color = QColor(RED if v >= self.threshold else GREEN)
            p.setPen(QPen(color, 2))
            if prev:
                p.drawLine(prev[0], prev[1], x, y)
            p.setBrush(color)
            p.drawEllipse(x - 2, y - 2, 4, 4)
            prev = (x, y)
        p.setPen(QColor("#6e7681"))
        p.drawText(r.left(), self.rect().bottom() - 4, f"{t0:.0f} s")
        p.drawText(r.right() - 40, self.rect().bottom() - 4, f"{t_end:.0f} s")


# ----------------------------- workers -----------------------------

class FileWorker(QThread):
    progress = Signal(float)              # seconds analysed
    done = Signal(object)
    error = Signal(str)

    def __init__(self, det, path):
        super().__init__()
        self.det, self.path = det, path

    def run(self):
        try:
            self.done.emit(self.det.score_file(self.path, progress=self.progress.emit))
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


class LiveWorker(QThread):
    """Collects audio (microphone, or a file played in real time) and scores the
    most recent ~4 s every HOP_SECONDS."""

    scored = Signal(float, float, bool)   # time since start, P(AI) (nan if no speech), no_speech
    status = Signal(str)
    error = Signal(str)

    def __init__(self, det, device=None, simulate=None):
        super().__init__()
        self.det, self.device, self.simulate = det, device, simulate
        self.running = True
        self.lock = threading.Lock()
        self.buffer = np.zeros(0, np.float32)     # last NB_SAMP samples at 16 kHz
        self.rate = SAMPLE_RATE

    def recent(self):
        with self.lock:
            return self.buffer.copy()

    def _push(self, samples):
        samples = to_standard_rate(samples, self.rate) if self.rate != SAMPLE_RATE else samples
        with self.lock:
            self.buffer = np.concatenate((self.buffer, samples))[-NB_SAMP:]

    def _callback(self, indata, frames, t, status):
        self._push(indata[:, 0].astype(np.float32).copy())

    def _open_stream(self):
        for rate in (SAMPLE_RATE, None):          # 16 kHz if the device allows, else its own rate
            try:
                if rate is None:
                    rate = int(sd.query_devices(self.device, "input")["default_samplerate"])
                stream = sd.InputStream(device=self.device, samplerate=rate, channels=1, dtype="float32",
                                        blocksize=int(rate * 0.05), callback=self._callback)
                self.rate = rate
                return stream
            except Exception as e:
                last = e
        raise last

    def _simulated(self):
        """Feed a file in 50 ms blocks at real-time speed, as if it came from the mic."""
        wav = load_file(self.simulate)
        block = SAMPLE_RATE // 20
        for i in range(0, len(wav), block):
            if not self.running:
                return
            self._push(wav[i:i + block])
            yield
            time.sleep(block / SAMPLE_RATE)

    def run(self):
        try:
            if self.simulate:
                self.status.emit(f"Simulating microphone with {Path(self.simulate).name}")
                feed = self._simulated()
                stream = None
            else:
                stream = self._open_stream()
                stream.start()
                name = sd.query_devices(self.device if self.device is not None else sd.default.device[0])["name"]
                self.status.emit(f"Listening • {name} • {self.rate} Hz")
            start = next_score = time.monotonic()
            while self.running:
                if self.simulate and next(feed, "end") == "end":
                    self.status.emit("Simulated input finished.")
                    break
                now = time.monotonic()
                if now >= next_score:
                    next_score = now + HOP_SECONDS
                    audio = self.recent()
                    if len(audio) >= MIN_LIVE_SECONDS * SAMPLE_RATE:
                        audio = audio - audio.mean()                       # mic DC offset
                        r = self.det.score_waveform(audio)
                        self.scored.emit(now - start, r["ai_probability"], r["no_speech"])
                if not self.simulate:
                    time.sleep(0.01)
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")

    def stop(self):
        self.running = False


# ----------------------------- main window -----------------------------

STYLE = """
QMainWindow, QWidget { background:#0d1117; color:#e6edf3; }
QTabWidget::pane { border:1px solid #30363d; border-radius:10px; }
QTabBar::tab { background:#161b22; color:#8b949e; padding:10px 18px; border:0; }
QTabBar::tab:selected { color:#e6edf3; background:#21262d; border-bottom:2px solid #58a6ff; }
QPushButton { background:#21262d; border:1px solid #30363d; border-radius:8px; padding:10px 15px; color:#e6edf3; font-weight:600; }
QPushButton:hover { background:#30363d; border-color:#58a6ff; }
QPushButton:disabled { color:#484f58; background:#161b22; border-color:#21262d; }
QPushButton#primary { background:#238636; border-color:#2ea043; }
QPushButton#primary:hover { background:#2ea043; }
QPushButton#danger { background:#3b1f22; border-color:#7d3038; }
QComboBox { background:#21262d; border:1px solid #30363d; border-radius:6px; padding:6px; }
QProgressBar { background:#161b22; border:1px solid #30363d; border-radius:5px; height:9px; }
QProgressBar::chunk { background:#58a6ff; border-radius:5px; }
QFrame#card { background:#161b22; border:1px solid #30363d; border-radius:12px; }
QFrame#card[state="human"] { background:#11261c; border-color:#238636; }
QFrame#card[state="ai"] { background:#2b171b; border-color:#da3633; }
QFrame#card QLabel, QFrame#card ConfidenceMeter { background:transparent; }
QSlider::groove:horizontal { height:6px; background:#30363d; border-radius:3px; }
QSlider::handle:horizontal { background:#58a6ff; width:14px; margin:-5px 0; border-radius:7px; }
"""


def card():
    f = QFrame()
    f.setObjectName("card")
    return f


def set_state(widget, state):
    widget.setProperty("state", state)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class Window(QMainWindow):
    def __init__(self, det, simulate=None):
        super().__init__()
        self.det, self.simulate = det, simulate
        self.file_worker = self.live_worker = None
        self.live_points, self.session = [], []
        self.recent = deque(maxlen=SMOOTH_WINDOWS)
        self.setWindowTitle("AI Voice Detector")
        self.setMinimumSize(860, 800)
        self.setStyleSheet(STYLE)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(28, 22, 28, 18)
        title = QLabel("AI Voice Detector")
        title.setFont(QFont("Sans", 23, QFont.Weight.Bold))
        outer.addWidget(title)
        sub = QLabel("AASIST (297K params) • Silero speech detector • 4-second windows")
        sub.setStyleSheet(f"color:{GREY}")
        outer.addWidget(sub)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.build_live_tab()
        self.build_file_tab()

        row = QHBoxLayout()
        self.thr_label = QLabel()
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setRange(5, 95)
        self.thr_slider.setValue(int(round(det.threshold * 100)))
        self.thr_slider.valueChanged.connect(self.set_threshold)
        self.thr_slider.setToolTip("Lower = flags AI more readily (more false alarms on humans)")
        row.addWidget(self.thr_label)
        row.addWidget(self.thr_slider, 1)
        outer.addLayout(row)
        foot = QLabel(f"Model: {det.source} on {det.device}" + ("  •  speech detector on" if det.vad else ""))
        foot.setStyleSheet("color:#6e7681")
        outer.addWidget(foot)
        self.set_threshold(self.thr_slider.value())

        self.wave_timer = QTimer(self)
        self.wave_timer.setInterval(40)
        self.wave_timer.timeout.connect(self.refresh_wave)

    # ---------- shared ----------

    def set_threshold(self, value):
        self.det.threshold = value / 100
        self.thr_label.setText(f"Call it AI at P(AI) ≥ {self.det.threshold:.2f}")
        self.live_graph.set_points(self.live_points, self.det.threshold)
        self.update_live_verdict()

    def verdict(self, label, card_widget, p_ai, no_speech):
        if no_speech:
            label.setText("NO SPEECH")
            label.setStyleSheet(f"color:{GREY}")
            set_state(card_widget, "")
        elif p_ai >= self.det.threshold:
            label.setText("AI-GENERATED")
            label.setStyleSheet(f"color:{RED}")
            set_state(card_widget, "ai")
        else:
            label.setText("HUMAN")
            label.setStyleSheet(f"color:{GREEN}")
            set_state(card_widget, "human")

    # ---------- live tab ----------

    def build_live_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(12)

        controls = QHBoxLayout()
        self.mic_box = QComboBox()
        self.mic_box.addItem("Default microphone", None)
        try:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    self.mic_box.addItem(f"{i}: {d['name']}", i)
        except Exception:
            pass
        if self.simulate:
            self.mic_box.insertItem(0, f"Simulated: {Path(self.simulate).name}", "simulate")
            self.mic_box.setCurrentIndex(0)
        self.start_btn = GlowButton("Start microphone")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start_live)
        self.stop_btn = GlowButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_live)
        reset = GlowButton("Reset")
        reset.clicked.connect(self.reset_live)
        controls.addWidget(self.mic_box, 1)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(reset)
        lay.addLayout(controls)

        self.live_wave = WaveformWidget("MICROPHONE IDLE")
        lay.addWidget(self.live_wave)

        self.live_card = card()
        c = QVBoxLayout(self.live_card)
        self.live_result = QLabel("PRESS START AND SPEAK")
        self.live_result.setFont(QFont("Sans", 26, QFont.Weight.Bold))
        self.live_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_meter = ConfidenceMeter()
        self.live_detail = QLabel("")
        self.live_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_detail.setStyleSheet(f"color:{GREY}")
        c.addWidget(self.live_result)
        c.addWidget(self.live_meter)
        c.addWidget(self.live_detail)
        lay.addWidget(self.live_card)

        self.live_graph = ProbabilityGraph(span_seconds=HISTORY_SECONDS)
        lay.addWidget(self.live_graph, 1)
        self.live_status = QLabel("Microphone is stopped.")
        self.live_status.setStyleSheet(f"color:{GREY}")
        lay.addWidget(self.live_status)
        self.tabs.addTab(tab, "Live microphone")

    def start_live(self):
        if self.live_worker and self.live_worker.isRunning():
            return
        choice = self.mic_box.currentData()
        self.reset_live()
        self.live_worker = LiveWorker(self.det, device=None if choice == "simulate" else choice,
                                      simulate=self.simulate if choice == "simulate" else None)
        self.live_worker.scored.connect(self.on_live_score)
        self.live_worker.status.connect(self.live_status.setText)
        self.live_worker.error.connect(lambda e: self.live_status.setText(f"Microphone error: {e}"))
        self.live_worker.finished.connect(self.on_live_stopped)
        self.live_worker.start()
        self.live_wave.live, self.live_wave.caption = True, "LIVE INPUT"
        self.wave_timer.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.live_result.setText("LISTENING…")
        self.live_result.setStyleSheet("")

    def stop_live(self):
        if self.live_worker:
            self.live_worker.stop()

    def on_live_stopped(self):
        self.wave_timer.stop()
        self.live_wave.live, self.live_wave.caption = False, "MICROPHONE IDLE"
        self.live_wave.update()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.session:
            self.live_status.setText(f"Stopped. Whole session: P(AI) {np.mean(self.session):.2f} over "
                                     f"{len(self.session)} speech windows.")

    def reset_live(self):
        self.live_points, self.session = [], []
        self.recent.clear()
        self.live_graph.set_points([], self.det.threshold)
        self.live_meter.set_ai(None)
        self.live_detail.setText("")
        set_state(self.live_card, "")

    def refresh_wave(self):
        if self.live_worker:
            self.live_wave.set_samples(self.live_worker.recent()[-SAMPLE_RATE * 2:])

    def on_live_score(self, t, p_ai, no_speech):
        self.live_points.append((t, float("nan") if no_speech else p_ai))
        if not no_speech:
            self.recent.append(p_ai)
            self.session.append(p_ai)
        elif self.recent and t - self._last_speech_t > 3:   # silence for a while: start fresh
            self.recent.clear()
        if not no_speech:
            self._last_speech_t = t
        self.live_graph.set_points(self.live_points, self.det.threshold)
        self.update_live_verdict(listening=no_speech)

    _last_speech_t = 0.0

    def update_live_verdict(self, listening=False):
        if not hasattr(self, "live_result"):
            return
        if not self.recent:
            if self.live_worker and self.live_worker.isRunning():
                self.live_result.setText("LISTENING… (no speech)")
                self.live_result.setStyleSheet(f"color:{GREY}")
                set_state(self.live_card, "")
            return
        smooth = float(np.mean(self.recent))
        self.verdict(self.live_result, self.live_card, smooth, False)
        self.live_meter.set_ai(smooth)
        self.live_detail.setText(
            f"last {len(self.recent)} speech windows: P(AI) {smooth:.2f}"
            + ("   •   (silence now)" if listening else "")
            + f"   •   session average {np.mean(self.session):.2f} over {len(self.session)} windows")

    # ---------- file tab ----------

    def build_file_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(12)
        src = card()
        c = QVBoxLayout(src)
        self.file_label = QLabel("Choose a recording (any length, any format ffmpeg reads).")
        self.file_label.setStyleSheet(f"color:{GREY}")
        self.file_label.setWordWrap(True)
        row = QHBoxLayout()
        browse = GlowButton("Browse…")
        browse.clicked.connect(self.pick_file)
        self.analyze_btn = GlowButton("Analyze")
        self.analyze_btn.setObjectName("primary")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.start_file)
        row.addWidget(browse)
        row.addWidget(self.analyze_btn)
        row.addStretch()
        self.file_bar = QProgressBar()
        self.file_bar.setTextVisible(False)
        self.file_status = QLabel("")
        self.file_status.setStyleSheet(f"color:{GREY}")
        c.addWidget(self.file_label)
        c.addLayout(row)
        c.addWidget(self.file_bar)
        c.addWidget(self.file_status)
        lay.addWidget(src)

        self.file_card = card()
        r = QVBoxLayout(self.file_card)
        self.file_result = QLabel("READY")
        self.file_result.setFont(QFont("Sans", 26, QFont.Weight.Bold))
        self.file_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_meter = ConfidenceMeter()
        self.file_detail = QLabel("")
        self.file_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_detail.setStyleSheet(f"color:{GREY}")
        r.addWidget(self.file_result)
        r.addWidget(self.file_meter)
        r.addWidget(self.file_detail)
        lay.addWidget(self.file_card)
        self.file_graph = ProbabilityGraph()
        lay.addWidget(self.file_graph, 1)
        self.tabs.addTab(tab, "File analysis")
        self.path = None

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a recording", "", MEDIA)
        if path:
            self.path = path
            self.file_label.setText(Path(path).name)
            self.analyze_btn.setEnabled(True)

    def start_file(self):
        if not self.path:
            return
        self.analyze_btn.setEnabled(False)
        self.file_bar.setRange(0, 0)           # busy indicator; long files report progress below
        self.file_result.setText("ANALYZING…")
        self.file_result.setStyleSheet("")
        self.file_worker = FileWorker(self.det, self.path)
        self.file_worker.progress.connect(lambda s: self.file_status.setText(f"{s / 60:.1f} min analysed…"))
        self.file_worker.done.connect(self.file_done)
        self.file_worker.error.connect(self.file_error)
        self.file_worker.start()

    def file_done(self, r):
        self.file_bar.setRange(0, 1)
        self.file_bar.setValue(1)
        self.analyze_btn.setEnabled(True)
        self.verdict(self.file_result, self.file_card, r["ai_probability"], r["no_speech"])
        self.file_meter.set_ai(None if r["no_speech"] else r["ai_probability"])
        n = len(r["window_probs"])
        if r["no_speech"]:
            self.file_detail.setText(f"{r['duration']:.1f} s, no speech found")
        else:
            ai_share = float((r["window_probs"] >= self.det.threshold).mean())
            self.file_detail.setText(f"{r['duration']:.1f} s  •  {n} speech windows  •  "
                                     f"{ai_share:.0%} of windows look AI  •  mean P(AI) {r['ai_probability']:.2f}")
        self.file_status.setText("Done.")
        self.file_graph.set_points(list(zip(r["window_starts"], r["window_probs"])), self.det.threshold)

    def file_error(self, e):
        self.file_bar.setRange(0, 1)
        self.analyze_btn.setEnabled(True)
        self.file_result.setText("ERROR")
        self.file_result.setStyleSheet(f"color:{RED}")
        self.file_status.setText(e)

    def closeEvent(self, event):
        if self.live_worker and self.live_worker.isRunning():
            self.live_worker.stop()
            self.live_worker.wait(2000)
        event.accept()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--simulate", default=None, help="audio file to feed into the live tab instead of the mic")
    args = p.parse_args()
    det = Detector(args.checkpoint, device=args.device)
    app = QApplication(sys.argv[:1])
    w = Window(det, simulate=args.simulate)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
