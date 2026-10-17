"""Local web UI for the detector: open http://127.0.0.1:8765 after starting it.

    python server.py                                   # default model, http://127.0.0.1:8765
    python server.py --checkpoint models/checkpoints/v4/best.pt --port 8765 --device cpu

The page (web/index.html) captures the microphone in the browser and sends the latest
~4 s to this server every 0.5 s; the model runs here, exactly like detect.py (Silero
VAD first, then AASIST on the standard 16 kHz / 4.04 s / -26 dBFS window). Files can be
dropped on the page too. Only listens on 127.0.0.1: nothing leaves the machine.
Python standard library only (no Flask etc.).
"""

import argparse
import json
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from detector.audio_utils import to_standard_rate
from detector.config import NB_SAMP, ROOT, SAMPLE_RATE
from detector.model import Detector
from detector.sessions import save_session

PAGE = ROOT / "web" / "index.html"
MAX_BODY = 200 * 2**20          # 200 MB: long files / sessions


class Handler(BaseHTTPRequestHandler):
    det = None
    lock = threading.Lock()      # one model call at a time

    def log_message(self, fmt, *args):
        pass                     # keep the console quiet; errors are printed below

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            raise ValueError("body too large")
        return self.rfile.read(n)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/info":
            d = self.det
            self._send(200, {"model": d.source, "threshold": d.threshold, "device": str(d.device),
                             "vad": d.vad is not None, "sample_rate": SAMPLE_RATE,
                             "window_seconds": NB_SAMP / SAMPLE_RATE})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/score":          # raw float32 PCM at ?sr=...
                wav = to_standard_rate(np.frombuffer(self._body(), np.float32).copy(), int(q.get("sr", SAMPLE_RATE)))
                wav = wav[-NB_SAMP:] - wav[-NB_SAMP:].mean()
                t = time.perf_counter()
                with self.lock:
                    r = self.det.score_waveform(wav)
                self._send(200, {"no_speech": r["no_speech"],
                                 "p_ai": None if r["no_speech"] else r["ai_probability"],
                                 "ms": round((time.perf_counter() - t) * 1000, 1)})
            elif url.path == "/api/file":         # any audio/video file's bytes, ?name=...
                suffix = Path(q.get("name", "upload.wav")).suffix or ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                    f.write(self._body())
                    f.flush()
                    t = time.perf_counter()
                    with self.lock:
                        r = self.det.score_file(f.name)
                self._send(200, {"no_speech": r["no_speech"],
                                 "p_ai": None if r["no_speech"] else r["ai_probability"],
                                 "duration": r["duration"],
                                 "windows": [[float(a), float(b)] for a, b in zip(r["window_starts"], r["window_probs"])],
                                 "seconds": round(time.perf_counter() - t, 2)})
            elif url.path == "/api/save":         # raw float32 PCM session, ?sr=&label=&name=
                wav = to_standard_rate(np.frombuffer(self._body(), np.float32).copy(), int(q.get("sr", SAMPLE_RATE)))
                label = q.get("label") if q.get("label") in ("human", "ai") else None
                self._send(200, {"message": save_session(wav, label, q.get("name", "mic"))})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    Handler.det = Detector(args.checkpoint, device=args.device)
    noise = np.random.default_rng(0).standard_normal(NB_SAMP).astype(np.float32) * 0.05
    Handler.det.score_windows(noise[None])          # warm up (first GPU call is slow)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Model: {Handler.det.source} on {Handler.det.device}, threshold {Handler.det.threshold:.2f}")
    print(f"Open http://127.0.0.1:{args.port}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
