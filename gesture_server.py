"""Gesture bridge: webcam -> MediaPipe hands -> WebSocket -> browser game."""

from __future__ import annotations

import argparse
import base64
import json
import math
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from websockets.sync.server import ServerConnection, serve

HOST = "127.0.0.1"
PORT = 8765
BROADCAST_HZ = 20.0
HAND_LOST_MS = 300
SWIPE_VELOCITY = 1.4
SWIPE_REFRACTORY = 0.45
SWIPE_WINDOW = 0.16
MAX_READ_FAILS = 30
PREVIEW_W = 320
PREVIEW_H = 240
PREVIEW_EVERY = 2
DARK_MEAN = 12.0
DARK_FRAMES = 30
GAME_FILE = "gesture game.html"

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _finger_up(lm, tip: int, pip: int) -> bool:
    return _dist(lm[tip], lm[0]) > _dist(lm[pip], lm[0]) * 1.15


def _thumb_up(lm) -> bool:
    return _dist(lm[4], lm[5]) > _dist(lm[3], lm[5]) * 1.1


def encode_preview(frame, lm=None) -> str | None:
    """Downscale, optionally overlay the tracked hand, JPEG-encode as base64."""
    try:
        small = cv2.resize(frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
        if lm is not None:
            for a, b in HAND_CONNECTIONS:
                cv2.line(
                    small,
                    (int(lm[a].x * PREVIEW_W), int(lm[a].y * PREVIEW_H)),
                    (int(lm[b].x * PREVIEW_W), int(lm[b].y * PREVIEW_H)),
                    (95, 224, 255),
                    2,
                    cv2.LINE_AA,
                )
            for p in lm:
                cv2.circle(small, (int(p.x * PREVIEW_W), int(p.y * PREVIEW_H)), 3, (255, 209, 102), -1)
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


class Smoother:
    """Constant-velocity Kalman filter (OpenCV) to kill landmark jitter."""

    def __init__(self, dt: float = 1 / 30.0):
        self._fallback: tuple[float, float] | None = None
        self._dt = dt
        self.kf = None
        try:
            kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
            kf.transitionMatrix = self._transition(dt)
            kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kf.processNoiseCov = np.diag([1e-3, 1e-3, 1e-1, 1e-1]).astype(np.float32)
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3e-3
            kf.errorCovPost = np.eye(4, dtype=np.float32)
            self.kf = kf
        except Exception:
            self.kf = None

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        return np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )

    def __call__(self, x: float, y: float, dt: float | None = None) -> tuple[float, float]:
        if self.kf is None:
            if self._fallback is None:
                self._fallback = (x, y)
            else:
                self._fallback = (
                    self._fallback[0] * 0.6 + x * 0.4,
                    self._fallback[1] * 0.6 + y * 0.4,
                )
            return self._fallback
        try:
            if dt is not None and 0 < dt < 0.5:
                self.kf.transitionMatrix = self._transition(dt)
            self.kf.predict()
            out = self.kf.correct(np.array([[x], [y]], np.float32))
            return float(out[0][0]), float(out[1][0])
        except Exception:
            return x, y

    def reset(self) -> None:
        self._fallback = None
        if self.kf is not None:
            try:
                self.kf.statePost = np.zeros((4, 1), np.float32)
                self.kf.errorCovPost = np.eye(4, dtype=np.float32)
            except Exception:
                pass


class GestureEngine:
    def __init__(self, camera: int = 0, mirror: bool = True, debug: bool = False, preview: bool = True):
        self.camera = camera
        self.mirror = mirror
        self.debug = debug
        self.preview = preview
        self.cap: cv2.VideoCapture | None = None
        self.hands = None
        self.smoother = Smoother()
        self.last_preview: str | None = None
        self.preview_seq = 0
        self._sent_seq = -1
        self._tick = 0
        self._dark = 0
        self._bright = 0.0
        self._hist: list[tuple[float, float, float]] = []
        self._last_swipe = 0.0
        self._last_seen = 0.0
        self._last_smooth = time.time()
        self._frames = 0
        self._read_fails = 0
        self._fps = 0.0
        self._fps_mark = time.time()

    @property
    def fps(self) -> float:
        return self._fps

    def open(self) -> tuple[bool, str]:
        self.cap = cv2.VideoCapture(self.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            return False, f"cannot open camera {self.camera}"
        try:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        except Exception as exc:
            self.cap.release()
            self.cap = None
            return False, f"MediaPipe hand tracker failed to start ({exc})"
        return True, ""

    def _classify(self, lm) -> tuple[str, int]:
        up = [
            _finger_up(lm, 8, 6),
            _finger_up(lm, 12, 10),
            _finger_up(lm, 16, 14),
            _finger_up(lm, 20, 18),
        ]
        count = sum(up)
        thumb = _thumb_up(lm)
        total = count + (1 if thumb else 0)
        if count == 0:
            return "fist", 0
        if count == 4:
            return "open", total
        if count == 1 and up[0]:
            return "point", 1
        return "hand", count

    def _swipe(self, x: float, y: float, now: float) -> str | None:
        self._hist.append((now, x, y))
        cutoff = now - SWIPE_WINDOW
        while self._hist and self._hist[0][0] < cutoff:
            self._hist.pop(0)
        if len(self._hist) < 2 or now - self._last_swipe < SWIPE_REFRACTORY:
            return None
        t0, x0, y0 = self._hist[0]
        t1, x1, y1 = self._hist[-1]
        span = t1 - t0
        if span < 0.05:
            return None
        vy = (y1 - y0) / span
        if vy <= -SWIPE_VELOCITY:
            self._last_swipe = now
            return "jump"
        if vy >= SWIPE_VELOCITY:
            self._last_swipe = now
            return "duck"
        return None

    def read(self) -> dict:
        now = time.time()
        state = {
            "type": "state",
            "hand": False,
            "gesture": "none",
            "fingers": 0,
            "x": None,
            "y": None,
            "event": None,
            "fps": round(self._fps, 1),
            "bright": round(self._bright, 1),
        }
        if self.cap is None:
            return state
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._read_fails += 1
            if self._read_fails >= MAX_READ_FAILS:
                state["note"] = "camera stopped delivering frames"
            return state
        self._read_fails = 0
        if self.mirror:
            frame = cv2.flip(frame, 1)

        self._bright = float(
            cv2.cvtColor(cv2.resize(frame, (16, 12), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).mean()
        )
        self._dark = self._dark + 1 if self._bright < DARK_MEAN else 0
        if self._dark >= DARK_FRAMES:
            state["note"] = "camera image is almost black - light the scene or close other apps using the webcam"
            self._dark = 0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        tracked = None

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            tracked = lm
            h, w = frame.shape[:2]
            cx = sum(p.x for p in (lm[0], lm[5], lm[9], lm[13], lm[17])) / 5.0
            cy = sum(p.y for p in (lm[0], lm[5], lm[9], lm[13], lm[17])) / 5.0
            sx, sy = self.smoother(cx, cy, now - self._last_smooth)
            sx = min(1.0, max(0.0, sx))
            sy = min(1.0, max(0.0, sy))
            gesture, fingers = self._classify(lm)
            state.update(
                hand=True,
                gesture=gesture,
                fingers=fingers,
                x=round(sx, 4),
                y=round(sy, 4),
                event=self._swipe(sx, sy, now),
            )
            self._last_seen = now
            self._last_smooth = now
            if self.debug:
                pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                for (x, y) in pts:
                    cv2.circle(frame, (x, y), 3, (0, 230, 255), -1)
                cv2.circle(frame, (int(sx * w), int(sy * h)), 12, (255, 90, 190), 3)
                cv2.putText(
                    frame,
                    f"{gesture} ({fingers})",
                    (18, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        else:
            if now - self._last_seen > HAND_LOST_MS / 1000:
                self.smoother.reset()
                self._hist.clear()
            state["event"] = None

        self._frames += 1
        self._tick += 1
        if self.preview and self._tick % PREVIEW_EVERY == 0:
            data = encode_preview(frame, tracked)
            if data is not None:
                self.last_preview = data
                self.preview_seq += 1

        span = now - self._fps_mark
        if span >= 1.0:
            self._fps = self._frames / span
            self._frames = 0
            self._fps_mark = now

        if self.debug:
            cv2.imshow("gesture debug", frame)
            cv2.waitKey(1)
        return state

    def take_preview(self) -> str | None:
        """Return a newly encoded preview frame, or None if nothing new."""
        if self.preview_seq == self._sent_seq:
            return None
        self._sent_seq = self.preview_seq
        return self.last_preview

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()
        if self.cap is not None:
            self.cap.release()


class Hub:
    def __init__(self):
        self._clients: set[ServerConnection] = set()
        self._lock = threading.Lock()
        self._state: dict = {"type": "state", "hand": False, "gesture": "none"}

    def handler(self, conn: ServerConnection) -> None:
        with self._lock:
            self._clients.add(conn)
        try:
            conn.send(json.dumps(self._state))
            for _ in conn:
                pass
        except Exception:
            pass
        finally:
            with self._lock:
                self._clients.discard(conn)

    def publish(self, state: dict) -> None:
        self._state = state
        payload = json.dumps(state)
        with self._lock:
            clients = list(self._clients)
        for conn in clients:
            try:
                conn.send(payload)
            except Exception:
                with self._lock:
                    self._clients.discard(conn)

    def publish_frame(self, data: str) -> None:
        with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        payload = json.dumps({"type": "frame", "img": data})
        for conn in clients:
            try:
                conn.send(payload)
            except Exception:
                with self._lock:
                    self._clients.discard(conn)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._clients)


def capture_loop(engine: GestureEngine, hub: Hub, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            hub.publish(engine.read())
            data = engine.take_preview()
            if data:
                hub.publish_frame(data)
        except Exception as exc:
            hub.publish(
                {"type": "state", "hand": False, "gesture": "none", "error": str(exc)}
            )
            time.sleep(0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gesture control bridge for the game")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--debug", action="store_true", help="show the annotated camera window")
    ap.add_argument("--no-preview", action="store_true", help="do not stream video to the game")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    engine = GestureEngine(
        camera=args.camera,
        mirror=not args.no_mirror,
        debug=args.debug,
        preview=not args.no_preview,
    )
    ok, err = engine.open()
    if not ok:
        print(f"[gesture] ERROR {err}")
        print("[gesture] Close other apps using the webcam, then try again.")
        return 1
    print("[gesture] camera + MediaPipe ready")

    hub = Hub()
    stop = threading.Event()

    with serve(hub.handler, HOST, args.port) as server:
        threading.Thread(
            target=capture_loop, args=(engine, hub, stop), daemon=True
        ).start()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        time.sleep(0.2)

        url = f"ws://{HOST}:{args.port}"
        page = Path(__file__).with_name(GAME_FILE)
        print(f"[gesture] websocket listening on {url}")
        print(f"[gesture] open the game: {page}")
        if args.port != PORT:
            print(f"[gesture] WARNING the game page is hardcoded to port {PORT}; "
                  f"it will not connect on {args.port}")
        print("[gesture] ctrl+c to stop")

        if not args.no_browser and page.exists():
            webbrowser.open(page.as_uri())

        last = 0.0
        try:
            while True:
                time.sleep(0.1)
                now = time.time()
                if now - last >= 2.0:
                    last = now
                    print(f"[gesture] clients={hub.count} fps={engine.fps:.1f}")
        except KeyboardInterrupt:
            print("\n[gesture] stopping")
        finally:
            stop.set()
            time.sleep(0.2)
            engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())