"""
Air Drawing — virtual whiteboard with hand gestures + mouse toolbar control.

Hand gestures:
  - 1 finger (index only)  -> draw on canvas
  - 2 fingers (index+middle) -> pause drawing; point at toolbar to pick colors / clear

Mouse (toolbar only):
  - Click a color swatch -> change drawing color
  - Click Clear -> erase canvas

Keys: q / ESC -> quit
"""

import time
import urllib.parse
import webbrowser
from typing import List, Tuple

import cv2
import numpy as np

from drawing_utils import TOOLBAR_HEIGHT, DrawingCanvas
from hand_tracker import HandTracker

WINDOW_NAME = "Air Draw"

PREFERRED_RESOLUTIONS: List[Tuple[int, int]] = [
    (1920, 1080),
    (1280, 720),
    (1280, 800),
    (960, 540),
]


class FPSCounter:
    def __init__(self, smooth_factor: float = 0.9) -> None:
        self._smooth = smooth_factor
        self._fps = 0.0
        self._prev_time = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            instant = 1.0 / dt
            self._fps = self._smooth * self._fps + (1.0 - self._smooth) * instant
        return self._fps


class HighQualityCamera:
    def __init__(self) -> None:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions.")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.width, self.height = self._negotiate_resolution(cap)
        self._cap = cap
        self._buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)
        print(f"Camera: {self.width}x{self.height}")

    @staticmethod
    def _negotiate_resolution(cap: cv2.VideoCapture) -> Tuple[int, int]:
        best_w, best_h = 0, 0
        for target_w, target_h in PREFERRED_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            actual_h, actual_w = frame.shape[:2]
            if actual_w * actual_h > best_w * best_h:
                best_w, best_h = actual_w, actual_h
            if actual_w >= 1280 and actual_h >= 720:
                break
        if best_w == 0:
            ok, frame = cap.read()
            if ok and frame is not None:
                best_h, best_w = frame.shape[:2]
            else:
                best_w, best_h = 1280, 720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_h)
        return best_w, best_h

    def read(self) -> Tuple[bool, np.ndarray]:
        ok, raw = self._cap.read()
        if not ok or raw is None:
            return False, self._buffer
        h, w = raw.shape[:2]
        if w != self.width or h != self.height:
            cv2.resize(
                raw,
                (self.width, self.height),
                dst=self._buffer,
                interpolation=cv2.INTER_AREA if w > self.width else cv2.INTER_CUBIC,
            )
        else:
            np.copyto(self._buffer, raw)
        cv2.flip(self._buffer, 1, dst=self._buffer)
        return True, self._buffer

    def release(self) -> None:
        self._cap.release()


def apply_canvas_tint(frame: np.ndarray) -> None:
    tinted = frame.copy()
    cv2.rectangle(
        tinted,
        (0, TOOLBAR_HEIGHT),
        (frame.shape[1], frame.shape[0]),
        (30, 28, 26),
        -1,
    )
    cv2.addWeighted(tinted, 0.10, frame, 0.90, 0, frame)


def draw_search_overlay(frame: np.ndarray, query: str) -> None:
    prompt = "Google Search: " + query
    background = frame.copy()
    cv2.rectangle(background, (10, 90), (frame.shape[1] - 10, 140), (20, 20, 20), -1)
    cv2.addWeighted(background, 0.75, frame, 0.25, 0, frame)
    cv2.putText(
        frame,
        prompt,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Press Enter to search, Backspace to edit, Escape to cancel.",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (190, 190, 190),
        1,
        lineType=cv2.LINE_AA,
    )


def search_google(query: str) -> None:
    if not query.strip():
        return
    url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query.strip())}"
    webbrowser.open(url)


def on_mouse(event: int, x: int, y: int, flags: int, canvas: DrawingCanvas) -> None:
    """Mouse handler: toolbar clicks for colors and clear; hover highlights."""
    if y <= TOOLBAR_HEIGHT:
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            canvas.set_toolbar_hover((x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            canvas.end_stroke()
            canvas.handle_toolbar_point((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        canvas.set_toolbar_hover(None)


def main() -> None:
    camera = HighQualityCamera()
    tracker = HandTracker()
    canvas = DrawingCanvas(width=camera.width, height=camera.height)
    fps_counter = FPSCounter()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, canvas)

    was_drawing = False
    search_mode = False
    search_query = ""

    print("Air Draw started.")
    print("  Hand: 1 finger = draw | 2 fingers = pause + point at toolbar")
    print("  Mouse: click toolbar colors or Clear anytime")
    print("  Keyboard: c = clear, s = Google search, 1-7 = select color")
    print("  q / ESC = quit")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        apply_canvas_tint(frame)

        hand = tracker.process(frame)
        mode_label = "Mode: Ready"
        is_drawing = False

        if hand is not None:
            tip = hand.index_tip

            if hand.is_drawing_gesture:
                mode_label = "Mode: Drawing"
                is_drawing = True
                canvas.set_toolbar_hover(None)
                canvas.add_stroke_point(tip)

            elif hand.is_pause_gesture:
                mode_label = "Mode: Select"
                canvas.end_stroke()
                canvas.set_toolbar_hover(tip)
                canvas.handle_toolbar_point(tip)

            else:
                mode_label = "Mode: Idle"
                canvas.end_stroke()
                canvas.set_toolbar_hover(None)

            canvas.draw_cursor(frame, tip, drawing=is_drawing)

            if was_drawing and not is_drawing:
                canvas.end_stroke()

            was_drawing = is_drawing
        else:
            canvas.end_stroke()
            was_drawing = False
            mode_label = "Mode: No hand"

        canvas.composite(frame)
        canvas.draw_toolbar(frame)

        fps = fps_counter.tick()
        canvas.draw_status(
            frame,
            fps=fps,
            mode_label=mode_label,
            hand_detected=hand is not None,
            resolution=f"{camera.width}x{camera.height}",
        )

        if search_mode:
            draw_search_overlay(frame, search_query)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1)
        if key == -1:
            continue

        key_code = key & 0xFF
        if search_mode:
            if key_code in (13, 10):
                if search_query.strip():
                    print(f"Searching Google for: {search_query}")
                    search_google(search_query)
                else:
                    print("Search cancelled because query was empty.")
                search_mode = False
                search_query = ""
                continue
            if key_code in (8, 127):
                search_query = search_query[:-1]
                continue
            if key_code == 27:
                search_mode = False
                search_query = ""
                print("Google search cancelled.")
                continue
            if 32 <= key_code <= 126 and len(search_query) < 128:
                search_query += chr(key_code)
            continue

        if key_code in (ord("q"), 27):
            break
        if key_code == ord("c"):
            canvas.clear()
            print("Canvas cleared.")
        elif key_code == ord("s"):
            search_mode = True
            search_query = ""
            print("Type your search query and press Enter.")
        elif key_code in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7")):
            index = int(chr(key_code)) - 1
            if 0 <= index < len(canvas.colors):
                canvas.selected_color_index = index
                print(f"Selected color: {canvas.colors[index].name}")

    camera.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
