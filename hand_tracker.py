"""
Hand tracking module using MediaPipe Hands.

Wraps MediaPipe to detect hand landmarks, count raised fingers,
and return the index fingertip position for air-drawing gestures.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple
import time
import urllib.request

import cv2
import mediapipe as mp

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"


@dataclass
class HandState:
    """Snapshot of one detected hand used by the drawing app."""

    index_tip: Tuple[int, int]          # Pixel coords of index fingertip
    raised_finger_count: int            # Extended fingers (excluding thumb)
    is_drawing_gesture: bool            # Index only raised -> draw
    is_pause_gesture: bool              # Index + middle raised -> stop / select


class HandTracker:
    """Real-time hand landmark detection via MediaPipe Hands."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ) -> None:
        self._use_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

        if self._use_solutions:
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self._Image = None
            self._ImageFormat = None
        else:
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )
            from mediapipe.tasks.python.vision.hand_landmarker import (
                HandLandmarker,
                HandLandmarkerOptions,
            )

            self._Image = Image
            self._ImageFormat = ImageFormat
            self._VisionTaskRunningMode = VisionTaskRunningMode
            self._HandLandmarker = HandLandmarker

            model_path = self._ensure_model_downloaded()
            self._hands = HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path)),
                    running_mode=VisionTaskRunningMode.VIDEO,
                    num_hands=max_hands,
                    min_hand_detection_confidence=detection_confidence,
                    min_hand_presence_confidence=detection_confidence,
                    min_tracking_confidence=tracking_confidence,
                )
            )

    def process(self, frame_bgr: Any) -> Optional[HandState]:
        """
        Run hand detection on a BGR frame.

        Returns HandState for the first detected hand, or None if no hand found.
        """
        # MediaPipe expects RGB input.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._use_solutions:
            frame_rgb.flags.writeable = False
            results = self._hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            if not results.multi_hand_landmarks:
                return None

            landmarks = results.multi_hand_landmarks[0]
        else:
            mp_image = self._Image(self._ImageFormat.SRGB, frame_rgb)
            timestamp_ms = int(time.time() * 1000)
            results = self._hands.detect_for_video(mp_image, timestamp_ms)
            if not results or not results.hand_landmarks:
                return None

            landmarks = results.hand_landmarks[0]
        height, width = frame_bgr.shape[:2]

        index_tip = self._landmark_to_pixel(self._get_landmark(landmarks, 8), width, height)
        fingers = self._finger_states(landmarks)
        raised_count = sum(fingers.values())

        # Precise gestures: index-only draws, index+middle pauses for toolbar use.
        index_up = fingers["index"]
        middle_up = fingers["middle"]
        others_down = not fingers["ring"] and not fingers["pinky"]

        is_drawing = index_up and not middle_up and others_down
        is_pause = index_up and middle_up and others_down

        return HandState(
            index_tip=index_tip,
            raised_finger_count=raised_count,
            is_drawing_gesture=is_drawing,
            is_pause_gesture=is_pause,
        )

    @staticmethod
    def _landmark_to_pixel(landmark: Any, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized [0, 1] landmark coords to pixel coordinates."""
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return x, y

    def _get_landmark(self, landmarks: Any, index: int) -> Any:
        if hasattr(landmarks, "landmark"):
            return landmarks.landmark[index]
        return landmarks[index]

    def _finger_states(self, landmarks: Any) -> dict:
        """
        Return whether index/middle/ring/pinky are extended.

        Thumb is ignored because it often triggers accidentally during pointing.
        """
        lm = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks

        def is_up(tip_idx: int, pip_idx: int) -> bool:
            return lm[tip_idx].y < lm[pip_idx].y

        return {
            "index": is_up(8, 6),
            "middle": is_up(12, 10),
            "ring": is_up(16, 14),
            "pinky": is_up(20, 18),
        }

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    @staticmethod
    def _get_model_path() -> Path:
        return Path(__file__).resolve().parent / MODEL_FILENAME

    def _ensure_model_downloaded(self) -> Path:
        model_path = self._get_model_path()
        if model_path.exists():
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(MODEL_URL) as response, open(model_path, "wb") as output:
                output.write(response.read())
        except Exception as exc:
            raise RuntimeError(
                "Unable to download the MediaPipe hand landmarker model. "
                "Check network connectivity and try again."
            ) from exc

        return model_path
