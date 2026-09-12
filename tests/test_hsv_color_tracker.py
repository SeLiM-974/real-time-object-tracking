"""Test baseline semantics with lightweight stubs; no camera or video IO."""

import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


RELEASE_PATH = Path(__file__).resolve().parents[1] / "baselines" / "hsv_color_tracker.py"


class FakeFrame:
    def __init__(self, index):
        self.index = index

    def copy(self):
        return FakeFrame(self.index)


class FakeCapture:
    def __init__(self, backend):
        self.backend = backend
        self.next_index = 0

    def isOpened(self):
        return self.backend.camera_open

    def get(self, prop):
        return {1: 640, 2: 480}[prop]

    def read(self):
        if self.next_index >= len(self.backend.frames):
            return False, None
        frame = FakeFrame(self.next_index)
        self.next_index += 1
        return True, frame

    def release(self):
        self.backend.camera_released = True


class FakeWriter:
    def __init__(self, backend):
        self.backend = backend

    def isOpened(self):
        return self.backend.writer_open

    def write(self, frame):
        self.backend.log("write", frame)

    def release(self):
        self.backend.writer_released = True


class FakeCV2(types.ModuleType):
    CAP_PROP_FRAME_WIDTH = 1
    CAP_PROP_FRAME_HEIGHT = 2
    COLOR_BGR2HSV = 3
    RETR_EXTERNAL = 4
    CHAIN_APPROX_SIMPLE = 5
    FONT_HERSHEY_SIMPLEX = 6

    def __init__(self, frames, keys=None, camera_open=True, writer_open=True, fail_frame=None):
        super().__init__("cv2")
        self.frames = frames
        self.keys = iter(keys or [-1] * len(frames))
        self.camera_open = camera_open
        self.writer_open = writer_open
        self.fail_frame = fail_frame
        self.events = []
        self.capture_calls = []
        self.writer_calls = []
        self.camera_released = False
        self.writer_released = False
        self.windows_destroyed = False

    def VideoCapture(self, source):
        self.capture_calls.append(source)
        return FakeCapture(self)

    def VideoWriter_fourcc(self, *codec):
        return "".join(codec)

    def VideoWriter(self, path, codec, fps, dimensions):
        self.writer_calls.append((Path(path).name, codec, fps, dimensions))
        return FakeWriter(self)

    def log(self, name, frame, *args, **kwargs):
        self.events.append((name, frame.index, args, kwargs))

    def flip(self, frame, flip_code):
        self.log("flip", frame, flip_code)
        return frame

    def cvtColor(self, frame, code):
        self.log("cvtColor", frame, code)
        return frame

    def inRange(self, frame, lower, upper):
        self.log("inRange", frame, lower, upper)
        return frame

    def findContours(self, frame, mode, method):
        self.log("findContours", frame, mode, method)
        if frame.index == self.fail_frame:
            raise RuntimeError("Synthetic processing failure")
        return self.frames[frame.index], None

    def contourArea(self, contour):
        return contour[0]

    def boundingRect(self, contour):
        return contour[1]

    def putText(self, frame, *args, **kwargs):
        self.log("putText", frame, *args, **kwargs)

    def arrowedLine(self, frame, *args, **kwargs):
        self.log("arrowedLine", frame, *args, **kwargs)

    def line(self, frame, *args, **kwargs):
        self.log("line", frame, *args, **kwargs)

    def rectangle(self, frame, *args, **kwargs):
        self.log("rectangle", frame, *args, **kwargs)

    def circle(self, frame, *args, **kwargs):
        self.log("circle", frame, *args, **kwargs)

    def imshow(self, title, frame):
        self.log("imshow", frame, title)

    def waitKey(self, delay):
        return next(self.keys, -1)

    def destroyAllWindows(self):
        self.windows_destroyed = True


def contour(center_x, center_y, area=900):
    return area, (center_x - 10, center_y - 10, 20, 20)


def execute(path, backend, ticks=None, run_main=True):
    fake_np = types.ModuleType("numpy")
    fake_np.array = tuple
    fake_np.hypot = math.hypot
    fake_time = types.ModuleType("time")
    clock = iter(ticks if ticks is not None else [i / 20 for i in range(500)])
    fake_time.perf_counter = lambda: next(clock)
    namespace = {"__name__": "hsv_test_module", "__file__": str(path.resolve())}
    with patch.dict(sys.modules, {"cv2": backend, "numpy": fake_np, "time": fake_time}):
        exec(compile(path.read_text(encoding="utf-8-sig"), str(path), "exec"), namespace)
        if run_main and "main" in namespace:
            namespace["main"]()
    return namespace


class HSVReleaseTests(unittest.TestCase):
    def exercise(self, frames, keys=None, ticks=None):
        backend = FakeCV2(frames, keys)
        execute(RELEASE_PATH, backend, ticks)
        self.assertEqual(backend.capture_calls, [0])
        self.assertEqual(
            backend.writer_calls,
            [("tracking_output.mp4", "mp4v", 30, (640, 480))],
        )
        self.assertTrue(backend.camera_released)
        self.assertTrue(backend.writer_released)
        self.assertTrue(backend.windows_destroyed)
        return backend

    def test_largest_contour_and_area_threshold(self):
        frames = [[], [contour(10, 10, 500)], [contour(30, 40, 501)],
                  [contour(400, 300, 600), contour(70, 80, 1200)]]
        result = self.exercise(frames)
        rectangles = [event for event in result.events if event[0] == "rectangle"]
        self.assertEqual(len(rectangles), 2)
        self.assertEqual(rectangles[-1][2][0], (60, 70))

    def test_distance_threshold_boundary(self):
        result = self.exercise([[contour(x, 100)] for x in (100, 101, 103)])
        distance_labels = [event[2][0] for event in result.events
                           if event[0] == "putText" and event[2][0].startswith("Total Distance")]
        self.assertEqual(distance_labels, ["Total Distance: 0.0", "Total Distance: 0.0", "Total Distance: 2.0"])

    def test_motion_directions_and_trail_window(self):
        for dx, dy, expected in [(3, 0, "Right"), (-3, 0, "Left"),
                                 (0, 3, "Down"), (0, -3, "Up"),
                                 (3, 3, "Down"), (0, 0, "Still")]:
            with self.subTest(direction=expected, dx=dx, dy=dy):
                result = self.exercise([[contour(200 + dx * i, 200 + dy * i)] for i in range(40)])
                labels = [event[2][0] for event in result.events if event[0] == "putText"]
                self.assertIn(f"Direction: {expected}", labels)
                # The original draws even a zero-length arrow for Still.
                self.assertEqual(sum(event[0] == "arrowedLine" for event in result.events), 35)

    def test_short_loss_keeps_bridge(self):
        frames = [[contour(100, 100)], [contour(105, 100)], [], [], [contour(200, 100)]]
        result = self.exercise(frames)
        distances = [event[2][0] for event in result.events
                     if event[0] == "putText" and event[2][0].startswith("Total Distance")]
        self.assertEqual(distances[-1], "Total Distance: 100.0")
        self.assertTrue(any(event[0] == "line" and event[2][:2] == ((105, 100), (200, 100)) for event in result.events))

    def test_ten_frame_loss_resets_history_not_distance(self):
        frames = [[contour(100, 100)], [contour(105, 100)]] + [[] for _ in range(11)]
        frames += [[contour(400, 100)], [contour(405, 100)]]
        result = self.exercise(frames)
        labels = [event[2][0] for event in result.events if event[0] == "putText"]
        self.assertEqual(labels.count("LOST"), 10)
        self.assertEqual(labels.count("SEARCHING"), 1)
        self.assertIn("Total Distance: 10.0", labels)
        self.assertFalse(any(event[0] == "line" and event[2][:2] == ((105, 100), (400, 100)) for event in result.events))

    def test_reset_then_quit_preserves_write_order(self):
        frames = [[contour(100 + i * 10, 100)] for i in range(5)]
        result = self.exercise(frames, [-1, ord("r"), -1, ord("q"), -1])
        distances = [event[2][0] for event in result.events
                     if event[0] == "putText" and event[2][0].startswith("Total Distance")]
        self.assertEqual(distances, ["Total Distance: 0.0", "Total Distance: 10.0", "Total Distance: 0.0", "Total Distance: 10.0"])
        self.assertEqual(sum(event[0] == "write" for event in result.events), 4)

    def test_zero_elapsed_keeps_previous_measurements(self):
        frames = [[contour(100 + i * 2, 100)] for i in range(4)]
        result = self.exercise(frames, ticks=[0.0, 0.0, 0.0, 0.5, 0.5])
        fps_labels = [event[2][0] for event in result.events
                      if event[0] == "putText" and event[2][0].startswith("FPS:")]
        self.assertEqual(fps_labels, ["FPS: 0.0", "FPS: 0.0", "FPS: 2.0", "FPS: 2.0"])

    def test_speed_smoothing_includes_small_steps(self):
        result = self.exercise(
            [[contour(x, 100)] for x in (100, 101, 103)],
            ticks=[0.0, 0.5, 1.0, 1.5],
        )
        speed_labels = [event[2][0] for event in result.events
                        if event[0] == "putText" and event[2][0].startswith("Speed:")]
        self.assertEqual(speed_labels, ["Speed: 0.0 px/s", "Speed: 2.0 px/s", "Speed: 3.0 px/s"])

    def test_import_does_not_open_camera_or_read_clock(self):
        backend = FakeCV2([])
        execute(RELEASE_PATH, backend, ticks=[], run_main=False)
        self.assertEqual(backend.capture_calls, [])
        self.assertEqual(backend.writer_calls, [])

    def test_camera_open_failure_cleanup(self):
        backend = FakeCV2([], camera_open=False)
        with self.assertRaisesRegex(RuntimeError, "Could not open camera"):
            execute(RELEASE_PATH, backend)
        self.assertTrue(backend.camera_released)
        self.assertTrue(backend.windows_destroyed)
        self.assertEqual(backend.writer_calls, [])

    def test_writer_open_failure_cleanup(self):
        backend = FakeCV2([], writer_open=False)
        with self.assertRaisesRegex(RuntimeError, "Could not open video writer"):
            execute(RELEASE_PATH, backend)
        self.assertTrue(backend.camera_released)
        self.assertTrue(backend.writer_released)
        self.assertTrue(backend.windows_destroyed)

    def test_processing_failure_cleanup(self):
        backend = FakeCV2([[contour(100, 100)]], fail_frame=0)
        with self.assertRaisesRegex(RuntimeError, "Synthetic processing failure"):
            execute(RELEASE_PATH, backend)
        self.assertTrue(backend.camera_released)
        self.assertTrue(backend.writer_released)
        self.assertTrue(backend.windows_destroyed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
