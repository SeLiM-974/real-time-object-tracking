"""Headless tests for the existing tracking and reporting behavior.

Run from the repository root: python -m unittest discover -s tests -v
Only the Python standard library is needed. Camera, model, drawing, clocks,
and report files are replaced with small test doubles. These tests do not
measure detection accuracy, real-camera compatibility, or runtime performance.
"""

import contextlib
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


TRACKER_PATH = Path(__file__).resolve().parents[1] / "src" / "yolo_tracker.py"


def detection(track_id, x, y=200, class_id=0):
    """Represent the small subset of a YOLO box that the app reads."""
    return SimpleNamespace(
        id=None if track_id is None else [track_id],
        cls=[class_id],
        conf=[0.9],
        xyxy=[[x - 10, y - 20, x + 10, y + 20]],
    )


class ReportFile(io.StringIO):
    """Keep the JSON text in memory when the app closes its output file."""

    def __init__(self, scenario, path):
        super().__init__()
        self.scenario = scenario
        self.path = str(path)

    def close(self):
        if not self.closed:
            self.scenario.saved_files[self.path] = self.getvalue()
        super().close()


class Scenario:
    """Feed deterministic (timestamp, detections) frames through main()."""

    def __init__(self, frames=(), keys=None, *, camera_ok=True,
                 writer_ok=True, inference_error_at=None):
        self.frames = list(frames)
        self.keys = keys or {}
        self.camera_ok = camera_ok
        self.writer_ok = writer_ok
        self.inference_error_at = inference_error_at
        self.index = -1
        self.now = 0.0
        self.width, self.height = 640, 480
        self.camera_opens = 0
        self.writer_opens = 0
        self.model_loads = 0
        self.camera_releases = 0
        self.writer_releases = 0
        self.window_closes = 0
        self.written_frames = []
        self.draws = []
        self.track_options = []
        self.saved_files = {}

    def read(self):
        self.index += 1
        if self.index == len(self.frames):
            self.now += 0.1
            return False, None
        self.now = self.frames[self.index][0]
        return True, SimpleNamespace(index=self.index)

    def set_camera(self, property_id, value):
        if property_id == 3:
            self.width = value
        elif property_id == 4:
            self.height = value
        return True

    def release_camera(self):
        self.camera_releases += 1

    def release_writer(self):
        self.writer_releases += 1

    def close_windows(self):
        self.window_closes += 1

    def capture(self, source):
        self.camera_opens += 1
        return SimpleNamespace(
            isOpened=lambda: self.camera_ok,
            read=self.read,
            set=self.set_camera,
            get=lambda prop: {3: self.width, 4: self.height}[prop],
            release=self.release_camera,
        )

    def writer(self, path, codec, fps, size):
        self.writer_opens += 1
        self.recording_settings = (codec, fps, size)
        return SimpleNamespace(
            isOpened=lambda: self.writer_ok,
            write=lambda frame: self.written_frames.append(frame.index),
            release=self.release_writer,
        )

    def load_model(self, path):
        self.model_loads += 1
        return SimpleNamespace(
            names={0: "person", 39: "bottle"}, track=self.track,
        )

    def track(self, frame, **options):
        self.track_options.append(options)
        if self.index == self.inference_error_at:
            raise RuntimeError("Simulated inference failure")
        return [SimpleNamespace(boxes=self.frames[self.index][1])]

    def open_report(self, path, mode, encoding):
        if mode != "w" or encoding != "utf-8":
            raise AssertionError("Expected a UTF-8 text report")
        return ReportFile(self, path)

    def load(self):
        """Import the real module with no optional dependencies or side effects."""
        cv2 = ModuleType("cv2")
        cv2.CAP_PROP_FRAME_WIDTH = 3
        cv2.CAP_PROP_FRAME_HEIGHT = 4
        cv2.FONT_HERSHEY_SIMPLEX = 0
        cv2.VideoCapture = self.capture
        cv2.VideoWriter = self.writer
        cv2.VideoWriter_fourcc = lambda *letters: "".join(letters)
        cv2.flip = lambda frame, axis: frame
        cv2.waitKey = lambda delay: self.keys.get(self.index, -1)
        cv2.destroyAllWindows = self.close_windows
        for name in ("line", "rectangle", "putText", "arrowedLine", "imshow"):
            def draw(frame, *args, _name=name, **kwargs):
                self.draws.append((frame.index, _name, args))
            setattr(cv2, name, draw)

        numpy = ModuleType("numpy")
        numpy.hypot = math.hypot
        ultralytics = ModuleType("ultralytics")
        ultralytics.YOLO = self.load_model

        with patch.dict(sys.modules, {
            "cv2": cv2, "numpy": numpy, "ultralytics": ultralytics,
        }):
            spec = importlib.util.spec_from_file_location("tracker_under_test", TRACKER_PATH)
            module = importlib.util.module_from_spec(spec)
            # Compile source directly: do not create an import bytecode cache.
            source = TRACKER_PATH.read_text(encoding="utf-8-sig")
            exec(compile(source, str(TRACKER_PATH), "exec"), module.__dict__)

        scenario = self

        class TestDateTime:
            @staticmethod
            def now():
                return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
                    seconds=scenario.now
                )

        module.time = SimpleNamespace(perf_counter=lambda: self.now)
        module.datetime = TestDateTime
        module.open = self.open_report
        return module

    def run(self, **settings):
        module = self.load()
        for name, value in settings.items():
            setattr(module, name, value)
        with contextlib.redirect_stdout(io.StringIO()):
            module.main()
        return self.report()

    def report(self):
        if len(self.saved_files) != 1:
            raise AssertionError("Expected exactly one in-memory report")
        return json.loads(next(iter(self.saved_files.values())))

    def labels(self, frame_index):
        return [args[0] for index, name, args in self.draws
                if index == frame_index and name == "putText"]


class TrackerTests(unittest.TestCase):
    def assert_released(self, scenario, writer_expected=True):
        self.assertEqual(scenario.camera_releases, 1)
        self.assertEqual(scenario.writer_releases, int(writer_expected))
        self.assertEqual(scenario.window_closes, 1)

    def test_import_does_not_open_camera_load_model_or_save_report(self):
        scenario = Scenario()
        module = scenario.load()
        self.assertTrue(callable(module.main))
        self.assertEqual(scenario.camera_opens, 0)
        self.assertEqual(scenario.writer_opens, 0)
        self.assertEqual(scenario.model_loads, 0)
        self.assertEqual(scenario.saved_files, {})

    def test_multi_id_motion_and_report(self):
        scenario = Scenario([
            (0.1 * (i + 1), [detection(1, 100 + i * 10),
                              detection(2, 300, 200 + i * 8, 39)])
            for i in range(7)
        ])
        report = scenario.run()
        self.assertEqual(report["session"]["unique_track_count"], 2)
        self.assertAlmostEqual(report["session"]["duration_seconds"], 0.8)
        for key in ("started_at", "ended_at"):
            self.assertIsNotNone(datetime.fromisoformat(report["session"][key]).tzinfo)
        for track_id, name, distance, speed in (
            ("1", "person", 60.0, 100.0), ("2", "bottle", 48.0, 80.0),
        ):
            track = report["tracks"][track_id]
            self.assertEqual(track["class_name"], name)
            self.assertEqual(track["frames_tracked"], 7)
            self.assertAlmostEqual(track["first_seen"], 0.1)
            self.assertAlmostEqual(track["last_seen"], 0.7)
            self.assertAlmostEqual(track["lifetime_seconds"], 0.6)
            self.assertAlmostEqual(track["measured_seconds"], 0.6)
            self.assertAlmostEqual(track["total_distance_px"], distance)
            self.assertAlmostEqual(track["average_speed_px_s"], speed)
            self.assertAlmostEqual(track["max_smoothed_speed_px_s"], speed)
            self.assertEqual(track["temporary_loss_count"], 0)
        self.assertIn("Direction: Right", scenario.labels(6))
        self.assertIn("Direction: Down", scenario.labels(6))
        self.assertIn("Active tracks: 2", scenario.labels(6))
        self.assertEqual(scenario.written_frames, list(range(7)))
        self.assertTrue(all(options == {"persist": True, "conf": 0.5}
                            for options in scenario.track_options))
        self.assert_released(scenario)

    def test_gap_skips_unobserved_jump_and_counts_one_temporary_loss(self):
        scenario = Scenario([
            (0.1, [detection(1, 100)]), (0.2, [detection(1, 110)]),
            (0.3, []), (0.4, []), (0.5, [detection(1, 900)]),
            (0.6, [detection(1, 910)]),
        ])
        track = scenario.run()["tracks"]["1"]
        self.assertEqual(track["frames_tracked"], 4)
        self.assertEqual(track["temporary_loss_count"], 1)
        self.assertAlmostEqual(track["total_distance_px"], 20.0)
        self.assertAlmostEqual(track["measured_seconds"], 0.2)
        self.assertAlmostEqual(track["average_speed_px_s"], 100.0)
        self.assertAlmostEqual(track["lifetime_seconds"], 0.5)
        self.assertIn("0.00 px/s", scenario.labels(4))
        self.assertFalse(any(index == 4 and name == "line"
                             for index, name, args in scenario.draws))

    def test_timeout_drops_live_history_but_retains_report(self):
        scenario = Scenario([
            (0.1, [detection(1, 100)]), (0.2, [detection(1, 110)]),
            (2.3, []), (2.4, [detection(1, 900)]),
            (2.5, [detection(1, 910)]), (4.6, []),
        ])
        report = scenario.run()
        track = report["tracks"]["1"]
        self.assertEqual(report["session"]["unique_track_count"], 1)
        self.assertEqual(track["frames_tracked"], 4)
        self.assertEqual(track["temporary_loss_count"], 0)
        self.assertAlmostEqual(track["total_distance_px"], 20.0)
        self.assertAlmostEqual(track["first_seen"], 0.1)
        self.assertAlmostEqual(track["last_seen"], 2.5)
        self.assertIn("0.0", scenario.labels(3))  # Live distance restarts.
        self.assertIn("Active tracks: 0", scenario.labels(5))

    def test_reset_restarts_measurements_and_timer_not_the_model(self):
        scenario = Scenario([
            (1.1, [detection(1, 100)]), (1.2, [detection(1, 110)]),
            (1.3, [detection(1, 900)]), (1.4, [detection(1, 910)]),
            (1.5, [detection(2, 500)]),
        ], keys={1: ord("r"), 3: ord("q")})
        report = scenario.run()
        track = report["tracks"]["1"]
        self.assertEqual(set(report["tracks"]), {"1"})
        self.assertEqual(track["frames_tracked"], 2)
        self.assertAlmostEqual(track["total_distance_px"], 10.0)
        self.assertAlmostEqual(track["first_seen"], 0.1)
        self.assertAlmostEqual(report["session"]["duration_seconds"], 0.2)
        self.assertEqual(datetime.fromisoformat(report["session"]["started_at"])
                         .astimezone(timezone.utc).second, 1)
        self.assertEqual(scenario.model_loads, 1)
        self.assertEqual(scenario.written_frames, [0, 1, 2, 3])
        self.assert_released(scenario)

    def test_class_filter_and_idless_boxes_are_not_counted(self):
        scenario = Scenario([
            (0.1, [detection(None, 100), detection(1, 200),
                   detection(2, 300, class_id=39)]),
            (0.2, []), (0.3, [detection(2, 400, class_id=39)]),
        ])
        report = scenario.run(ALLOWED_CLASSES=[39])
        self.assertEqual(set(report["tracks"]), {"2"})
        self.assertIn("Active tracks: 1", scenario.labels(0))
        self.assertIn("Active tracks: 0", scenario.labels(1))

    def test_empty_detections_still_write_video_and_empty_report(self):
        scenario = Scenario([(0.1, []), (0.2, [])])
        report = scenario.run()
        self.assertEqual(report["tracks"], {})
        self.assertEqual(report["session"]["unique_track_count"], 0)
        self.assertEqual(scenario.written_frames, [0, 1])
        self.assert_released(scenario)

    def test_zero_elapsed_does_not_divide_or_accumulate_distance(self):
        scenario = Scenario([
            (0.1, [detection(1, 100)]), (0.1, [detection(1, 110)]),
            (0.2, [detection(1, 120)]),
        ])
        track = scenario.run()["tracks"]["1"]
        self.assertAlmostEqual(track["total_distance_px"], 10.0)
        self.assertAlmostEqual(track["measured_seconds"], 0.1)
        self.assertAlmostEqual(track["average_speed_px_s"], 100.0)

    def test_trail_and_speed_windows_are_bounded(self):
        frames = [(0.1 * (i + 1), [detection(1, 100 + i * (i + 1) // 2)])
                  for i in range(40)]
        scenario = Scenario(frames)
        track = scenario.run()["tracks"]["1"]
        final_segments = [args for index, name, args in scenario.draws
                          if index == 39 and name == "line"]
        self.assertEqual(len(final_segments), 29)
        self.assertIn("345.00 px/s", scenario.labels(39))
        self.assertAlmostEqual(track["max_smoothed_speed_px_s"], 345.0)
        self.assertAlmostEqual(track["average_speed_px_s"], 200.0)

    def test_fps_panel_uses_only_recent_window(self):
        times = [0.1 * (i + 1) for i in range(5)]
        times += [0.5 + 0.2 * (i + 1) for i in range(35)]
        scenario = Scenario([(time, []) for time in times])
        scenario.run()
        self.assertIn("FPS: 5.0", scenario.labels(39))

    def test_inference_failure_releases_resources_and_propagates(self):
        scenario = Scenario([(0.1, [])], inference_error_at=0)
        with self.assertRaisesRegex(RuntimeError, "inference failure"):
            scenario.run()
        self.assertEqual(scenario.saved_files, {})
        self.assert_released(scenario)

    def test_writer_failure_releases_resources(self):
        scenario = Scenario(writer_ok=False)
        with self.assertRaisesRegex(RuntimeError, "video writer"):
            scenario.run()
        self.assertEqual(scenario.written_frames, [])
        self.assert_released(scenario)

    def test_camera_failure_stops_before_writer(self):
        scenario = Scenario(camera_ok=False)
        with self.assertRaisesRegex(RuntimeError, "camera"):
            scenario.run()
        self.assertEqual(scenario.writer_opens, 0)
        self.assert_released(scenario, writer_expected=False)


if __name__ == "__main__":
    unittest.main()
