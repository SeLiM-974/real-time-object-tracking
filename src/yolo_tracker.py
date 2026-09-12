"""Track webcam objects with YOLO and report their image-plane motion.

Speed and distance are measured in pixels, not physical units.
Press Q to quit or R to reset the current measurement session.
"""

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# This file lives in src/; models and generated outputs stay at the repo root.
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "yolov8n.pt"
VIDEO_OUTPUT_PATH = PROJECT_DIR / "yolo_tracking_output.mp4"

# Camera and recording settings.
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
VIDEO_CODEC = "mp4v"
VIDEO_FPS = 30  # Fixed playback rate; this is not the measured processing FPS.

# Detection and motion settings.
CONF_THRESHOLD = 0.5
ALLOWED_CLASSES = None  # None allows all classes; [0] allows only people.
TRAIL_LENGTH = 30
SPEED_WINDOW = 10
FPS_WINDOW = 30
DIRECTION_WINDOW = 6
MOTION_THRESHOLD = 10.0  # Pixels per second.
TRACK_TIMEOUT = 2.0  # Seconds before removing an unseen ID's live history.

# OpenCV colors use BGR order.
COLORS = [
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
]


def draw_text(frame, text, position):
    """Draw an overlay label with the shared text style."""
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2,
    )


def save_session_report(session_summary, session_start, session_started_at):
    """Save session metadata and per-ID summaries to a timestamped JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    now = time.perf_counter()
    session_duration = now - session_start
    session_ended_at = datetime.now().astimezone().isoformat(timespec="seconds")

    report = {
        "session": {
            "started_at": session_started_at,
            "ended_at": session_ended_at,
            "duration_seconds": session_duration,
            # Track IDs are not guaranteed to represent unique physical objects.
            "unique_track_count": len(session_summary),
        },
        "tracks": session_summary,
    }

    report_path = PROJECT_DIR / f"session_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)


def main():
    """Run capture, tracking, motion analysis, drawing, and recording."""
    model = YOLO(str(MODEL_PATH))

    # Live data is removed when an ID times out.
    trails = {}
    total_distances = {}
    last_seen_times = {}
    last_seen_frames = {}
    speeds = {}
    speed_histories = {}
    average_speeds = {}
    directions = {}
    track_colors = {}

    # Report data survives track timeouts, but is cleared by R.
    session_summary = {}

    frame_index = 0
    previous_frame_time = None
    fps_history = deque(maxlen=FPS_WINDOW)
    average_fps = 0.0

    cap = cv2.VideoCapture(CAMERA_INDEX)
    writer = None

    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {frame_width} x {frame_height}")

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        writer = cv2.VideoWriter(
            str(VIDEO_OUTPUT_PATH), fourcc, VIDEO_FPS,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not open video writer")

        session_start = time.perf_counter()
        session_started_at = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            now = time.perf_counter()

            # Processing FPS, smoothed over recent frames.
            if previous_frame_time is not None:
                frame_elapsed = now - previous_frame_time
                if frame_elapsed > 0:
                    fps = 1 / frame_elapsed
                    fps_history.append(fps)
                    average_fps = sum(fps_history) / len(fps_history)

            previous_frame_time = now

            # Remove stale live histories without deleting their report entries.
            expired_ids = []
            for saved_id, last_seen in last_seen_times.items():
                if now - last_seen > TRACK_TIMEOUT:
                    expired_ids.append(saved_id)

            for expired_id in expired_ids:
                trails.pop(expired_id, None)
                total_distances.pop(expired_id, None)
                last_seen_times.pop(expired_id, None)
                last_seen_frames.pop(expired_id, None)
                speeds.pop(expired_id, None)
                speed_histories.pop(expired_id, None)
                average_speeds.pop(expired_id, None)
                directions.pop(expired_id, None)
                track_colors.pop(expired_id, None)

            # Detect and track objects in the current frame.
            frame = cv2.flip(frame, 1)
            results = model.track(
                frame, persist=True, conf=CONF_THRESHOLD
            )
            boxes = results[0].boxes
            active_count = 0

            for box in boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if (
                    ALLOWED_CLASSES is not None
                    and class_id not in ALLOWED_CLASSES
                ):
                    continue

                active_count += 1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Initialize this ID's live history and persistent summary.
                if track_id not in trails:
                    trails[track_id] = deque(maxlen=TRAIL_LENGTH)
                    total_distances[track_id] = 0.0
                    speeds[track_id] = 0.0
                    speed_histories[track_id] = deque(maxlen=SPEED_WINDOW)
                    average_speeds[track_id] = 0.0

                if track_id not in session_summary:
                    session_summary[track_id] = {
                        "class_name": class_name,
                        "first_seen": now - session_start,
                        "last_seen": now - session_start,
                        "lifetime_seconds": 0.0,
                        "frames_tracked": 0,
                        "total_distance_px": 0.0,
                        "max_smoothed_speed_px_s": 0.0,
                        "measured_seconds": 0.0,
                        "average_speed_px_s": 0.0,
                        "temporary_loss_count": 0,
                    }

                # Short names refer to the same per-ID data, not copies.
                trail = trails[track_id]
                track_summary = session_summary[track_id]
                speed_history = speed_histories[track_id]

                # Do not measure a jump across frames where this ID was absent.
                if track_id in last_seen_frames:
                    frame_gap = frame_index - last_seen_frames[track_id]
                    if frame_gap > 1:
                        track_summary["temporary_loss_count"] += 1
                        trail.clear()
                        speed_history.clear()
                        speeds[track_id] = 0.0
                        average_speeds[track_id] = 0.0

                trail.append((center_x, center_y))
                track_summary["last_seen"] = now - session_start
                track_summary["lifetime_seconds"] = (
                    track_summary["last_seen"] - track_summary["first_seen"]
                )
                track_summary["frames_tracked"] += 1

                if track_id not in track_colors:
                    color_index = track_id % len(COLORS)
                    track_colors[track_id] = COLORS[color_index]

                color = track_colors[track_id]
                for i in range(1, len(trail)):
                    thickness = int(i / len(trail) * 5) + 1
                    cv2.line(frame, trail[i - 1], trail[i], color, thickness)

                # Measure only consecutive observations of the same ID.
                if len(trail) >= 2:
                    previous_center = trail[-2]
                    last_center = trail[-1]
                    dx = last_center[0] - previous_center[0]
                    dy = last_center[1] - previous_center[1]
                    step_distance = np.hypot(dx, dy)
                    elapsed = now - last_seen_times[track_id]

                    if elapsed > 0:
                        total_distances[track_id] += step_distance
                        track_summary["total_distance_px"] += step_distance
                        track_summary["measured_seconds"] += elapsed
                        track_summary["average_speed_px_s"] = (
                            track_summary["total_distance_px"]
                            / track_summary["measured_seconds"]
                        )

                        speeds[track_id] = step_distance / elapsed
                        speed_history.append(speeds[track_id])
                        average_speeds[track_id] = (
                            sum(speed_history) / len(speed_history)
                        )

                        if (
                            average_speeds[track_id]
                            > track_summary["max_smoothed_speed_px_s"]
                        ):
                            track_summary["max_smoothed_speed_px_s"] = (
                                average_speeds[track_id]
                            )

                last_seen_times[track_id] = now
                last_seen_frames[track_id] = frame_index

                # Use a longer displacement window to reduce direction jitter.
                direction = "Waiting"
                if len(trail) >= DIRECTION_WINDOW:
                    old_center = trail[-DIRECTION_WINDOW]
                    current_center = trail[-1]
                    direction_dx = current_center[0] - old_center[0]
                    direction_dy = current_center[1] - old_center[1]

                    if average_speeds[track_id] < MOTION_THRESHOLD:
                        direction = "Still"
                    elif direction_dx == 0 and direction_dy == 0:
                        direction = "Still"
                    elif abs(direction_dx) > abs(direction_dy):
                        direction = "Right" if direction_dx > 0 else "Left"
                    else:
                        direction = "Down" if direction_dy > 0 else "Up"

                    if direction != "Still":
                        cv2.arrowedLine(
                            frame, old_center, current_center,
                            (255, 0, 0), 4, tipLength=0.4,
                        )

                directions[track_id] = direction

                # Object overlays.
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_text(
                    frame, f"{average_speeds[track_id]:.2f} px/s",
                    (x1, max(20, y1 + 45)),
                )
                draw_text(
                    frame, f"ID: {track_id} | {class_name} | {confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                )
                draw_text(
                    frame, f"{total_distances[track_id]:.1f}",
                    (x1, max(20, y1 + 20)),
                )
                draw_text(
                    frame, f"Direction: {directions[track_id]}",
                    (x1, max(20, y1 + 70)),
                )

            # Global panel is drawn last so object overlays cannot cover it.
            cv2.rectangle(frame, (10, 10), (260, 115), (30, 30, 30), -1)
            draw_text(frame, f"Active tracks: {active_count}", (20, 60))
            draw_text(frame, f"FPS: {average_fps:.1f}", (20, 30))

            writer.write(frame)
            cv2.imshow("result", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                # Reset our measurements, not YOLO's internal tracking IDs.
                trails.clear()
                total_distances.clear()
                last_seen_times.clear()
                last_seen_frames.clear()
                speeds.clear()
                speed_histories.clear()
                average_speeds.clear()
                directions.clear()
                track_colors.clear()
                session_summary.clear()
                session_start = time.perf_counter()
                session_started_at = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

    save_session_report(session_summary, session_start, session_started_at)


if __name__ == "__main__":
    main()




