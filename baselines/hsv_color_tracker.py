"""Track the largest green contour as a classical computer vision baseline.

Speed and distance are measured in pixels, not physical units.
Press Q to quit or R to reset the current motion measurements.
"""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# Paths are relative to the repository, regardless of the working directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
VIDEO_OUTPUT_PATH = PROJECT_DIR / "tracking_output.mp4"

# Camera and recording settings. Keep the camera's default resolution.
CAMERA_INDEX = 0
VIDEO_CODEC = "mp4v"
VIDEO_FPS = 30  # Fixed playback rate, not the measured processing FPS.

# Color segmentation and motion settings.
LOWER_GREEN = np.array([35, 100, 100])
UPPER_GREEN = np.array([85, 255, 255])
MIN_AREA = 500
MIN_STEP_DISTANCE = 2.0  # Only distance accumulation uses this pixel threshold.
TRAIL_LENGTH = 30
SPEED_WINDOW = 10
FPS_WINDOW = 30
DIRECTION_WINDOW = 6
MOTION_THRESHOLD = 10.0  # Pixels per second.
MAX_LOST_FRAMES = 10

# OpenCV colors use BGR order.
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)


def draw_text(frame, text, position, color=GREEN):
    """Draw an overlay label with the shared text style."""
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
        0.7, color, 2,
    )


def main():
    """Run color segmentation, motion analysis, drawing, and recording."""
    trail = deque(maxlen=TRAIL_LENGTH)
    fps_history = deque(maxlen=FPS_WINDOW)
    speed_history = deque(maxlen=SPEED_WINDOW)
    total_distance = 0.0
    average_fps = 0.0
    average_speed = 0.0
    lost_count = 0
    has_tracked = False
    last_detection_time = None

    # Preserve the original first FPS interval, including camera setup time.
    previous_frame_time = time.perf_counter()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    writer = None

    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        writer = cv2.VideoWriter(
            str(VIDEO_OUTPUT_PATH), fourcc, VIDEO_FPS,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not open video writer")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()
            elapsed = now - previous_frame_time
            previous_frame_time = now
            if elapsed > 0:
                fps = 1 / elapsed
                fps_history.append(fps)
                average_fps = sum(fps_history) / len(fps_history)

            frame = cv2.flip(frame, 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )

            output_frame = frame.copy()
            object_detected = False

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                if cv2.contourArea(largest_contour) > MIN_AREA:
                    object_detected = True
                    has_tracked = True
                    lost_count = 0
                    draw_text(output_frame, "TRACKING", (50, 150))

                    x, y, width, height = cv2.boundingRect(largest_contour)
                    center_x = x + width // 2
                    center_y = y + height // 2
                    trail.append((center_x, center_y))

                    if len(trail) >= 2 and last_detection_time is not None:
                        previous_center = trail[-2]
                        current_center = trail[-1]
                        dx = current_center[0] - previous_center[0]
                        dy = current_center[1] - previous_center[1]
                        step_distance = np.hypot(dx, dy)
                        detection_elapsed = now - last_detection_time

                        if detection_elapsed > 0:
                            speed = step_distance / detection_elapsed
                            if step_distance >= MIN_STEP_DISTANCE:
                                total_distance += step_distance
                            speed_history.append(speed)
                            average_speed = sum(speed_history) / len(speed_history)

                    last_detection_time = now
                    draw_text(
                        output_frame, f"Total Distance: {total_distance:.1f}", (200, 70),
                    )
                    draw_text(
                        output_frame, f"Speed: {average_speed:.1f} px/s", (20, 40),
                    )

                    direction = "Waiting"
                    if len(trail) >= DIRECTION_WINDOW:
                        old_center = trail[-DIRECTION_WINDOW]
                        direction_dx = center_x - old_center[0]
                        direction_dy = center_y - old_center[1]

                        if average_speed < MOTION_THRESHOLD:
                            direction = "Still"
                        elif abs(direction_dx) > abs(direction_dy):
                            direction = "Right" if direction_dx > 0 else "Left"
                        else:
                            direction = "Down" if direction_dy > 0 else "Up"

                        cv2.arrowedLine(
                            output_frame, old_center, (center_x, center_y),
                            BLUE, 4, tipLength=0.4,
                        )

                    draw_text(output_frame, f"Direction: {direction}", (20, 70))

                    for index in range(1, len(trail)):
                        thickness = int(index / len(trail) * 5) + 1
                        cv2.line(
                            output_frame, trail[index - 1], trail[index],
                            GREEN, thickness,
                        )

                    cv2.rectangle(
                        output_frame, (x, y), (x + width, y + height), GREEN, 2,
                    )
                    cv2.circle(output_frame, (center_x, center_y), 5, RED, -1)

            if not object_detected:
                if has_tracked:
                    lost_count += 1
                    draw_text(output_frame, "LOST", (50, 150), RED)
                else:
                    draw_text(output_frame, "SEARCHING", (50, 150), YELLOW)

                # Short gaps retain the original baseline's movement history.
                if lost_count >= MAX_LOST_FRAMES:
                    trail.clear()
                    has_tracked = False
                    last_detection_time = None
                    speed_history.clear()
                    average_speed = 0.0
                    lost_count = 0

            draw_text(output_frame, f"FPS: {average_fps:.1f}", (20, 100))
            writer.write(output_frame)
            cv2.imshow("Object Tracking", output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                trail.clear()
                total_distance = 0.0
                lost_count = 0
                has_tracked = False
                last_detection_time = None
                speed_history.clear()
                average_speed = 0.0

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
