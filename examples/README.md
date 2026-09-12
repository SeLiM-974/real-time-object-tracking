# Demo assets

This directory contains the selected documentation examples, not automatic webcam output.

- [demo.mp4](demo.mp4): an unchanged copy of the restored recording supplied as `yolo_tracking_output` (without an extension) in the project root. The source decodes as 123 frames at 1280 × 720 and 30 FPS: 4.10 seconds of playback. No frames were removed, retimed, or relabeled for this copy.
- [screenshot.png](screenshot.png): frame 97 (zero-based), approximately 3.23 seconds into the encoded video. It is a full-resolution frame extracted directly from that video, with no crops, retouching, or added overlays.
- [session_example.json](session_example.json): the full report from `session_20260911_194803.json`, preserving all fields, measurements, and predicted labels. It describes a 16.32-second measurement session and two tracking IDs.

## Reading the example

The scene contains a phone and a drinking glass against a plain background. The phone is ID 1 and the glass is ID 2 in the reviewed sequence. The final visible distance totals are 904.4 px and 866.5 px, respectively, consistent with the report's rounded values. The earlier screenshot intentionally shows intermediate measurements, not the final report totals.

The report stores each ID's first class prediction. The glass is initially classified as `vase` and later appears as `cup` in the overlays; labels can change without a new ID. No predicted class has been corrected or removed for presentation.

The report records 5 temporary losses for ID 1 and 7 for ID 2. These count returns after missing processed frames, not ID switches. This is a compact demonstration, not a ground-truth-annotated evaluation or proof of uninterrupted tracking.

## Timing

The video writer assigns 30 FPS to the frames it receives. The measurement report uses elapsed capture-loop timestamps. Because the actual processing loop is slower and includes initial warm-up, the 4.10-second encoded video plays faster than the 16.32-second session. Its playback timestamps must not be used as the clock for the displayed px/s measurements.

The application code and the source recording have not been changed to reconcile these durations. The report has not been shortened, rescaled, or recomputed from the video.

## Before publication

Review the complete recording, including any reflections or private information, before pushing it to GitHub. Only these deliberately selected assets belong in the public examples directory. The original recording and generated session report remain unchanged in the project root and are excluded from Git; the extensionless restored source is explicitly ignored as well.

The MP4 uses the application's original codec. If an in-browser player cannot decode it, download it and open it in a local video player.
