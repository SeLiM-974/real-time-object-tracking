# Model weights

The main application expects the original pretrained **YOLOv8n detection model** at `models/yolov8n.pt`.

Model binaries are intentionally ignored by Git. Keep your existing file locally; no training or ONNX export is needed. The HSV baseline needs no weights.

From the repository root, after installing requirements:

~~~bash
python -c "from pathlib import Path; from ultralytics import YOLO; YOLO(str(Path('models/yolov8n.pt').resolve()))"
~~~

The existing Ultralytics loader fetches the official pretrained asset if that absolute path is missing. An internet connection is needed for a missing model; the command does not open the camera. Alternatively, obtain the file through the [official YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/) / [Ultralytics assets](https://github.com/ultralytics/assets/releases) and place it here.

Use trusted official weights. This project does not need any of the other model variants or an `.onnx` file.

Weights have their own licensing conditions; see [Ultralytics licensing](https://www.ultralytics.com/license). Excluding the binary from Git is repository hygiene, not a license exemption.
