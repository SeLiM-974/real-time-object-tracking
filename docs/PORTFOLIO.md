# GitHub & CV Presentation

This file contains copy-ready portfolio text. It does not claim benchmark scores, calibrated physical speed, custom training, or a new tracking algorithm.

## Repository names

Recommended: **object-tracking-motion-analytics**

Alternatives:

- `yolo-motion-analytics`
- `real-time-tracking-analytics`

The existing local directory name, `real-time-object-tracking`, is also suitable. Renaming the GitHub repository does not require renaming the application scripts.

## GitHub description

Webcam-based YOLO multi-object tracking with per-track motion analytics, annotated video export, JSON session reports, and a classical HSV/OpenCV baseline.

## README title

**Object Tracking & Motion Analytics**

## CV bullets — English

- Developed a Python/OpenCV application using pretrained YOLOv8 for multi-object tracking, with per-ID trajectories, direction estimation, smoothed pixel-speed measurements, and distance analytics.
- Implemented track lifecycle management and observation-gap handling, preserving per-track session summaries in JSON and exporting annotated video.
- Retained an HSV segmentation and contour-based classical CV baseline, and added deterministic tests covering motion calculations, resets, timeouts, and resource cleanup.

## CV maddeleri — Türkçe

- Python/OpenCV ve önceden eğitilmiş YOLOv8 kullanarak ID bazlı hareket izi, yön, yumuşatılmış piksel hızı ve mesafe analizi sunan çoklu nesne takip uygulaması geliştirdim.
- Takip yaşam döngüsü ve eksik gözlem aralıklarını yöneten mantığı kurarak nesne bazlı oturum özetlerini JSON'a, işlenmiş görüntüyü video dosyasına aktardım.
- HSV segmentasyonu ve contour tabanlı klasik CV sürümünü baseline olarak korudum; hareket hesapları, sıfırlama, timeout ve kaynak kapatma davranışlarını otomatik testlerle doğruladım.

## Suggested GitHub topics

`python` `opencv` `yolo` `yolov8` `computer-vision` `object-tracking` `multi-object-tracking` `motion-analysis` `video-analytics` `numpy` `hsv`

## What to explain in an interview

- YOLO supplies detections and tracking IDs; the project's contribution is the per-ID analytics, lifecycle logic, recording, and reporting.
- Displayed speed is a recent moving average; report average speed is accepted distance divided by accepted observation time.
- A missing observation breaks the YOLO motion history so a reacquired object does not create an unobserved distance jump.
- Pixel speed is image-plane motion, not physical velocity.
- The baseline selects one large green contour and retains its original limitations. It is not a multi-ID alternative to YOLO.
- Tests use controlled input and mocks, not real-world accuracy or performance measurements.

Only claim techniques and code you can explain. Do not add numerical accuracy/FPS improvements to the CV without a defined experiment and recorded evidence.
