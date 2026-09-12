# Release Checklist

## Prepared

- [x] YOLO main application in `src/yolo_tracker.py`.
- [x] Classical HSV baseline in `baselines/hsv_color_tracker.py`.
- [x] Existing algorithms and normal processing/drawing behavior preserved.
- [x] Small functions, grouped settings, import-safe entry points, and resource cleanup.
- [x] Runtime requirements based on the development environment.
- [x] README with controls, output semantics, units, baseline comparison, and known limitations.
- [x] Two-ID session JSON copied without changing measurements or model labels; video/report timing and temporary losses documented.
- [x] Restored demo video and a directly extracted screenshot included and linked from the README.
- [x] 25 deterministic tests pass without a webcam, model download, or real output files.
- [x] Ignore rules prepared for models, generated reports/videos, environments, and temporary files.

## Before public release

- [x] Owner approved AGPL-3.0 for the project's original source code; the full [LICENSE](../LICENSE) and README notice are included. Third-party dependencies and weights retain their own terms; see [Ultralytics licensing](https://www.ultralytics.com/license).
- [ ] Review the selected video and screenshot for consent, reflections, or private information before public upload.
- [ ] Test both applications manually with the webcam using the documented interpreter.
- [ ] Check `q`, `r`, the resulting video, and a YOLO report after the folder reorganization.
- [ ] Verify installation in a clean environment; the pinned requirements are based on installed package metadata, not a newly provisioned environment.
- [ ] Check that the selected OpenCV build supports the camera, GUI, and `mp4v` writer.
- [ ] Review the files staged for Git. Exclude model binaries, raw webcam videos, private session data, virtual environments, and personal IDE configuration.
- [ ] Choose the repository name/description/topics from [PORTFOLIO.md](PORTFOLIO.md).
- [x] Create a private GitHub repository for release review.
- [ ] Make the repository public only after reviewing the final contents and resolving the license.

## Scope of verification

Tests check application calculations, state transitions, output structure, and resource cleanup with test doubles. They do not establish detection accuracy, association quality, codec availability, or a guaranteed real-time frame rate.

A private comparison against the original scripts also checked that the existing drawing/recording and motion behavior was retained. Neither script is automatically run against your real webcam during repository preparation.

The GitHub repository was created as **private** for release review. Local model files, original recordings, and generated session reports remain unchanged. Only the selected example video and report belong in the repository; a private upload is not approval of the remaining public-release checklist items.
