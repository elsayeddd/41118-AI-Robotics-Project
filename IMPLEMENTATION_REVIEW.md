# Implementation Review and Roadmap (Aligned to README Goals)

## Project target from README
The baseline target is a **vision-guided pick-and-place pipeline** with measurable reliability across many randomized trials, then optional RL controller as a stretch goal.

## Cross-file issues found

1. **Scene duplication risk between `02_tabletop_scene.py` and `03_camera_capture.py`.**
2. **Hidden camera-orientation coupling between `03_camera_capture.py` and `04_detect_objects.py`.**
3. **No shared constants for task-critical geometry.**
4. **No explicit staged roadmap in-repo for moving from CV-only to full demo + evaluation harness.**

## Modifications applied in this change set
- Added shared scene module (`common_scene.py`)
- Refactored scene + camera scripts to use shared setup
- Persisted and validated camera orientation contract (`cam_up`)
- Added implementation roadmap for next phases

## Recommended next implementation direction
1. Build `05_pick_and_place.py` baseline controller
2. Create `06_eval_trials.py` evaluation harness
3. Version task configs in one file
4. Add acceptance criteria aligned with demo goals
5. Add RL controller plug-in interface as stretch
