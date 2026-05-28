# AI Direction: Clutter-Aware Cube Sorting Robot

## Project Story

The robot is not just moving coloured blocks. It is solving a cluttered tabletop
sorting problem: find cube-shaped target objects among distractors, ignore
non-target clutter, choose a pick order, and move every cube into the sorting
zone.

## How The Feedback Is Addressed

- Randomised training: `06_generate_yolo_dataset.py` randomises cube count,
  distractor count, cube colours, camera pose, field of view, image noise, and
  robot-arm occlusion.
- Stronger story: the task is framed as clutter-aware cube sorting.
- Robot blocking the view: the runtime parks the arm before detection, and the
  YOLO dataset includes arm occlusion so the detector learns partial views.
- YOLO from pretrained model: `07_train_yolo.py` loads pretrained YOLO weights
  by default and fine-tunes them on the synthetic cube dataset.
- No trained model yet: the new pipeline creates labels automatically from
  PyBullet segmentation and trains a model into `models/yolo_cube.pt`.
- Deep learning focus: perception uses YOLO, sorting decisions use PPO RL, and
  TensorBoard logs both model training and reward improvement.
- Colour detection is too simple: YOLO is trained to detect the object class
  `cube` regardless of colour, with cluttered distractor objects in the scene.
- Camera calibration is not the core contribution: projection is only used to
  convert YOLO detections into approximate pick positions. The AI contribution
  is detection, curriculum/domain randomisation, and RL sorting policy.

## Main Pipeline

1. Generate cluttered synthetic YOLO data:

   ```bash
   python 06_generate_yolo_dataset.py --samples 1200 --val-samples 240
   ```

2. Fine-tune pretrained YOLO:

   ```bash
   python 07_train_yolo.py
   ```

3. Train the RL sorting policy:

   ```bash
   python 08_train_rl_sorting_policy.py
   tensorboard --logdir runs
   ```

4. Run the cluttered sorting demo:

   ```bash
   python 05_pick_and_place.py
   ```

5. Evaluate robustness across random cluttered scenes:

   ```bash
   python 09_evaluate_system.py --scenes 50 --save-images
   ```

## What To Show In The Report

- YOLO examples: generated cluttered images, labels, detections, precision/recall
  curves, and validation predictions.
- RL learning: TensorBoard reward curve improving across curriculum stages.
- Robotics demo: before/after screenshots showing cubes moved to the target zone
  while distractor objects remain on the table.
- Quantitative evaluation: `evaluation/latest/summary.json` and
  `evaluation/latest/metrics.csv` reporting cube recall, false positives,
  scene pass rate, and localisation error across random cluttered scenes.
- Ablation: compare easy/no-clutter training to randomised clutter training and
  show why randomisation prevents overfitting.
