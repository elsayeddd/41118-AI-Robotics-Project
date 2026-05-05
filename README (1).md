# Robot Arm Pick-and-Place — Engineering Project

A simulated Franka Panda robot arm that uses computer vision to identify
objects on a tabletop and pick them up. Built in PyBullet, runs in WSL.

**This README is for the whole team — read it first.**

---

## TL;DR

- **Project:** simulated robot arm doing vision-guided pick-and-place
- **Simulator:** PyBullet (runs in WSL, no ROS, no Gazebo needed)
- **Robot:** Franka Panda arm (on the instructor's HD-eligible robot list)
- **AI components:** computer vision (always-on) + reinforcement learning (stretch goal)
- **Target grade band:** Distinction, with HD as stretch
- **Key dates:** teaser video due **1 May**, in-class demo **12 May**, final report end of semester
- **Current status:** simulator + scene + overhead camera + colour-based object detection all working
- **Next steps:** grasp execution, evaluation harness, then record the teaser video

---

## What the project actually does

A Franka Panda robot arm sits at the edge of a virtual tabletop. Several
coloured cubes are placed at random positions on the table. An overhead
camera captures an image of the scene. A vision system processes that image
and locates the target cube. The arm then plans a path, moves to the cube,
closes its gripper, lifts it, carries it to a marked target zone, and
releases it. The system runs hundreds of trials with cubes in different
positions, recording how often the grasp succeeds, how accurately the cube
lands, and how long it takes.

That's the whole project. Everything else is detail layered on top.

### Why this is a real project, not a tutorial walkthrough

A tutorial would hardcode the cube position. The arm would be told
"go to coordinates (0.5, 0.0, 0.65) and close the gripper." That works
exactly once and proves nothing.

Our project is different because **the cube position is discovered by
vision, not hardcoded.** The arm has to look at the world, understand it,
and act on what it sees. The whole system is a closed loop: perception
feeds planning, planning feeds control, control changes the world, and
the next perception cycle measures the result.

The other thing that makes it a project rather than a tutorial is the
**evaluation**. We're not just trying to get a single successful grasp —
we run the system across many trials with varied conditions and measure
how reliably it works. That measurement is what the rubric calls "proper
evaluation using the correct metrics."

---

## Where the AI is in this

This is the most important section to read carefully. Markers will ask
"where's the AI?" and we need to be able to answer it precisely.

### AI Component 1: Computer vision for perception (always on)

The arm cannot act on what it cannot see. The vision component takes a
camera image and outputs the position of each object in the scene. This
is genuine machine perception applied to a robotics problem, and it's
one of the main course topics.

We're starting with **colour-based segmentation in HSV space using OpenCV**
as the V1 vision system. This is fast, easy to debug, and works reliably
when objects are colour-distinct. It's classical CV rather than deep
learning, but it's still a perception pipeline doing the work the rubric
asks for. For the final submission we'll likely upgrade to a fine-tuned
**YOLO model** trained on synthetic images from the simulator — that
turns the perception component into a learned ML model rather than a
hand-tuned threshold.

### AI Component 2: Reinforcement learning for control (stretch goal)

This is the optional second AI component that pushes us from Distinction
into HD territory. Instead of using classical inverse kinematics to
compute joint angles for grasping, we replace the controller with an
**RL policy** (PPO or SAC, from stable-baselines3) that has learned,
through thousands of simulated grasps, how to map "what I see" to
"motor commands." The arm effectively learns to grasp by trial and error
in simulation.

This is harder, takes hours to train, and can fail to converge — which
is why it's the stretch, not the baseline. The plan is to build the
classical version first as our guaranteed deliverable for the 12 May
demo, and architect the code so the RL policy drops in cleanly as a
replacement for the controller later.

### Mapping to course topics

The rubric requires combining at least **one** main course topic for
Distinction, **two** for HD. Our pairing:

- Perception: computer vision
- Control (stretch): reinforcement learning

CV alone gets us the Distinction floor. CV + RL gets us over the HD line.

---

## Why this scope was chosen

A few quick notes on why we picked this project over alternatives:

- **Robot arm is on the HD-approved list** alongside ground vehicles,
  drones, humanoids, and HRI setups. A pure software agent would have
  capped us at Distinction.
- **Manipulation splits cleanly into three roles** (perception, control,
  evaluation) which suits a 3-person group. Navigation tends to collapse
  into one person babysitting RL training while the others wait.
- **The simulator generates all our data.** No external dataset hunting,
  no labelling marathons. We render images on demand and have ground-truth
  positions for free.
- **Evaluation metrics are well-known and uncontroversial:** grasp
  success rate, placement error, time to completion, robustness across
  positions. The marker can't argue with the choice.
- **Staged risk:** the classical-control version works guaranteed; RL
  is upside-only. We can never end up demo-less on 12 May.

---

## Current status — what's built

Five files in the repo so far. Everything has been tested and works.

- `01_hello_panda.py` — sanity check, loads PyBullet and the Panda arm,
  prints joint info
- `02_tabletop_scene.py` — builds the project scene: arm + table +
  3 coloured cubes + target zone
- `03_camera_capture.py` — adds an overhead camera, captures
  `scene_view.png`, saves camera params for the next stage
- `04_detect_objects.py` — loads the camera image, finds the cubes via
  HSV thresholding, converts pixel positions to world coordinates
- `README.md` — this file

What this means concretely: we have a working simulator, a working
perception pipeline, and a working pixel-to-world coordinate transform.
The arm doesn't move yet, and the trials don't loop yet. Those are the
next two pieces.

---

## Setup — get this running on your machine

### Prerequisites

- WSL (Windows 11 with WSLg recommended; Windows 10 also works headless)
- About 2 GB free disk space
- 30–45 minutes of patience for the first install

### One-time setup

In WSL:

```bash
# Make sure system packages are current
sudo apt update
sudo apt install -y python3-venv python3-full python3-pip build-essential libgl1 libglib2.0-0

# Create the project folder and venv
mkdir -p ~/robot-project
cd ~/robot-project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install Python packages — pybullet compiles from source, takes 5–15 min
pip install pybullet numpy opencv-python pillow
```

The `pip install pybullet` line is the slow one. You'll see a wall of
C++ compiler output — that's normal, don't cancel it. When it finishes,
verify everything is there:

```bash
pip list | grep -E "pybullet|numpy|opencv|pillow"
```

You should see four lines.

### Get the project files

The five files in this repo need to live in `~/robot-project/`. Either
clone from wherever we're sharing this, or copy from your Windows
Downloads folder:

```bash
cp /mnt/c/Users/YOUR_WINDOWS_USERNAME/Downloads/*.py ~/robot-project/
cp /mnt/c/Users/YOUR_WINDOWS_USERNAME/Downloads/README.md ~/robot-project/
```

### Every time you come back to work

The venv doesn't persist across terminal sessions, so each new terminal:

```bash
cd ~/robot-project
source venv/bin/activate
```

You'll know it's active when your prompt shows `(venv)` at the start.

---

## Running it

Run the scripts in order. Each one builds on the previous.

```bash
python 01_hello_panda.py     # confirms PyBullet and the arm load
python 02_tabletop_scene.py  # the actual project scene
python 03_camera_capture.py  # generates scene_view.png
python 04_detect_objects.py  # uses scene_view.png, generates detection_result.png
```

After running 03 and 04, open these images to see what's happening:

- `scene_view.png` — what the overhead camera sees (top-down view of cubes)
- `detection_result.png` — same image with detected cubes circled and labelled

Open them from WSL with `explorer.exe scene_view.png`, or browse from
Windows to `\\wsl$\Ubuntu\home\YOUR_USER\robot-project\`.

### If the GUI doesn't work

If you're on Windows 10 or WSLg is misbehaving, the PyBullet GUI window
won't open. Each script has a `GUI = True` flag near the top — flip it
to `False` and the script will run headless and save a PNG instead of
opening a window. Functionality is identical.

---

## Group roles and where to start

We agreed (or are about to agree at the sync) on this split:

### Person 1 — Simulation and control
- Owns: scripts 01, 02, eventually `05_pick_and_place.py` (grasp execution)
- Start by: running all four existing scripts end-to-end and confirming
  they work on your machine. Take screenshots of the GUI for the teaser.
- Next task: write the inverse-kinematics grasp routine in
  `05_pick_and_place.py`.

### Person 2 — Computer vision
- Owns: scripts 03, 04, eventually a YOLO-based replacement
- Start by: running scripts 03 and 04, then experimenting with the HSV
  ranges in `04_detect_objects.py` to understand how thresholding works.
- Next task: validate the pixel-to-world conversion is accurate by
  comparing detected positions against the ground-truth positions printed
  by `03_camera_capture.py`. They should agree to within ~2 cm.

### Person 3 — Evaluation and integration
- Owns: the evaluation harness and the teaser video
- Start by: drafting the teaser video script.
- Next task: design the metrics logging. We need a CSV-style log that
  records, for every trial: trial number, target colour, ground-truth
  position, detected position, grasp success (yes/no), placement error
  in cm, time taken in seconds.

---

## The teaser video — due 1 May

3–5 minutes long. Five required topics, in order:

1. **Group composition** — who, what role
2. **What the project is about** — the pitch from the top of this README
3. **Data** — self-generated from the simulator, no external dataset
4. **Course components** — CV (always), RL (stretch); name them explicitly
5. **Validation strategy mapped to the marking rubric** — this is the
   one most students botch, so spend the most time on it

The key thing is to be specific about the grade band we're targeting and
to walk through the rubric criteria one by one, naming exactly how
this project clears each one.

### Footage we already have

- `scene_view.png` — top-down camera view of the table with cubes
- `detection_result.png` — same image with cubes detected and labelled
- Live PyBullet GUI of the arm and table (screen capture during demo)

The detection result image is the strongest piece of footage — it
visibly shows AI doing something: looking at an image and finding
things in it.

---

## What's coming next (after the teaser)

Roughly in this order:

1. **`05_pick_and_place.py`** — full pipeline: detect → move → grasp →
   place. Uses inverse kinematics, no learning yet. Targets the 12 May demo.
2. **Evaluation harness** — runs N trials, records metrics, outputs a
   results CSV and a summary.
3. **YOLO upgrade** — replaces colour thresholding with a fine-tuned
   YOLO model trained on synthetic images. Bumps the perception from
   classical CV to learned ML.
4. **RL stretch** — trains a PPO policy to replace the inverse-kinematics
   controller. Bumps us from one main topic to two, targeting HD.
5. **Final report and video** — packages everything for the end-of-semester
   submission.

---

## Q&A talking points (anticipate these from the marker)

The marker will probably push on three things in the 1 May Q&A. Be
ready for all of them.

**"Where's the AI?"**
In the perception pipeline — a vision model identifies and localises
objects from the camera feed — and, in our extended version, in the
control policy as well, where a reinforcement-learning agent maps
visual observations to motor commands for grasping.

**"Why simulation only and not real hardware?"**
The rubric specifies simulation, and sim lets us run thousands of
evaluation trials for proper metrics — which the rubric also requires.

**"What's your fallback if RL doesn't converge?"**
The classical-control version, which is the V1 milestone anyway. The
project does not depend on RL working — RL is the upgrade path from
Distinction to HD, not the foundation.

**"How is this beyond the tutorials?"**
PyBullet tutorials show the arm moving to a hard-coded pose; CV tutorials
show object detection on static images. Neither shows the closed-loop
integration — perception, planning, execution, evaluation across varied
scenes — which is what we're building. Plus the comparative evaluation
against a baseline.

---

## Common issues and fixes

**`pip install` fails with "externally-managed-environment"**
You forgot to activate the venv. Run `source venv/bin/activate` first.

**`apt install` shows lots of 404 errors**
Stale package list. Run `sudo apt update` and try again.

**PyBullet GUI window is black or doesn't open**
Either WSLg isn't installed (Windows 10) or it's flaky. Flip `GUI = True`
to `GUI = False` in the script, run again, and view the saved PNG.

**`libGL error: failed to load driver` when running PyBullet**
Run `sudo apt install -y libgl1 libglib2.0-0` to get the missing libs.

**Detection finds the wrong cubes or misses some**
HSV thresholds in `04_detect_objects.py` may need tuning. Open
`scene_view.png` in any image editor, hover over a cube, and read off
the HSV values. Adjust the ranges in the `COLOUR_RANGES` dict.

**Pixel-to-world conversion looks wrong**
The current conversion assumes the camera is straight overhead. If
anyone changes the camera angle, the conversion in `04_detect_objects.py`
needs to be updated to use proper camera intrinsics.

---

## Open questions to resolve at the sync

- Final group role allocation
- Confirm we're going Version A (scripted control) for 12 May demo
- Decide who narrates which section of the teaser video
- Decide recording day (aim for 30 April, leaves 1 May as buffer)
- Decide whether to add a fourth/fifth cube colour or stick with three
- Confirm where we're hosting the project repo (GitHub?)

---

That's the full picture. Run the scripts, eyeball the outputs, and
come to the sync ready to claim a role.
