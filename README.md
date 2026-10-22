# BlazeHand

**A webcam hand-gesture controller for Pokémon Blaze Online.**

BlazeHand lets you play a keyboard-driven browser game using only hand gestures in front of
a webcam. It tracks your hand with MediaPipe, recognises gestures with a combination of
rule-based 3D-geometry logic and a trained image classifier (MobileNetV3-Small), and sends
the resulting keyboard/mouse input to the game. Everything runs locally on the PC; no video
or data leaves the machine.

Built as a course project and demonstrated on **Pokémon Blaze Online**
(https://play.pokemonblazeonline.com/), a keyboard-controlled browser MMO.

---

## 1. The project in brief

Playing a game normally means both hands are locked to a keyboard and mouse. BlazeHand
replaces that with a camera:

1. The webcam sees your hand(s).
2. MediaPipe extracts 21 hand landmarks per hand, 30 times per second.
3. A recognition layer turns hand shapes into game actions:
   - **Finger counting** (0–4 raised fingers) walks the character — fist = stop.
   - **Pinch** (thumb touches index) presses the confirm/interact key, and clicks the
     mouse when aiming in battle.
   - **Pointing** (index up) moves the cursor in battle menus when used with the trained
     classifier in battle-cursor mode.
4. A gesture overlay floats above the game with a live camera view, landmarks, available
   gestures, context badge (overworld/battle) and lock state.
5. The game keeps keyboard focus; BlazeHand only *sends* keys/clicks to it.

The result: walking, opening menus, choosing battle moves and selecting targets can be done
with gestures, without touching the keyboard.

---

## 2. Target game

| Item | Value |
|---|---|
| Game | Pokémon Blaze Online (fan-made browser MMO) |
| Link | https://play.pokemonblazeonline.com/ |
| Input it expects | Arrow keys / WASD to walk, `1`–`4` battle moves, `Space` confirm/interact, `x` cancel/back |
| How BlazeHand talks to it | OS-level key events + mouse (pynput), exactly like a keyboard |

The game runs in the browser in the background; the BlazeHand overlay is pinned on top.
Click the game canvas once so it owns keyboard focus — the overlay never steals input.

---

## 3. Features

- **Finger-count movement** — 1/2/3/4 fingers walk up/down/left/right, fist stops.
  Robust 4-cue voting per finger (joint angles + straightness + radial distance) with a
  temporal filter, so single noisy frames never change direction.
- **Pinch-to-confirm** — thumb-index pinch with hysteresis (`0.25` on / `0.40` off) and a
  2-frame confirm; fires `Space` (or a real mouse click in battle-cursor mode).
- **Automatic battle detection** — periodically scores a screenshot of the game window
  (bottom 45 % where battle menus live) and switches between *overworld* and *battle*
  input mappings. Manual override with `b`.
- **Battle-cursor mode** — the index fingertip drives the OS cursor with a full smoothing
  stack (1€ filter, motion predictor, dead-zone, jump guard). Pinch selects the move.
- **Battle auto-lock** — hold the hand still ~0.7 s and the cursor freezes in place so
  tremor can't move it off the button; pinch still clicks at the frozen spot. Move the hand
  away (0.06 of frame width) or wave ~2 s to release. Finger shaping never unlocks it.
- **Two-hand `1+1` toggle** — show `1` with both hands for ~0.27 s to flip between forced
  overworld and forced battle-cursor (handy when auto-detection is unsure).
- **Trained gesture model (AI)** — MobileNetV3-Small fine-tuned on HaGRID answers "is the
  hand pointing up?" for the battle aim/lock gate; falls back to pure geometry if PyTorch
  or the model is unavailable.
- **Failsafes** — all keys are released on hand loss, context switch, quit, or exception;
  key output can be toggled off (`k`) for safe testing; `q`/`Esc` always exits.
- **Privacy** — camera frames are processed locally and never uploaded.
- **71 hardware-free unit tests** covering geometry, filters, locks, context smoothing,
  duo toggle and the key controller (the suite never opens a camera or screen).

---

## 4. Gestures and controls

### 4.1 Overworld — movement (default `fingers` mode)

| Gesture | Action |
|---|---|
| Fist | STOP |
| 1 finger (index) | Walk up / forward |
| 2 fingers | Walk down / back |
| 3 fingers | Walk left |
| 4 fingers | Walk right |
| Pinch | `Space` — interact / confirm / talk |

Finger counts must hold for a few frames before changing direction (`flip_frames`,
`hold_frames`); brief tracking dropouts keep the character walking instead of stuttering.
A legacy `joystick` movement mode (palm position as a virtual stick) is also available via
`game.movement.mode`.

### 4.2 Battle — two interaction styles (`game.battle.interact`)

**Cursor mode (default)** — aim with your hand:

| Gesture | Action |
|---|---|
| Index pointing up | Moves the cursor (aim the move button) |
| Pinch | Left-click on the current cursor position (select) |
| Fist / hand not aiming | Cursor parks / coasts |
| Hand still ~0.7 s | Auto-lock (cursor freezes; pinch still clicks) |
| Move hand > 0.06 or wave ~2 s | Unlock |
| `1+1` with both hands | Toggle back to forced movement |

**Taps mode** — no cursor, direct move selection:

| Gesture | Action |
|---|---|
| 1 / 2 / 3 / 4 fingers | Tap keys `1` / `2` / `3` / `4` |
| Fist | `x` — cancel / back |
| Pinch | `Space` — confirm |

### 4.3 Duo toggle (both hands)

Show `1` on each hand simultaneously for ~0.27 s: switches between forced **overworld** and
forced **battle** mode (same gesture switches back). `strict_handedness: false` accepts any
two distinct `1` hands; `v` does the same from the keyboard.

### 4.4 Overlay keyboard shortcuts

| Key | Action |
|---|---|
| `q` / `Esc` | Quit (releases all keys) |
| `k` | Toggle keyboard/mouse output on/off |
| `h` | Toggle the help legend |
| `b` | Cycle context: auto → force battle → force overworld |
| `v` | Manual duo toggle (same as `1+1` both hands) |
| `c` | Recenter virtual joystick at current palm position (joystick mode only) |

---

## 5. How it works — core mechanism

```
webcam (OpenCV, 1280x720 @30 fps, buffer=1)
   │
   ▼
MediaPipe HandLandmarker (21 landmarks per hand, up to 2 hands)
   │
   ├──► rule-based gesture layer
   │      • per-finger 4-cue vote → FingerStateFilter → finger count
   │      • pinch ratio (tip distance / hand size) + hysteresis → click
   │      • fist / palm / run debouncers (joystick mode)
   │
   ├──► trained classifier (optional, ~12 ms/frame on GPU)
   │      • MobileNetV3-Small, full-frame, 19 classes → BlazeHand pose
   │      • battle aim/lock gate ("is the hand pointing?")
   │
   ▼
context layer: screenshot probe (~2 Hz) → battle score → ContextSmoother
   │            (3 frames to enter battle, 5 to leave; `b` overrides)
   ▼
action routing
   ├── overworld: held arrow/WASD keys from finger count
   ├── battle taps: single key taps from stable poses (1.2 s repeat guard)
   └── battle cursor: index tip → OS cursor
         OneEuro filter → predictor → dead-zone → jump guard
         pinch → mouse click; StillLock freezes/parks the cursor
   │
   ▼
OS input (pynput) → browser game (Pokémon Blaze Online)
```

**Cursor smoothing chain** (`pointer.py`, `filters.py`): hand position is mapped from a
configurable motion area of the camera image to the screen, mirrored so motion feels
natural, then filtered with a **1€ filter** (Casiez 2012 — jitter-free at rest, responsive
in motion), followed by a small constant-velocity predictor, a dead-zone and a jump guard
that rejects physically impossible cursor jumps.

**Auto-lock geometry**: stillness and displacement are measured on the hand's rigid frame
— **wrist + middle knuckle** — not on the fingertips. Pinch shaping moves the index/thumb a
lot; measuring tips made the lock release exactly when the user was about to click. Frames
that are pinching or not aiming *pause* the lock clock instead of voting.

**Battle detection** (`game_context.py`): the bottom 45 % of the game window is scored with
colour/template heuristics at ~2 Hz, debounced with asymmetric hysteresis (enter 3 frames,
exit 5). Optional user templates live in `assets/battle_templates/`. Failures degrade to
overworld, and `b`/`v` give manual control.

---

## 6. The trained model

| Item | Value |
|---|---|
| Architecture | MobileNetV3-Small (ImageNet-pretrained, ~2.5 M params) |
| Input | Full camera frame, 224×224 center crop (HaGRID full-frame distribution) |
| Classes | 19 HaGRID classes (`fist`, `one`, `peace`, `three`, `four`, `palm`, `stop`, … `no_gesture`) |
| Output mapping | `labels.json` → BlazeHand poses (`one`→`index_only`, `four`→`four`, …) |
| Checkpoint | `models/blazehand-mini/best.pt` (~6 MB) |
| Inference | PyTorch, CUDA — ~12 ms/frame on an RTX 4050 Laptop; CPU fallback |
| Where it is used | Battle-cursor aim gate + auto-lock gating (README §5) |
| Fallback | If PyTorch/model is missing, the game runs fully rule-based |

**Dataset:** [HaGRID](https://github.com/hukenovs/hagrid) (Kapitanov et al., WACV 2024) —
hand-gesture images photographed by thousands of people in real rooms. The project uses the
classification mirror
[`cj-mills/hagrid-classification-512p-no-gesture-150k-zip`](https://huggingface.co/datasets/cj-mills/hagrid-classification-512p-no-gesture-150k-zip):
**153,735 images, 19 classes**, license **CC BY-SA 4.0**.

For course scope, training uses a stratified subset: `data/hagrid-mini/` —
**2,850 images, 150 per class** (`training/make_subset.py`), 40 epochs with strong
augmentation (~7 minutes on the RTX 4050).

**Result on the held-out 10 % split — accuracy 0.947, macro-F1 0.948:**

| BlazeHand pose | HaGRID class(es) | Val F1 |
|---|---|---|
| `fist` (STOP) | `fist` | 0.97 |
| `index_only` (UP / aim) | `one` | 0.84 |
| `peace` (DOWN) | `peace`, `two_up` | 0.80–0.90 |
| `three` (LEFT) | `three`, `three2` | 0.97 |
| `four` (RIGHT) | `four`, `palm`, `stop` | 0.97–1.00 |
| none (no action) | `no_gesture` + others | 0.88–1.00 |

Known weak spots: `one` vs `peace` confuse each other and their inverted variants most —
1-vs-2 raised fingers is the hardest fine-grained split at 150 samples/class. More samples
of those classes, or merging `peace`/`two_up`, is the obvious next improvement.

Artifacts in `models/blazehand-mini/`: `best.pt`, `last.pt` (resume state), `labels.json`
(classes + mapping + license note), `metrics.txt`, `confusion_matrix.npz`,
`training_log.csv`. ONNX export is implemented (`--export-onnx`) but blocked on Windows
machines with Application Control enabled (see §10).

---

## 7. Installation and running

Requirements: Windows 10/11 (tested on Windows 11 + Python 3.14), a webcam, Python 3.10+.

```powershell
# 1) environment
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# 2) optional: trained-model runtime (battle aim/lock). The game works without it.
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3) run
.venv/Scripts/python src/main.py --mode game
```

The MediaPipe `hand_landmarker.task` model is downloaded automatically on first run
(`src/model_utils.py`). Allow Python access to the camera in Windows privacy settings.

### Modes (`src/main.py --mode …`)

| Mode | What it does |
|---|---|
| `tracker` | Camera + landmarks only — use it to check tracking/lighting |
| `pointer` | Index fingertip moves the OS cursor |
| `full` | Pointer + pinch = mouse click (default) |
| `game` | **The BlazeHand controller**: movement, battle taps, battle cursor, auto-lock, duo toggle |

Keys for `pointer`/`full`: `q`/`Esc` quit, `p` toggles mouse output.

### Demo checklist

1. Start the game, click the game canvas once (keyboard focus).
2. Start BlazeHand with `--mode game`; the overlay appears topmost.
3. Walk with 1/2/3/4 fingers; fist stops; pinch interacts.
4. In battle, point at the move and pinch — or hold still for auto-lock, then pinch.
5. If the context badge is wrong, press `b` or use the two-hand `1+1` toggle.

---

## 8. Configuration

All tunables live in `config/settings.yaml` (comments explain units/defaults). Highlights:

| Section | What it controls |
|---|---|
| `camera` | Device index, resolution, fps, exposure/focus |
| `tracker` | MediaPipe confidences, max hands, hand preference |
| `pointer` | 1€ filter, predictor, dead-zone, jump guard, motion area, mirroring |
| `click` | Pinch on/off ratios, confirm frames, click cooldown |
| `game.movement` | `fingers` (count walking) or legacy `joystick` |
| `game.fingers` | Flip/hold/stop frame counts and geometry thresholds |
| `game.actions` | Pinch/fist/palm/run keys and hold times (joystick extras) |
| `game.context` | Probe rate, threshold, enter/exit frames, template dir, debug logs |
| `game.battle` | `interact: cursor\|taps`, move keys, tap timing |
| `game.battle.cursor.lock` | Auto-lock timing: `still_frames` 20 (~0.7 s), unlock displacement/meter |
| `game.battle.cursor.classifier` | Trained model path, confidence threshold, aim poses |
| `game.duo` | Two-hand toggle timing, handedness strictness, cursor hand preference |

Every setting is optional; missing keys fall back to safe defaults in code.

---

## 9. Project structure

```
config/settings.yaml     all tunables
models/
  hand_landmarker.task   MediaPipe hand model (auto-downloaded)
  blazehand-mini/        trained classifier + metrics + labels
src/
  main.py                entry point (mode selection, pointer/full loops)
  camera.py              webcam capture + warmup
  tracker.py             MediaPipe hand tracking (single/two hands)
  landmarks.py           21-point geometry helpers (angles, distances, pinch)
  gestures_game.py       finger counting, voting, filters, joystick, duo toggle
  gestures_click.py      pinch-to-click detector with hysteresis
  click_lock.py          ClickLock (pointer mode) + StillLock (battle auto-lock)
  pointer.py             hand → screen mapping + index gate
  filters.py             1€/EMA smoothing, predictor, dead-zone, jump guard, cooldown
  game_mode.py           the game loop: context, modes, keys, cursor, HUD
  game_context.py        battle-screen detection + battle pose mapping
  game_keys.py           keyboard/mouse backend (pynput) + safe release
  game_overlay.py        game HUD, d-pad, battle menu, cursor, duo meter, help
  gesture_classifier.py  trained-model wrapper (lazy PyTorch import)
  overlay.py / actions.py / model_utils.py / config_loader.py
training/
  train_gesture.py       training loop (AMP, OneCycleLR, resume, ONNX export)
  make_subset.py         stratified subset builder
  check_classifier.py    sanity check: best.pt through the game wrapper
  extract_dataset.py / resize_dataset.py
tests/                   hardware-free unit tests (71)
data/                    datasets (git-ignored, local only)
```

---

## 10. Testing and troubleshooting

```powershell
.venv/Scripts/python -m pytest tests/ -q      # 71 passed
```

The suite covers landmarks/geometry, filters, pinch detector, pointer gates, ClickLock,
StillLock (lock/unlock/pause/cooldown), finger states, joystick, key controller, battle
taps, context smoothing and the duo toggle — with synthetic hands, no hardware.

| Symptom | Fix |
|---|---|
| `WinError 4551 — Application Control policy has blocked this file` | Windows 11 **Smart App Control** blocks unsigned native DLLs (MediaPipe/ONNX). Disable it: Windows Security → App & browser control → Smart App Control → Off (one-way). Required for hand tracking on such machines. |
| Camera black / not opening | Check Windows camera privacy setting; try another `camera.index`; keep `auto_exposure: true` |
| Cursor jitters at rest | Raise `pointer.one_euro.min_cutoff`; improve lighting (bare hand visible) |
| Pinch fires too rarely / too often | Retune `click.pinch_on` / `pinch_off` (keep a real gap = hysteresis) |
| Character stutters between counts | Raise `game.fingers.flip_frames` / `hold_frames` |
| Battle not detected | Press `b`, add template PNGs, set `game.context.debug: true` and read the scores |
| Cursor locks too eagerly / won't release | `lock.still_frames` up / `unlock_displacement` up; check the hand is fully visible |
| Clicks land slightly off after aiming | Wait for `LOCKED` (0.7 s) before pinching — clicks land at the frozen spot |
| Keys go to the wrong window | Click the game canvas once; the overlay must not have focus |
| `No module named mediapipe/pytest` | Use `.venv/Scripts/python`, not the system interpreter |

---

## 11. Limitations and future work

- Rule-based recognition is geometric; heavy background clutter or poor lighting can still
  confuse MediaPipe's detector. The trained classifier helps in battle but is not yet used
  for movement.
- The classifier was trained on full HaGRID frames, not a webcam; per-class F1 for `one`
  (0.84) and `peace` (0.80) leaves room for more data or a crop-based model.
- Battle detection is a screen heuristic; game UI changes can break it (`b`/duo toggle are
  the fallbacks, templates can be added).
- Not all game actions are mapped yet (e.g. menus, inventory) — the action layer is
  table-driven and easy to extend.
- ONNX export blocked by Application Control on the dev machine; PyTorch checkpoint works
  everywhere. A CPU-only ONNX build would simplify deployment.

---

## 12. Credits and license

- **HaGRID** — A. Kapitanov, A. Makhlyarchuk, K. Kvanchiani, *HaGRID — HAnd Gesture
  Recognition Image Dataset*, WACV 2024. Dataset mirror by cj-mills (Hugging Face).
  **License: CC BY-SA 4.0** — credit HaGRID if you publish the model or results.
- **MediaPipe** (Google) for hand landmark detection; **OpenCV** for camera/UI;
  **pynput** for OS input; **PyTorch/torchvision** for the classifier.
- **Pokémon Blaze Online** is a third-party fan game; this project is unaffiliated and only
  sends normal keyboard/mouse input to it. Check the game's rules before automating play.
