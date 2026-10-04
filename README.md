# BlazeHand

**Play Pokémon Blaze Online with your hands.** BlazeHand turns a commodity webcam into a
touch-free game controller: walk the overworld with finger-count gestures, flip into battle
mode with a two-hand `1+1` hold, aim with your index finger, and click by pinching — with an
auto-locking battle cursor that parks itself when your hand rests.

- Target game: [Pokémon Blaze Online (browser)](https://play.pokemonblazeonline.com/)
- Stack: Python, OpenCV, MediaPipe Hand Landmarker, pynput / pyautogui for OS input
- No gloves, no extra hardware — bare hands in decent light, one webcam

> Folder name on disk is still `custom-hand-gesture-controlled-system` (history); the
> project itself is **BlazeHand**. Only docs carry the new name, nothing else moved.

---

## 1. Quickstart

```powershell
# from the repo root
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python src/main.py --mode game
```

- The hand model (`models/hand_landmarker.task`) downloads itself on first run.
- Camera defaults: index `0`, `1280x720` @ 30 fps (see `config/settings.yaml` → `camera`).
- **Do NOT install tensorflow.** `src/` never imports it, and on Python 3.14 / locked-down
  Windows it breaks MediaPipe (DLL blocked). MediaPipe works fine without it.

Run from the **repo root** (code inserts `src/` on `sys.path` itself):

| Command | What you get |
|---|---|
| `.venv/Scripts/python src/main.py` | Full pointer: move mouse + pinch-click (`full`, the default) |
| `.venv/Scripts/python src/main.py --mode tracker` | Camera + hand landmarks only, no input |
| `.venv/Scripts/python src/main.py --mode pointer` | Move mouse, no clicking |
| `.venv/Scripts/python src/main.py --mode game` | **BlazeHand game controller** (this README's focus) |
| `.venv/Scripts/python -m pytest tests/ -q` | Hardware-free test suite (66 tests) |

Game-mode setup for PBO:

1. Open <https://play.pokemonblazeonline.com> in your browser, log in.
2. Click the game canvas **once** so it has keyboard focus.
3. Start `--mode game`. A small camera window floats **always-on-top**.
4. Click **back on the browser game** — keys/clicks are sent OS-wide, the overlay only displays.
5. To quit, focus the overlay window and press `q` (or `Esc`).

---

## 2. How BlazeHand plays the game

Two contexts, one controller. The overlay badge always shows which is active
(green `OVERWORLD`, red `BATTLE`), and every transition releases held walk keys so you
never walk into (or out of) a fight.

### 2.1 Overworld — finger-count walking (default)

Hold up a finger count with **one hand**. The pose drives held keys (arrows + WASD
together, so a lost-focus arrow scroll never leaves you standing still):

| Fingers | Pose name | Action |
|---|---|---|
| ✊ fist (0) | `fist` | **STOP** — releases all direction keys (fast, ~2 frames) |
| ☝️ 1 finger (index) | `index_only` | Walk **forward / UP** |
| ✌️ 2 fingers (index+middle) | `peace` | Walk **backward / DOWN** |
| 3 fingers (+ring) | `three` | Walk **LEFT** |
| 🖐️ 4 fingers (thumb ignored) | `open_palm` | Walk **RIGHT** |
| 🤏 thumb–index pinch | pinch channel | **Confirm / interact** (`Space`) |

Recognition is layered so borderline frames don't moonwalk your character:

1. **4-cue voting per finger** (`gestures_game.py`): PIP angle + DIP angle + radial order +
   segment straightness; 3 of 4 must agree (one bad cue — foreshortening, side view,
   jitter — is tolerated). Ring/pinky get looser thresholds.
2. **`FingerStateFilter`**: the whole 4-bit pattern must repeat N frames (`flip_frames: 3`)
   before it commits — a flickering ring finger never strobes you 2↔3↔2.
3. **`FingerCountDriver`**: new directions need a hold (`hold_frames: 4`), STOP engages
   faster (`stop_frames: 2`), and `UNKNOWN` blips hold the current direction instead of
   stutter-stepping.

### 2.2 Getting into battle

Three equivalent ways — use whichever fits the moment:

| Method | How | Notes |
|---|---|---|
| **Auto-detect** | On by default | Screenshots ~2 Hz, scores the bottom screen ROI for a battle panel (Canny + Hough line mass, plus optional template PNGs in `assets/battle_templates/`). Needs `enter_frames: 3` battle readings to switch in, `exit_frames: 5` overworld readings to leave. Fails safe to overworld. |
| **`b` key** | Cycles auto → force-battle → force-overworld → auto | Manual override when the heuristic misreads PBO's UI. |
| **Duo `1+1` gesture** | Hold ☝️ on the **LEFT** hand + ☝️ on the **RIGHT** hand together (~8 frames / ~0.27 s) | Flips forced OVERWORLD ↔ forced BATTLE. Same `1+1` flips back. Latch-until-release + 1.5 s cooldown stop bounce; missed frames only *decay* the hold meter. `v` is the keyboard backup for the same flip. |

Duo details (`game.duo`): `strict_handedness: false` by default, so any two distinct `1` hands
count (MediaPipe handedness flips when hands cross — a missed toggle is worse than a loose
one, and the hold + latch + cooldown already guard accidents). The overlay shows a `1+1`
meter that fills during the hold and reads `RELEASE` (lower a hand to re-arm) after firing.

### 2.3 Battle, cursor style (default: `battle.interact: cursor`)

Lower one hand and **point with the other** (default: `Right`, `duo.cursor_hand`):

- Index tip drives the OS cursor — same smoothing stack as pointer mode (motion-area
  window, 1€ adaptive filter, motion predictor, dead-zone, jump guard).
- **Pinch = left-click to select** (plus optional extra key via `battle.cursor.key_tap`).
- **Fist parks the cursor**: the index gate closes and coast holds the last position.
- Pinch shaping bends the index, so a live pinch **freezes the walk driver** — confirming
  never stutter-steps you.

### 2.4 Battle, taps style (`battle.interact: taps`)

No cursor. The same finger counts become **single key taps** (not held keys):

| Fingers | Key (default `battle.move_keys`) |
|---|---|
| 1 / 2 / 3 / 4 | `1` / `2` / `3` / `4` (move slots) |
| fist | `x` (cancel / back) |
| pinch | `Space` (confirm, unchanged) |

`BattleTapGate` fires once per stable hold (`stable_frames: 6`), then at most once per
`cooldown_ms: 1200` while held (hold-to-repeat is handy across turns). A *global* cooldown
after any tap blocks the next one — morphing 1→2 through a fist flash can't double-fire
cancel. `UNKNOWN` frames neither advance nor break the streak.

---

## 3. Battle-cursor auto-lock (`StillLock`)

> The problem it solves: you aim at a move, hold still to click — and hand tremor walks
> the cursor off the button at the worst moment.

**Behavior:**

1. Hand (whole hand: index tip + thumb tip + wrist, all three must agree) stays still
   **~1 second** → the cursor **locks itself**. Marker turns red with a `LOCKED` tag, HUD
   shows `Cursor: LOCKED (x, y)`.
2. While locked, the OS cursor is never moved — it stays parked — **but pinch still
   selects**: pinch-clicks land exactly on the frozen spot. Pinch shaping moves
   thumb+index a lot, so the lock **never even sees pinch frames** (updates are skipped
   while pinching, same freeze precedent as the walk driver). A pinch can neither make
   nor break the lock.
3. **~2 seconds of real hand motion** releases it (`! CURSOR FREE`). A lone jitter spike
   contributes at most +1 to the unlock meter and drains away; move-and-hold stalls out.
   15-frame relock cooldown stops the tail of the releasing motion from snap-freezing.
4. The smoothing chain keeps tracking the live hand *during* the lock, so release resyncs
   instantly with no teleport (jump guard reseeds on unlock).

Tuning (`config/settings.yaml` → `game.battle.cursor.lock`, all @ ~30 fps):

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Set `false` for a permanently free cursor |
| `still_frames` | `30` | Sustained stillness to lock (~1 s) |
| `stillness_threshold` | `0.003` | Per-point EMA speed below this = still |
| `unlock_frames` | `60` | Sustained motion to release (~2 s) |
| `motion_threshold` | `0.008` | Per-point EMA speed reaching this = moving |
| `unlock_decay` | `3` | Still frames drain the unlock meter this fast |
| `relock_cooldown_frames` | `15` | No snap re-lock right after unlock |
| `velocity_alpha` | `0.35` | Per-point speed EMA (spike rejection) |

Units are normalized-landmark units/frame. If your camera runs much faster/slower than
30 fps and the lock feels twitchy or lazy, scale the `*_frames` with your fps first,
thresholds second. Class: `StillLock` in `src/click_lock.py` (the older `ClickLock` with
~60 ms shake-unlock is untouched and still serves pointer/`full` mode).

---

## 4. All gestures & keys reference

### Pointer / full modes (`--mode pointer`, `--mode full`)

| Input | Action |
|---|---|
| Index fingertip | Move OS mouse (mirrored, motion-area window, smoothed) |
| Pinch (thumb–index touch, `full` only) | OS left-click (hysteresis `pinch_on 0.25` / `pinch_off 0.40`, `stable_frames: 2`, 250 ms cooldown) |
| Hold hand still ~1.1 s | Click-lock freezes cursor (red `LOCKED` frame); shake / deliberate move releases |
| `p` | Toggle mouse output (safe testing) |
| `q` / `Esc` | Quit |

### Game mode (`--mode game`)

| Input | Overworld | Battle (cursor) | Battle (taps) |
|---|---|---|---|
| fist | STOP | park cursor | cancel (`x`) |
| 1 / 2 / 3 / 4 fingers | walk U / D / L / R | aim hand shape (cursor follows index) | tap `1`–`4` |
| pinch | confirm (`Space`) | **click to select** | confirm (`Space`) |
| still ~1 s (cursor) | — | **auto-lock** (pinch still clicks) | — |
| 2 s motion (cursor) | — | release lock | — |
| `1+1` both hands (~0.27 s) | → force battle-cursor | → force movement | → force movement |

### Game-mode keyboard map (overlay window focused)

| Key | Action |
|---|---|
| `q` / `Esc` | Quit (releases all keys) |
| `k` | Toggle key/mouse output on/off |
| `h` | Toggle help strip |
| `b` | Cycle battle context: auto → force-battle → force-overworld → auto |
| `v` | Duo-toggle backup: flip forced OVERWORLD ↔ BATTLE |
| `c` | Recenter joystick (joystick movement mode only) |

### Overlay HUD guide

- Top-center badge: `OVERWORLD` (green) / `BATTLE` (red border + frame edge), `(FORCED)` tag
  when auto-detect is overridden.
- `Hands: Left:1...(index_only) Right:...` — per-hand stabilized bars + poses.
- `DUO 1+1: N%` meter (`RELEASE` while latched).
- Battle-cursor: crosshair marker on the aiming fingertip (white while pinching, red
  `LOCKED` while parked), `Cursor: (x, y)` / `Cursor: LOCKED … (move N%)`, `Pinch:` ratio.
- Battle-taps: 2×2 move grid with the last pick highlighted (`Battle sel: N`).
- Overworld: virtual D-pad with the active direction lit; joystick radar top-right
  (joystick movement mode).

---

## 5. Configuration (`config/settings.yaml`)

Everything is YAML with live defaults; game mode reads `game.*`:

- `game.move_keys: both` — `arrows` | `wasd` | `both` (both holds arrow+WASD together).
- `game.movement.mode: fingers` — `fingers` (0–4 counts) | `joystick` (legacy palm-position
  driving with full-frame motion area, `deadzone`, `coast_frames`).
- `game.fingers` — `flip_frames`, `hold_frames`, `stop_frames`, `max_unknown_hold`,
  `pip_thresh`, `dip_thresh`, `straight_thresh`.
- `game.actions` — `pinch_key` (confirm), `fist_key`/`palm_key`/`run_key` + hold/cooldown
  timings (joystick-mode taps; `pinch_mouse_click` stays `false` in game mode).
- `game.context` — `mode: auto`, `probe_interval_s: 0.5`, `threshold: 0.5`,
  `enter_frames: 3`, `exit_frames: 5`, `roi_bottom: 0.55`, `templates_dir`,
  `debug: false` (logs per-probe edge/template scores).
- `game.battle` — `interact: cursor|taps`, `move_keys`, `cancel_key`, `stable_frames`,
  `cooldown_ms`; `battle.cursor`: `mouse_click`, `key_tap`, `require_index`,
  `cooldown_ms`, plus `battle.cursor.lock` (see §3 table).
- `game.duo` — `enabled`, `hold_frames: 8`, `miss_decay: 2`, `cooldown_s: 1.5`,
  `strict_handedness: false`, `cursor_hand: right`.
- Pointer/full tuning lives in `pointer` (1€ filter, predictor, dead-zone, jump guard,
  motion area), `click` (pinch hysteresis), `click_lock` (pointer-mode freeze),
  `tracker` (MediaPipe confidences, `max_num_hands`, `prefer_hand`), `camera`, `display`.

Drop battle-UI crops as PNGs into `assets/battle_templates/` (create it) to sharpen
auto-detect via template matching on top of the edge heuristic.

---

## 6. Project structure

```text
BlazeHand
├── src/
│   ├── main.py            # CLI: --mode tracker|pointer|full|game
│   ├── game_mode.py       # game loop: contexts, duo toggle, battle cursor+lock
│   ├── gestures_game.py   # finger-count driving, DuoToggle, cursor picking
│   ├── game_context.py    # overworld/battle auto-detect + battle tap mapping
│   ├── game_keys.py       # held-key controller (diff-based, no stuck keys)
│   ├── game_overlay.py    # D-pad, battle grid, duo meter, cursor marker, HUD
│   ├── click_lock.py      # ClickLock (pointer) + StillLock (battle cursor)
│   ├── gestures_click.py  # pinch detector (ratio + hysteresis + EMA)
│   ├── tracker.py         # MediaPipe wrapper (process + process_all for 2 hands)
│   ├── pointer.py         # index-tip -> screen mapping + gates
│   ├── filters.py         # 1€/EMA smoothing, predictor, dead-zone, jump guard
│   ├── landmarks.py       # 21-point geometry helpers
│   ├── actions.py         # OS mouse backend (pynput)
│   ├── camera.py          # webcam capture + warmup
│   ├── overlay.py         # debug drawing (landmarks, motion area, lock frame)
│   ├── config_loader.py   # settings.yaml loader
│   └── model_utils.py     # hand_landmarker.task auto-download
├── config/settings.yaml   # all tunables (see §5)
├── models/hand_landmarker.task
├── tests/                 # hardware-free suite: 66 tests
│   ├── test_precision.py  # pinch, filters, pointer gate, ClickLock
│   ├── test_game.py       # finger states, joystick, key controller
│   ├── test_game_context.py
│   ├── test_duo_toggle.py
│   └── test_still_lock.py # §3 lock contract
├── idea.md / usecases.md  # legacy notes, not BlazeHand docs
└── requirements.txt
```

Failsafes everywhere: hand lost → keys released + state reset; any exception →
`release_all()` before exit; screenshot/scoring failures degrade to overworld.

## 7. Tests

```powershell
.venv/Scripts/python -m pytest tests/ -q
# 66 passed
```

Hardware-facing modules (`tracker`, `camera`, `main`, live screen scorer) are excluded by
design (`tests/conftest.py`); pure logic (landmarks, filters, gesture drivers, locks,
context smoother, duo toggle) is fully covered with synthetic hands.

## 8. Troubleshooting

| Symptom | Try |
|---|---|
| Cursor jitters at rest | Raise `pointer.one_euro.min_cutoff` slightly; check lighting (needs a visible bare hand) |
| Pinch never fires / fires constantly | Retune `click.pinch_on` / `pinch_off` (ratio units; keep a real gap = hysteresis) |
| Character moonwalks between counts | Raise `game.fingers.flip_frames` / `hold_frames` |
| Battle never auto-detects | Press `b` (force), add template PNGs, or set `game.context.debug: true` and watch scores |
| Cursor locks too eagerly / won't release | `battle.cursor.lock.still_frames`↑ / `unlock_frames`↓, or `motion_threshold`↓ slightly |
| Keys go to the wrong window | Click the PBO canvas once (it must hold keyboard focus); overlay only displays |
| Black camera image | Keep `camera.auto_exposure: true` unless you lock `exposure`+`gain` deliberately |
| `No module named pytest/mediapipe` | Use `.venv/Scripts/python`, not system Python |
