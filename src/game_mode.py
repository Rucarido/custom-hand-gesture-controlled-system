"""Game mode loop — finger-count driving + battle context + overlay.

Run:
  python src/main.py --mode game

Setup for Pokemon Blaze Online (browser):
  1. Open https://play.pokemonblazeonline.com in your browser, log in,
     click the game canvas ONCE so it has keyboard focus.
  2. Start this mode. A small camera window floats on top (always-on-top).
  3. Click BACK on the browser game so IT keeps focus — pynput sends keys
     globally, the overlay only displays. To quit, focus the overlay and
     press q (or Esc).

CONTEXT-AWARE (movement.mode = "fingers" default):
  OVERWORLD — finger-count walking (held keys):
    fist (0) -> STOP | 1 -> UP | 2 -> DOWN | 3 -> LEFT | 4 -> RIGHT
    pinch -> confirm (Space)
  BATTLE (auto-detected, 'b' forces) — same fingers pick moves (taps):
    1 -> move 1 | 2 -> move 2 | 3 -> move 3 | 4 -> move 4 (keys 1-4)
    fist -> cancel/back (X) | pinch -> confirm (Space)
    Hold a shape ~0.2 s to fire; release + re-show to fire again.
    Walk keys auto-release on entering battle.

TWO-HAND TOGGLE (1+1): hold "1" on the LEFT hand and "1" on the RIGHT
hand together (~0.3 s) to flip forced OVERWORLD <-> forced BATTLE.
Same 1+1 flips back. Latch-until-release + cooldown stop bounce, and
single-frame misses only decay (never reset) the hold meter.
  BATTLE-CURSOR (battle.interact = "cursor", default): the cursor hand's
    index tip drives the OS cursor (smoothed, jump-guarded); pinch clicks
    to select. Fist parks the cursor (index gate closes, coast holds).
    Lower one hand and point with the other to aim.
    CURSOR-LOCK (battle.cursor.lock): hand still ~1 s -> the cursor
    freezes by itself; pinch does NOT break the lock — it just clicks at
    the frozen spot. ~2 s of real hand motion releases it again.

LEGACY (movement.mode = "joystick"): palm position walks in overworld;
in battle the finger taps above still apply (robust classifier).

Recognition: 4-cue finger voting + FingerStateFilter + FingerCountDriver
(overworld holds) / BattleTapGate (battle fire-once). Pinch freezes the
walk driver so confirming never stutter-steps.

Failsafes: hand lost -> all keys released; q/Esc/exception ->
release_all() in finally.
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gestures_game as GG  # noqa: E402
from camera import Camera  # noqa: E402
from click_lock import StillLock  # noqa: E402
from config_loader import load_config  # noqa: E402
from filters import (  # noqa: E402
    Cooldown,
    DeadZone,
    JumpGuard,
    MotionPredictor,
    make_pointer_filter,
)
from game_context import (  # noqa: E402
    BATTLE,
    OVERWORLD,
    BattleTapGate,
    ContextSmoother,
    battle_key_for_pose,
    read_battle_bool,
)
from game_keys import GameKeyController, build_real_backend  # noqa: E402
from game_overlay import (  # noqa: E402
    draw_battle_menu,
    draw_context_badge,
    draw_cursor_marker,
    draw_dpad,
    draw_duo_meter,
    draw_game_hud,
    draw_help,
    draw_joystick_dot,
)
from gestures_click import PinchClickDetector  # noqa: E402
from gestures_game import (  # noqa: E402
    ActionDebouncer,
    DuoToggle,
    FingerCountDriver,
    FingerStateFilter,
    VirtualJoystick,
    classify_static,
    finger_bar,
    is_duo_one_one,
    palm_center,
    pick_cursor_hand,
    robust_finger_states,
)
from landmarks import choose_hand_index  # noqa: E402
from overlay import draw_hand  # noqa: E402
from pointer import PointerMapper  # noqa: E402
from tracker import HandTracker  # noqa: E402


def make_topmost(window_name: str) -> None:
    """Pin an OpenCV window above the browser (Windows only, best-effort)."""
    try:
        import ctypes

        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if hwnd:
            HWND_TOPMOST, SWP_NOSIZE, SWP_NOMOVE = -1, 0x0001, 0x0002
            ctypes.windll.user32.SetWindowPos(
                hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE
            )
    except Exception as e:
        print(f"[game] topmost pin failed (non-fatal): {e}")


def _resolve_action_key(name: str, fallback: str) -> str:
    name = (name or fallback).strip().lower()
    aliases = {
        "space": "space", "spacebar": "space",
        "return": "enter", "enter": "enter",
        "escape": "esc", "esc": "esc",
        "shift": "shift",
    }
    if name in aliases:
        return aliases[name]
    return name[:1] if len(name) == 1 else fallback


def run_game() -> None:
    cfg = load_config()
    cam_cfg = cfg["camera"]
    trk_cfg = cfg["tracker"]
    disp_cfg = cfg["display"]
    clk_cfg = cfg["click"]
    game_cfg = cfg.get("game", {}) or {}
    move_cfg = game_cfg.get("movement", {}) or {}
    fing_cfg = game_cfg.get("fingers", {}) or {}
    joy_cfg = game_cfg.get("joystick", {}) or {}
    act_cfg = game_cfg.get("actions", {}) or {}
    over_cfg = game_cfg.get("overlay", {}) or {}
    ctx_cfg = game_cfg.get("context", {}) or {}
    bat_cfg = game_cfg.get("battle", {}) or {}
    duo_cfg = game_cfg.get("duo", {}) or {}
    cur_cfg = bat_cfg.get("cursor", {}) or {}
    ptr_cfg = cfg.get("pointer", {}) or {}

    mode = str(move_cfg.get("mode", "fingers")).lower()
    if mode not in ("fingers", "joystick"):
        print(f"[game] unknown movement.mode={mode!r}, falling back to fingers")
        mode = "fingers"
    battle_interact = str(bat_cfg.get("interact", "cursor")).lower()
    if battle_interact not in ("cursor", "taps"):
        print(f"[game] unknown battle.interact={battle_interact!r}, "
              "falling back to cursor")
        battle_interact = "cursor"
    duo_enabled = bool(duo_cfg.get("enabled", True))

    camera = Camera(
        index=cam_cfg["index"], width=cam_cfg["width"], height=cam_cfg["height"],
        buffer_size=cam_cfg["buffer_size"], fps=cam_cfg.get("fps", 30),
        backend=cam_cfg.get("backend", "auto"),
        auto_focus=cam_cfg.get("auto_focus", True),
        auto_exposure=cam_cfg.get("auto_exposure", True),
        exposure=cam_cfg.get("exposure"), gain=cam_cfg.get("gain"),
        brightness=cam_cfg.get("brightness"), focus=cam_cfg.get("focus"),
        auto_white_balance=cam_cfg.get("auto_white_balance"),
    )
    tracker = HandTracker(
        # Two-hand tracking: the 1+1 duo toggle needs both hands at once.
        # Single-hand pipelines are unaffected (they just take the winner).
        max_num_hands=max(2, trk_cfg.get("max_num_hands", 1)) if duo_enabled
        else trk_cfg.get("max_num_hands", 1),
        min_detection_confidence=trk_cfg["min_detection_confidence"],
        min_hand_presence_confidence=trk_cfg["min_hand_presence_confidence"],
        min_tracking_confidence=trk_cfg["min_tracking_confidence"],
        use_gpu=trk_cfg.get("use_gpu", True),
        prefer_hand=trk_cfg.get("prefer_hand", ""),
    )

    mirror = disp_cfg.get("mirror", True)
    joystick = VirtualJoystick(
        center_x=joy_cfg.get("center_x", 0.5),
        center_y=joy_cfg.get("center_y", 0.5),
        deadzone=joy_cfg.get("deadzone", 0.12),
        hold_frames=joy_cfg.get("hold_frames", 2),
        release_frames=joy_cfg.get("release_frames", 3),
        mirror_x=joy_cfg.get("mirror_x", mirror),
    )
    finger_filter = FingerStateFilter(
        flip_frames=fing_cfg.get("flip_frames", 3),
    )
    driver = FingerCountDriver(
        hold_frames=fing_cfg.get("hold_frames", 4),
        stop_frames=fing_cfg.get("stop_frames", 2),
        max_unknown_hold=fing_cfg.get("max_unknown_hold", 10),
    )
    pinch = PinchClickDetector(
        pinch_on=clk_cfg["pinch_on"], pinch_off=clk_cfg["pinch_off"],
        stable_frames=clk_cfg.get("stable_frames", 2),
        smooth_alpha_up=clk_cfg.get("smooth_alpha_up", 0.85),
        smooth_alpha_down=clk_cfg.get("smooth_alpha_down", 0.5),
    )
    cooldown = Cooldown(cooldown_ms=clk_cfg.get("cooldown_ms", 250))

    pinch_key = _resolve_action_key(act_cfg.get("pinch_key", "space"), "space")
    fist_key = _resolve_action_key(act_cfg.get("fist_key", "x"), "x")
    palm_key = _resolve_action_key(act_cfg.get("palm_key", "enter"), "enter")
    run_key = _resolve_action_key(act_cfg.get("run_key", "shift"), "shift")
    fist_deb = ActionDebouncer(
        stable_frames=act_cfg.get("fist_hold_frames", 12),
        cooldown_ms=act_cfg.get("fist_cooldown_ms", 800),
    )
    palm_deb = ActionDebouncer(
        stable_frames=act_cfg.get("palm_hold_frames", 20),
        cooldown_ms=act_cfg.get("palm_cooldown_ms", 1000),
    )
    run_enter = max(1, act_cfg.get("run_hold_frames", 6))
    run_exit = max(1, act_cfg.get("run_release_frames", 6))
    _run_count, _run_off_count, _run_held = 0, 0, False

    # --- battle context (auto-detect + 'b' manual override) ---
    smoother = ContextSmoother(
        enter_frames=ctx_cfg.get("enter_frames", 3),
        exit_frames=ctx_cfg.get("exit_frames", 5),
    )
    battle_move_keys = list(bat_cfg.get("move_keys", ["1", "2", "3", "4"]))
    battle_cancel_key = str(bat_cfg.get("cancel_key", "x"))
    battle_gate = BattleTapGate(
        stable_frames=bat_cfg.get("stable_frames", 6),
        cooldown_ms=bat_cfg.get("cooldown_ms", 1200),
    )
    probe_interval = float(ctx_cfg.get("probe_interval_s", 0.5))
    battle_threshold = float(ctx_cfg.get("threshold", 0.5))
    roi_bottom = float(ctx_cfg.get("roi_bottom", 0.55))
    templates_dir = str(
        ctx_cfg.get("templates_dir", "assets/battle_templates")
    )
    ctx_debug = bool(ctx_cfg.get("debug", False))
    startup = str(ctx_cfg.get("mode", "auto")).lower()
    force_mode: str | None = None  # None=auto, else OVERWORLD/BATTLE
    if startup in (OVERWORLD, BATTLE):
        force_mode = startup
        smoother.force(startup)
    context = smoother.context
    last_probe_t = 0.0
    last_battle_move: int | None = None
    _FORCE_CYCLE = [None, BATTLE, OVERWORLD]  # 'b' cycles auto->battle->over
    _force_idx = 0 if force_mode is None else (
        1 if force_mode == BATTLE else 2
    )

    # --- two-hand duo toggle (1+1 flips forced OVERWORLD <-> BATTLE) ---
    duo_toggle = DuoToggle(
        hold_frames=duo_cfg.get("hold_frames", 8),
        miss_decay=duo_cfg.get("miss_decay", 2),
        cooldown_s=float(duo_cfg.get("cooldown_s", 1.5)),
    )
    duo_strict = bool(duo_cfg.get("strict_handedness", False))
    cursor_prefer = str(duo_cfg.get("cursor_hand", "right")).title()
    if cursor_prefer not in ("Left", "Right"):
        cursor_prefer = "Right"
    # Per-hand finger filters keyed by handedness label, so each hand's
    # pose is temporally stable before the duo check runs.
    hand_filters: dict[str, FingerStateFilter] = {}

    def _hand_filter(label: str) -> FingerStateFilter:
        f = hand_filters.get(label)
        if f is None:
            f = FingerStateFilter(flip_frames=fing_cfg.get("flip_frames", 3))
            hand_filters[label] = f
        return f

    # --- battle cursor (index tip -> OS cursor, pinch -> click) ---
    cursor_mapper = PointerMapper(
        mirror=mirror,
        require_index_extended=bool(cur_cfg.get("require_index", True)),
        motion_area=ptr_cfg.get("motion_area", {}),
        gate_release_frames=ptr_cfg.get(
            "gate_release_frames",
            cfg.get("gestures", {}).get("debounce_frames", 2),
        ),
        coast_frames=ptr_cfg.get("coast_frames", 5),
    )
    cursor_smoother = make_pointer_filter(ptr_cfg)
    cursor_predictor = MotionPredictor(
        lead_s=ptr_cfg.get("predict_lead_ms", 24) / 1000.0,
        max_lead_px=ptr_cfg.get("predict_max_px", 48),
    )
    cursor_dead = DeadZone(threshold_px=ptr_cfg.get("dead_zone_px", 0))
    cursor_jump = JumpGuard(
        ptr_cfg.get("max_jump_ratio", 0.25),
        cursor_mapper.screen_w, cursor_mapper.screen_h,
        max_rejects=ptr_cfg.get("max_rejects", 3),
    )
    cursor_pinch = PinchClickDetector(
        pinch_on=clk_cfg["pinch_on"], pinch_off=clk_cfg["pinch_off"],
        stable_frames=clk_cfg.get("stable_frames", 2),
        smooth_alpha_up=clk_cfg.get("smooth_alpha_up", 0.85),
        smooth_alpha_down=clk_cfg.get("smooth_alpha_down", 0.5),
    )
    cursor_cooldown = Cooldown(
        cooldown_ms=cur_cfg.get("cooldown_ms", clk_cfg.get("cooldown_ms", 250))
    )
    cursor_mouse_click = bool(cur_cfg.get("mouse_click", True))
    _cursor_key_tap = str(cur_cfg.get("key_tap", "") or "").strip().lower()
    cursor_key_tap = (
        _resolve_action_key(_cursor_key_tap, "") if _cursor_key_tap else ""
    )
    _cursor_id: str | None = None  # handedness driving the cursor

    # --- battle auto-lock (still ~1 s -> freeze; 2 s motion -> free) ---
    lock_cfg = cur_cfg.get("lock", {}) or {}
    still_lock = StillLock(
        stillness_threshold=lock_cfg.get("stillness_threshold", 0.003),
        still_frames=lock_cfg.get("still_frames", 30),
        motion_threshold=lock_cfg.get("motion_threshold", 0.008),
        unlock_frames=lock_cfg.get("unlock_frames", 60),
        unlock_decay=lock_cfg.get("unlock_decay", 3),
        relock_cooldown_frames=lock_cfg.get("relock_cooldown_frames", 15),
        velocity_alpha=lock_cfg.get("velocity_alpha", 0.35),
    )
    lock_enabled = bool(lock_cfg.get("enabled", True))
    _frozen: tuple[int, int] | None = None  # OS cursor pos held while locked
    lock_px = None  # overlay marker pos frozen at lock time (frame px)
    _last_sent: tuple[int, int] | None = None  # last accepted cursor pos

    def _reset_cursor_nav():
        nonlocal _frozen, lock_px, _last_sent
        cursor_mapper.reset()
        cursor_smoother.reset()
        cursor_predictor.reset()
        cursor_dead.reset()
        cursor_jump.reset()
        cursor_pinch.reset()
        still_lock.reset()
        _frozen = None
        lock_px = None
        _last_sent = None

    def _switch_context(target: str, why: str):
        """Force a context switch with full input-state reset."""
        nonlocal context, force_mode, _force_idx, last_battle_move, _cursor_id
        nonlocal _run_count, _run_off_count, _run_held
        force_mode = target
        smoother.force(target)
        context = target
        _force_idx = 1 if target == BATTLE else 2
        keys.set_direction("stop")
        keys.set_modifier(run_key, False)
        _run_held = False
        _run_count, _run_off_count = 0, 0
        joystick.reset()
        driver.reset()
        battle_gate.reset()
        finger_filter.reset()
        pinch.reset()
        cursor_pinch.reset()
        for f in hand_filters.values():
            f.reset()
        _reset_cursor_nav()
        _cursor_id = None
        last_battle_move = None
        print(f"[game] context FORCED -> {target.upper()} ({why})")

    press, release = build_real_backend()
    keys = GameKeyController(press, release,
                             move_keys=game_cfg.get("move_keys", "both"))
    keys.enabled = game_cfg.get("keys_enabled_default", True)
    pinch_click_mouse = bool(act_cfg.get("pinch_mouse_click", False))
    from actions import MouseActions

    # Mouse is required for battle-cursor mode; otherwise only when the
    # legacy pinch-click option asks for it.
    mouse = MouseActions() if (
        pinch_click_mouse or battle_interact == "cursor"
    ) else None

    window = ("PBO Hand Controller "
              "[q=quit b=battle v=duo-toggle k=keys h=help c=center]")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if over_cfg.get("always_on_top", True):
        make_topmost(window)
    try:
        w = int(over_cfg.get("width", 640))
        cv2.resizeWindow(window, w, int(w * 9 / 16))
    except Exception:
        pass

    times = deque(maxlen=30)
    show_help = bool(over_cfg.get("help", True))
    last_palm = None
    print(f"[game] mode=game movement={mode} context={smoother.context} "
          f"{'(forced)' if force_mode else '(auto)'} | focus BROWSER game")
    if mode == "fingers":
        print("[game] OVERWORLD: fist=STOP 1=UP 2=DOWN 3=LEFT 4=RIGHT "
              f"pinch={pinch_key}")
    else:
        print(f"[game] OVERWORLD joystick walk | pinch={pinch_key} "
              f"fist={fist_key} palm={palm_key} run={run_key}")
    print(f"[game] BATTLE: 1/2/3/4 fingers = moves {battle_move_keys} "
          f"fist={battle_cancel_key} (cancel) pinch={pinch_key} | 'b' toggles")
    if duo_enabled:
        print("[game] DUO TOGGLE: hold '1' on LEFT + '1' on RIGHT "
              f"(~{duo_toggle.hold_frames} frames) to flip "
              "movement <-> battle-cursor; same 1+1 flips back")
    if battle_interact == "cursor":
        print(f"[game] BATTLE-CURSOR: {cursor_prefer} index aims, pinch "
              f"{'clicks' + (' + ' + cursor_key_tap if cursor_key_tap else '')}"
              " to select; fist parks the cursor")
        if lock_enabled:
            print("[game] CURSOR-LOCK: still "
                  f"{still_lock.still_frames} frames (~1s) -> lock "
                  "(pinch still clicks); "
                  f"{still_lock.unlock_frames} frames (~2s) motion -> free")

    try:
        while True:
            now = time.perf_counter()
            times.append(now)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

            frame, ts = camera.read()
            if frame is None:
                print("[game] frame read failed")
                break

            # --- context probe (throttled ~2 Hz; forced mode skips) ---
            if force_mode is None and (now - last_probe_t) >= probe_interval:
                last_probe_t = now
                try:
                    reading = read_battle_bool(
                        threshold=battle_threshold,
                        roi_bottom=roi_bottom,
                        templates_dir=templates_dir,
                        debug=ctx_debug,
                    )
                except Exception:
                    reading = False
                prev_ctx = smoother.context
                context = smoother.update(reading)
                if context != prev_ctx:
                    print(f"[game] context -> {context.upper()}")
                    # Never walk into/out of a fight: drop walk + run keys
                    # on every transition, reset both drivers' momentum.
                    keys.set_direction("stop")
                    keys.set_modifier(run_key, False)
                    _run_held = False
                    _run_count, _run_off_count = 0, 0
                    joystick.reset()
                    driver.reset()
                    battle_gate.reset()
                    finger_filter.reset()
                    for _f in hand_filters.values():
                        _f.reset()
                    _reset_cursor_nav()
                    _cursor_id = None
                    last_battle_move = None
            elif force_mode is not None:
                context = force_mode
            else:
                context = smoother.context

            hands = tracker.process_all(frame, ts)
            hud = [f"FPS: {fps:.1f}  Mode: game-{mode} + {context} (PBO)"]
            pose, direction = "no-hand", "stop"
            bar = ""
            hands_info: list[dict] = []
            cursor_px = None
            cursor_active = False
            cursor_pinching = False

            if hands:
                for _h in hands:
                    draw_hand(frame, _h, disp_cfg)
                # Per-hand stabilized poses (one temporal filter per
                # handedness label, so each hand is stable before the duo
                # check or the primary pipeline consumes it).
                for _h in hands:
                    _lm = _h["landmarks"]
                    _label = str(_h.get("handedness", "Unknown"))
                    _raw = robust_finger_states(
                        _lm,
                        pip_thresh=fing_cfg.get("pip_thresh", 140.0),
                        dip_thresh=fing_cfg.get("dip_thresh", 130.0),
                        straight_thresh=fing_cfg.get("straight_thresh", 0.80),
                    )
                    _stable = _hand_filter(_label).update(_raw)
                    hands_info.append({
                        "pose": GG._classify_from_states(_stable),
                        "handedness": _label,
                        "index_extended": bool(_stable.get("index")),
                        "landmarks": _lm,
                        "stable": _stable,
                        "bar": finger_bar(_stable),
                    })
                # Primary hand drives movement/taps (config preference,
                # else the largest = most intentional hand).
                _prim_idx = choose_hand_index(
                    [h["landmarks"] for h in hands_info],
                    [h["handedness"] for h in hands_info],
                    trk_cfg.get("prefer_hand", ""),
                )
                _prim = hands_info[max(0, _prim_idx)]
                lm = _prim["landmarks"]
                pose = _prim["pose"]
                bar = _prim["bar"]
                last_palm = palm_center(lm)
                hud.append("Hands: " + " ".join(
                    f"{h['handedness']}:{h['bar']}({h['pose']})"
                    for h in hands_info
                ))

                # --- duo 1+1 toggle: flip forced OVERWORLD <-> BATTLE ---
                _duo = is_duo_one_one(
                    hands_info, strict_handedness=duo_strict,
                ) if duo_enabled else False
                if duo_enabled:
                    hud.append(
                        f"DUO 1+1: {int(duo_toggle.progress * 100)}%"
                        f"{' LATCHED' if duo_toggle.latched else ''}"
                    )
                    if duo_toggle.update(_duo, now):
                        _target = OVERWORLD if context == BATTLE else BATTLE
                        _switch_context(_target, "duo 1+1")
                        hud.append(f"! DUO TOGGLE -> {_target.upper()}")

                # Pinch state first: shaping a pinch bends the index, so a
                # live pinch freezes the overworld walk driver (hold dir).
                pinching = (
                    pinch.state == PinchClickDetector.PINCHING
                    and mode == "fingers"
                    and context == OVERWORLD
                )

                if context == BATTLE and battle_interact == "cursor":
                    # --- battle-cursor: index aims, pinch clicks ---
                    direction = "stop"  # no walking in battle, ever
                    keys.set_direction("stop")
                    _ci = pick_cursor_hand(hands_info, prefer=cursor_prefer)
                    if _ci != -1:
                        _ch = hands_info[_ci]
                        _clm = _ch["landmarks"]
                        _clabel = _ch["handedness"]
                        if _cursor_id != _clabel:
                            _cursor_id = _clabel  # new aiming hand: reseed
                            _reset_cursor_nav()
                        # Battle auto-lock: feed the lock ONLY when not
                        # pinching. Shaping/holding a pinch swings thumb +
                        # index hard and would feed the motion meter, so the
                        # lock never sees it — pinch stays purely a click
                        # at the frozen cursor (uses prior-frame state, same
                        # freeze precedent as the walk driver).
                        if (lock_enabled and cursor_pinch.state
                                != PinchClickDetector.PINCHING):
                            lock_event = still_lock.update(_clm)
                        else:
                            lock_event = None
                        _raw_c = cursor_mapper.landmarks_to_screen(_clm)
                        if _raw_c is not None:
                            _sx, _sy = cursor_smoother.update(*_raw_c, t=ts)
                            _sx, _sy = cursor_predictor.update(_sx, _sy, t=ts)
                            _sx, _sy, _moved = cursor_dead.apply(_sx, _sy)
                            _accepted = cursor_jump.accept(_sx, _sy) and _moved
                            if lock_event == "lock":
                                _frozen = (
                                    _last_sent
                                    if _last_sent is not None
                                    else (int(_sx), int(_sy))
                                )
                                _fh0, _fw0 = frame.shape[:2]
                                _tip0 = _clm[HandTracker.INDEX_TIP]
                                lock_px = (
                                    int(_tip0[0] * _fw0), int(_tip0[1] * _fh0),
                                )
                                hud.append("! CURSOR LOCKED (held still)")
                            elif lock_event == "unlock":
                                _frozen = None
                                lock_px = None
                                cursor_jump.reset()  # resync instantly
                                hud.append("! CURSOR FREE (2s motion)")
                            if _accepted:
                                _last_sent = (int(_sx), int(_sy))
                                # While locked the filters keep tracking the
                                # live hand (seamless resync on unlock) but
                                # the OS cursor is never moved — it stays
                                # parked, so pinch-clicks land on target.
                                if (not still_lock.is_locked()
                                        and keys.enabled
                                        and mouse is not None):
                                    mouse.move(_sx, _sy)
                            cursor_active = True
                            _fh, _fw = frame.shape[:2]
                            _tip = _clm[HandTracker.INDEX_TIP]
                            cursor_px = (
                                int(_tip[0] * _fw), int(_tip[1] * _fh),
                            )
                            if still_lock.is_locked():
                                if lock_px is not None:
                                    cursor_px = lock_px  # marker parks too
                                hud.append(
                                    f"Cursor: LOCKED {_frozen} "
                                    f"(move {still_lock.move_progress * 100:.0f}%)"
                                )
                            else:
                                hud.append(f"Cursor: ({int(_sx)}, {int(_sy)})")
                        else:
                            hud.append("Cursor: index not extended")
                        cursor_pinching = (
                            cursor_pinch.state == PinchClickDetector.PINCHING
                        )
                        _caction = cursor_pinch.update(_clm)
                        _r = cursor_pinch.last_ratio
                        if _r is not None:
                            hud.append(f"Pinch: {_r:.2f}")
                        if _caction == "click":
                            if cursor_cooldown.ready():
                                if mouse is not None and keys.enabled:
                                    mouse.click()
                                if cursor_key_tap:
                                    keys.tap(cursor_key_tap)
                                cursor_cooldown.fire()
                                hud.append("! CLICK (select)")
                            else:
                                hud.append(
                                    f"Cooldown: "
                                    f"{cursor_cooldown.remaining_ms:.0f}ms"
                                )
                    else:
                        _cursor_id = None
                        cursor_pinch.reset()
                        hud.append("Cursor: no pointing hand")
                    hud.append(f"Aim: CURSOR ({cursor_prefer} hand)")
                elif context == BATTLE:
                    # --- battle: fingers = move taps, fist = cancel ---
                    direction = "stop"  # no walking in battle, ever
                    bkey = battle_key_for_pose(
                        pose,
                        move_keys=battle_move_keys,
                        cancel_key=battle_cancel_key,
                    )
                    fired_battle = battle_gate.update(bkey, now)
                    if fired_battle is not None:
                        keys.tap(fired_battle)
                        try:
                            last_battle_move = battle_move_keys.index(
                                fired_battle
                            )
                        except ValueError:
                            last_battle_move = None  # cancel key, no box
                        tag = ("cancel" if fired_battle == battle_cancel_key
                               else f"move {fired_battle}")
                        hud.append(f"! {fired_battle.upper()} ({tag})")
                    hud.append(f"Battle sel: "
                               f"{last_battle_move + 1 if last_battle_move is not None else '-'}")
                elif mode == "fingers":
                    if not pinching:
                        direction = driver.update(pose)
                    else:
                        direction = driver.active  # freeze while pinching
                    keys.set_direction(direction)
                else:
                    pose_legacy = classify_static(lm)
                    pose = pose_legacy  # HUD shows the legacy decision
                    # --- run modifier (peace hold with hysteresis) ---
                    if pose == "peace":
                        _run_count += 1
                        _run_off_count = 0
                    else:
                        _run_off_count += 1
                        if _run_off_count >= run_exit:
                            _run_count = 0
                    want_run = _run_held or (_run_count >= run_enter)
                    if want_run and not _run_held:
                        _run_held = True
                        keys.set_modifier(run_key, True)
                    elif not want_run and _run_held:
                        if pose != "peace":
                            _run_held = False
                            keys.set_modifier(run_key, False)

                    # --- discrete taps (joystick-overworld only) ---
                    fired = fist_deb.update(
                        "fist" if pose == "fist" else "other", now
                    )
                    if fired == "fist":
                        keys.tap(fist_key)
                        hud.append(f"! {fist_key.upper()} (cancel)")
                    fired_palm = palm_deb.update(
                        "open_palm" if pose == "open_palm" else "other", now
                    )
                    if fired_palm == "open_palm":
                        keys.tap(palm_key)
                        hud.append(f"! {palm_key.upper()} (menu)")

                    if pose in ("fist", "open_palm"):
                        direction = "stop"
                        joystick.reset()
                        keys.set_direction("stop")
                    else:
                        direction = joystick.update(*last_palm)
                        keys.set_direction(direction)

                # --- pinch confirm (overworld + battle-taps; battle-cursor
                # has its own cursor_pinch -> click path above) ---
                if not (context == BATTLE and battle_interact == "cursor"):
                    action = pinch.update(lm)
                    r = pinch.last_ratio
                    if r is not None:
                        hud.append(f"Pinch: {r:.2f}")
                    if action == "click":
                        if cooldown.ready():
                            keys.tap(pinch_key)
                            if mouse is not None and pinch_click_mouse:
                                mouse.click()
                            cooldown.fire()
                            hud.append(f"! {pinch_key.upper()} (confirm)")
                        else:
                            hud.append(
                                f"Cooldown: {cooldown.remaining_ms:.0f}ms")

                hud.append(f"Pose: {pose}")
                extra = "" if mode == "fingers" else (
                    " +RUN" if _run_held else ""
                )
                hud.append(f"Dir: {direction}{extra}")
                hud.append(f"Held: {','.join(sorted(keys.held)) or '-'}")
            else:
                hud.append("Hand: not detected")
                hud.append("Dir: stop (released)")
                if duo_enabled:
                    duo_toggle.update(False, now)  # decay hold / re-arm
                joystick.reset()
                finger_filter.reset()
                for _f in hand_filters.values():
                    _f.reset()
                driver.reset()
                battle_gate.reset()
                pinch.reset()
                cursor_pinch.reset()
                _reset_cursor_nav()
                _cursor_id = None
                fist_deb.reset()
                palm_deb.reset()
                _run_count, _run_off_count, _run_held = 0, 0, False
                keys.set_modifier(run_key, False)
                keys.set_direction("stop")
                last_palm = None
                last_battle_move = None

            hud.append(f"CTX: {context.upper()}"
                       f"{' (FORCED)' if force_mode else ' (auto)'} (b/v)")
            hud.append(f"KEYS: {'ON' if keys.enabled else 'OFF'} (k)")
            if context == OVERWORLD and mode == "joystick":
                hud.append(
                    f"Joy center: ({joystick.cx:.2f},{joystick.cy:.2f}) (c)"
                )
            elif context == OVERWORLD:
                hud.append("Show fingers to camera, hold shape to walk")
            elif battle_interact == "cursor":
                hud.append(f"Aim {cursor_prefer} index, pinch to select")
            else:
                hud.append(f"Battle moves: {','.join(battle_move_keys)} "
                           f"cancel={battle_cancel_key}")

            draw_joystick_dot(frame, last_palm,
                              center=(joystick.cx, joystick.cy),
                              mirror_x=joystick.mirror_x)
            draw_context_badge(frame, context, forced=force_mode is not None)
            if context == BATTLE and battle_interact == "cursor":
                if cursor_px is not None:
                    draw_cursor_marker(frame, cursor_px,
                                       pinching=cursor_pinching,
                                       locked=still_lock.is_locked())
            elif context == BATTLE:
                draw_battle_menu(frame, selected=last_battle_move)
            else:
                draw_dpad(frame, direction, run=_run_held)
            if duo_enabled:
                draw_duo_meter(frame, duo_toggle.progress,
                               latched=duo_toggle.latched)
            draw_game_hud(frame, hud)
            if show_help:
                draw_help(frame, mode=mode, context=context,
                          interact=battle_interact)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("k"):
                keys.enabled = not keys.enabled
                if not keys.enabled:
                    keys.release_all()
                print(f"[game] key output {'ON' if keys.enabled else 'OFF'}")
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("b"):
                _force_idx = (_force_idx + 1) % len(_FORCE_CYCLE)
                force_mode = _FORCE_CYCLE[_force_idx]
                if force_mode is None:
                    print("[game] context -> AUTO")
                else:
                    _switch_context(force_mode, "key b")
            elif key == ord("v"):
                # Keyboard backup for the duo 1+1 gesture (same effect:
                # flip forced OVERWORLD <-> BATTLE). Handy for testing
                # the battle-cursor without a two-hand hold.
                _switch_context(
                    OVERWORLD if context == BATTLE else BATTLE, "key v")
            elif key == ord("c") and last_palm is not None:
                joystick.cx = last_palm[0]
                joystick.cy = last_palm[1]
                print(f"[game] joystick recentered to "
                      f"({joystick.cx:.3f}, {joystick.cy:.3f})")
    finally:
        keys.release_all()
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()
        print("[game] done. all keys released.")


if __name__ == "__main__":
    run_game()
