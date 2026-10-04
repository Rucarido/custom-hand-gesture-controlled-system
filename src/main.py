"""
Live camera gesture control — Phase 1 & 2 prototype + game overlay.

Run from project root:
  python src/main.py                  # full: pointer + pinch click
  python src/main.py --mode tracker   # camera + landmarks only
  python src/main.py --mode pointer   # move mouse, no click
  python src/main.py --mode game      # PBO hand controller (see game_mode)

Keys (pointer/full):
  q / Esc  — quit
  p        — toggle mouse control on/off (safe testing)
Keys (game): see game_mode.run_game docstring (q/k/h/c).
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

# Allow imports when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from actions import MouseActions  # noqa: E402
from camera import Camera  # noqa: E402
from click_lock import ClickLock  # noqa: E402
from config_loader import load_config  # noqa: E402
from filters import (  # noqa: E402
    Cooldown,
    DeadZone,
    JumpGuard,
    MotionPredictor,
    make_pointer_filter,
)
from gestures_click import PinchClickDetector  # noqa: E402
from overlay import draw_hand, draw_hud, draw_lock_indicator, draw_motion_area  # noqa: E402
from pointer import PointerMapper  # noqa: E402
from tracker import HandTracker  # noqa: E402


def fps_counter():
    times = deque(maxlen=30)

    def tick() -> float:
        now = time.perf_counter()
        times.append(now)
        if len(times) < 2:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    return tick


def run(mode: str) -> None:
    cfg = load_config()
    cam_cfg = cfg["camera"]
    trk_cfg = cfg["tracker"]
    ptr_cfg = cfg["pointer"]
    clk_cfg = cfg["click"]
    disp_cfg = cfg["display"]
    gesture_cfg = cfg.get("gestures", {})

    camera = Camera(
        index=cam_cfg["index"],
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        buffer_size=cam_cfg["buffer_size"],
        fps=cam_cfg.get("fps", 30),
        backend=cam_cfg.get("backend", "auto"),
        auto_focus=cam_cfg.get("auto_focus", True),
        auto_exposure=cam_cfg.get("auto_exposure", True),
        exposure=cam_cfg.get("exposure"),
        gain=cam_cfg.get("gain"),
        brightness=cam_cfg.get("brightness"),
        focus=cam_cfg.get("focus"),
        auto_white_balance=cam_cfg.get("auto_white_balance"),
    )
    tracker = HandTracker(
        max_num_hands=trk_cfg["max_num_hands"],
        min_detection_confidence=trk_cfg["min_detection_confidence"],
        min_hand_presence_confidence=trk_cfg["min_hand_presence_confidence"],
        min_tracking_confidence=trk_cfg["min_tracking_confidence"],
        use_gpu=trk_cfg.get("use_gpu", True),
        prefer_hand=trk_cfg.get("prefer_hand", ""),
    )

    pointer = PointerMapper(
        mirror=disp_cfg["mirror"],
        require_index_extended=ptr_cfg["require_index_extended"],
        motion_area=ptr_cfg.get("motion_area", {}),
        gate_release_frames=gesture_cfg.get("debounce_frames", 2),
        coast_frames=ptr_cfg.get("coast_frames", 5),
    )
    # Adaptive 1€ smoother (config `pointer/filter`) — kills rest jitter,
    # stays snappy in motion. Fed the real camera timestamp so the cutoff
    # tracks the true frame rate; followed by a constant-velocity predictor
    # hiding ~1 frame of pipeline lag.
    smoother = make_pointer_filter(ptr_cfg)
    predictor = MotionPredictor(
        lead_s=ptr_cfg.get("predict_lead_ms", 24) / 1000.0,
        max_lead_px=ptr_cfg.get("predict_max_px", 48),
    )
    dead_zone = DeadZone(threshold_px=ptr_cfg.get("dead_zone_px", 0))
    jump_guard = JumpGuard(
        ptr_cfg.get("max_jump_ratio", 0.25),
        pointer.screen_w, pointer.screen_h,
        max_rejects=ptr_cfg.get("max_rejects", 3),
    )
    pinch = PinchClickDetector(
        pinch_on=clk_cfg["pinch_on"],
        pinch_off=clk_cfg["pinch_off"],
        stable_frames=clk_cfg.get("stable_frames", 2),
        smooth_alpha_up=clk_cfg.get("smooth_alpha_up", 0.85),
        smooth_alpha_down=clk_cfg.get("smooth_alpha_down", 0.5),
    )
    cooldown = Cooldown(cooldown_ms=clk_cfg.get("cooldown_ms", 250))
    lock_cfg = cfg.get("click_lock", {})
    click_lock = ClickLock(
        stillness_threshold=lock_cfg.get("stillness_threshold", 0.003),
        stillness_frames=lock_cfg.get("stillness_frames", 32),
        shake_buffer=lock_cfg.get("shake_buffer", 8),
        shake_min_changes=lock_cfg.get("shake_min_changes", 3),
        unlock_threshold=lock_cfg.get("unlock_threshold", 0.012),
        unlock_confirm_frames=lock_cfg.get("unlock_confirm_frames", 2),
        relock_cooldown_frames=lock_cfg.get("relock_cooldown_frames", 20),
    ) if lock_cfg.get("enabled", False) else None
    frozen_pos = None
    mouse = MouseActions()
    tick_fps = fps_counter()

    window = "Gesture Control [q=quit p=toggle pointer]"
    clicks = 0

    print(f"[main] mode={mode}  |  p=toggle mouse  |  q=quit")

    try:
        while True:
            frame, ts = camera.read()
            if frame is None:
                print("[main] frame read failed")
                break

            hand = tracker.process(frame, ts)
            hud = [f"FPS: {tick_fps():.1f}", f"Mode: {mode}"]

            draw_motion_area(frame, ptr_cfg.get("motion_area", {}), disp_cfg["mirror"])

            if hand:
                draw_hand(frame, hand, disp_cfg)
                hud.append(f"Hand: {hand['handedness']}")

                if mode in ("pointer", "full"):
                    raw = pointer.landmarks_to_screen(hand["landmarks"])

                    if raw:
                        # Real camera timestamp -> correct dt inside 1€;
                        # predictor hides ~1 frame of tracking lag.
                        sx, sy = smoother.update(*raw, t=ts)
                        sx, sy = predictor.update(sx, sy, t=ts)
                        sx, sy, moved = dead_zone.apply(sx, sy)

                        if click_lock is not None:
                            # Don't accumulate stillness mid-pinch: aiming a
                            # click holds the hand still, which used to freeze
                            # the cursor right before the click landed.
                            pinching = (
                                mode == "full"
                                and pinch.state == PinchClickDetector.PINCHING
                            )
                            if not pinching:
                                lock_event = click_lock.update(hand["landmarks"])
                            else:
                                lock_event = None
                            if lock_event == "lock":
                                frozen_pos = (sx, sy)
                            elif lock_event == "unlock":
                                frozen_pos = None
                                jump_guard.reset()

                        if click_lock is not None and click_lock.is_locked() and frozen_pos is not None:
                            sx, sy = frozen_pos
                            moved = False
                        else:
                            if jump_guard.accept(sx, sy) and moved:
                                mouse.move(sx, sy)

                        hud.append(f"Cursor: ({int(sx)}, {int(sy)})")
                        hud.append(f"Lock: {'ON' if click_lock and click_lock.is_locked() else 'OFF'}")
                    else:
                        hud.append("Pointer: index not extended")
                        jump_guard.reset()
                        smoother.reset()
                        predictor.reset()
                        pointer.reset()
                        if click_lock:
                            click_lock.reset()

                    if mode == "full":
                        r = pinch.last_ratio
                        if r is not None:
                            hud.append(f"Pinch ratio: {r:.2f}")

                        # State always advances (re-arm tracked through the
                        # cooldown); the cooldown gates *firing*, not sensing.
                        action = pinch.update(hand["landmarks"])
                        if action == "click":
                            if cooldown.ready():
                                mouse.click()
                                cooldown.fire()
                                clicks += 1
                                hud.append("CLICK!")
                            else:
                                hud.append(f"Cooldown: {cooldown.remaining_ms:.0f}ms")
            else:
                hud.append("Hand: not detected")
                smoother.reset()
                predictor.reset()
                pointer.reset()
                jump_guard.reset()
                pinch.reset()
                if click_lock:
                    click_lock.reset()
                frozen_pos = None

            draw_lock_indicator(frame, click_lock is not None and click_lock.is_locked())

            hud.append(f"Mouse control: {'ON' if mouse.enabled else 'OFF'}")
            hud.append(f"Clicks: {clicks}")
            draw_hud(frame, hud)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                mouse.enabled = not mouse.enabled
                print(f"[main] mouse control {'ON' if mouse.enabled else 'OFF'}")

    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()
        print(f"[main] done. total clicks={clicks}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["tracker", "pointer", "full", "game"],
        default="full",
        help="tracker=landmarks only | pointer=move mouse | "
        "full=move+click | game=PBO hand controller overlay",
    )
    args = parser.parse_args()
    if args.mode == "game":
        from game_mode import run_game

        run_game()
    else:
        run(args.mode)


if __name__ == "__main__":
    main()
