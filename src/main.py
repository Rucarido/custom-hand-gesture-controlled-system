"""
Live camera gesture control — Phase 1 & 2 prototype.

Run from project root:
  python src/main.py                  # full: pointer + pinch click
  python src/main.py --mode tracker   # camera + landmarks only
  python src/main.py --mode pointer   # move mouse, no click

Keys:
  q / Esc  — quit
  p        — toggle mouse control on/off (safe testing)
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

# Allow imports when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from actions import MouseActions
from camera import Camera
from config_loader import load_config
from filters import Cooldown, DeadZone, EMAFilter, JumpGuard
from gestures_click import PinchClickDetector
from overlay import draw_hand, draw_hud
from pointer import PointerMapper
from tracker import HandTracker


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

    camera = Camera(
        index=cam_cfg["index"],
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        buffer_size=cam_cfg["buffer_size"],
    )
    tracker = HandTracker(
        max_num_hands=trk_cfg["max_num_hands"],
        min_detection_confidence=trk_cfg["min_detection_confidence"],
        min_hand_presence_confidence=trk_cfg["min_hand_presence_confidence"],
        min_tracking_confidence=trk_cfg["min_tracking_confidence"],
    )

    pointer = PointerMapper(
        mirror=disp_cfg["mirror"],
        require_index_extended=ptr_cfg["require_index_extended"],
    )
    ema = EMAFilter(alpha=ptr_cfg["ema_alpha"])
    dead_zone = DeadZone(threshold_px=ptr_cfg["dead_zone_px"])
    jump_guard = JumpGuard(
        ptr_cfg["max_jump_ratio"], pointer.screen_w, pointer.screen_h,
    )
    pinch = PinchClickDetector(
        pinch_on=clk_cfg["pinch_on"],
        pinch_off=clk_cfg["pinch_off"],
        stable_frames=clk_cfg["stable_frames"],
    )
    cooldown = Cooldown(cooldown_ms=clk_cfg["cooldown_ms"])
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

            if hand:
                draw_hand(frame, hand, disp_cfg)
                hud.append(f"Hand: {hand['handedness']}")

                if mode in ("pointer", "full"):
                    raw = pointer.landmarks_to_screen(hand["landmarks"])

                    if raw:
                        sx, sy = ema.update(*raw)
                        sx, sy, moved = dead_zone.apply(sx, sy)

                        if jump_guard.accept(sx, sy) and moved:
                            mouse.move(sx, sy)

                        hud.append(f"Cursor: ({int(sx)}, {int(sy)})")
                    else:
                        hud.append("Pointer: index not extended")
                        jump_guard.reset()
                        ema.reset()

                    if mode == "full":
                        d = pinch.last_distance
                        if d is not None:
                            hud.append(f"Pinch dist: {d:.3f}")

                        if cooldown.ready():
                            action = pinch.update(hand["landmarks"])
                            if action == "click":
                                mouse.click()
                                cooldown.fire()
                                clicks += 1
                                hud.append("CLICK!")
                        else:
                            hud.append(f"Cooldown: {cooldown.remaining_ms:.0f}ms")
                            pinch.update(hand["landmarks"])  # keep state in sync
            else:
                hud.append("Hand: not detected")
                ema.reset()
                jump_guard.reset()
                pinch.reset()

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
        choices=["tracker", "pointer", "full"],
        default="full",
        help="tracker=landmarks only | pointer=move mouse | full=move+click",
    )
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
