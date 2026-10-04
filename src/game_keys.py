"""Keyboard output for game mode — held-key management.

PBO-style browser RPGs walk while a direction key is HELD, so the
controller diffs desired vs. currently-held keys and only presses /
releases on change (no per-frame spam, no stuck keys).

Hardware-facing (pynput) but testable: pass a fake backend recording
(press/release) calls and assert the diff logic without touching the OS.
The real backend is built lazily via build_real_backend() so importing
this module never requires pynput (keeps pytest hardware-free).
"""

DIRECTIONS = ("up", "down", "left", "right")


def _default_keymap(move_keys: str = "both"):
    """Map direction -> list of physical key names to hold.

    move_keys: "arrows" | "wasd" | "both". "both" presses the arrow AND
    the WASD equivalent together — PBO accepts either, and holding both
    keeps focus/IME quirks from eating one of them. Browser games also
    sometimes scroll on arrows when the canvas lacks focus; holding WASD
    alongside keeps movement working.
    """
    arrows = {"up": "up", "down": "down", "left": "left", "right": "right"}
    wasd = {"up": "w", "down": "s", "left": "a", "right": "d"}
    if move_keys == "wasd":
        return {d: [wasd[d]] for d in DIRECTIONS}
    if move_keys == "arrows":
        return {d: [arrows[d]] for d in DIRECTIONS}
    return {d: [arrows[d], wasd[d]] for d in DIRECTIONS}  # both


def build_real_backend():
    """Return (press, release) bound to a pynput keyboard Controller.

    Imported lazily — calling this on a headless CI box raises, but
    merely importing game_keys never does.
    """
    from pynput.keyboard import Controller, Key

    _SPECIAL = {
        "up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right,
        "space": Key.space, "enter": Key.enter, "esc": Key.esc,
        "shift": Key.shift, "tab": Key.tab, "backspace": Key.backspace,
    }

    kb = Controller()

    def _resolve(name: str):
        key = _SPECIAL.get(name.lower(), name.lower())
        return key  # single char strings work directly with pynput

    def press(name: str):
        kb.press(_resolve(name))

    def release(name: str):
        kb.release(_resolve(name))

    return press, release


class GameKeyController:
    """Diff-based held-key controller.

    Usage:
        keys = GameKeyController(*build_real_backend(), move_keys="both")
        keys.set_direction("up")     # holds up (+w), releases the rest
        keys.set_direction("stop")   # releases all directions
        keys.tap("space")            # press+release for confirm
        keys.set_modifier("shift", True)  # run while held
        keys.release_all()           # failsafe (hand lost / quit)
    """

    def __init__(self, press, release, move_keys: str = "both") -> None:
        self._press = press
        self._release = release
        self.keymap = _default_keymap(move_keys)
        self.held: set[str] = set()   # physical key names currently down
        self.enabled = True           # 'k' toggles (safe testing)
        self.current_dir = "stop"

    # -- low level ------------------------------------------------------

    def _down(self, name: str):
        if name not in self.held:
            if self.enabled:
                try:
                    self._press(name)
                except Exception:
                    pass
            self.held.add(name)

    def _up(self, name: str):
        if name in self.held:
            if self.enabled:
                try:
                    self._release(name)
                except Exception:
                    pass
            self.held.discard(name)

    # -- public API -----------------------------------------------------

    def set_direction(self, direction: str) -> None:
        """Hold exactly the keys for `direction` (or nothing for stop)."""
        direction = direction if direction in DIRECTIONS else "stop"
        want: set[str] = set()
        if direction != "stop":
            want = set(self.keymap[direction])
        # All physical direction keys across the whole map are managed.
        all_dir_keys: set[str] = set()
        for keys in self.keymap.values():
            all_dir_keys.update(keys)
        for k in all_dir_keys - want:
            if k in self.held and self._is_direction_key(k):
                self._up(k)
        for k in want - self.held:
            self._down(k)
        self.current_dir = direction

    def _is_direction_key(self, name: str) -> bool:
        for keys in self.keymap.values():
            if name in keys:
                return True
        return False

    def set_modifier(self, name: str, held: bool) -> None:
        """Hold/release a modifier such as shift (run)."""
        if held:
            self._down(name)
        else:
            self._up(name)

    def tap(self, name: str) -> None:
        """Momentary press+release (confirm/cancel/menu)."""
        if not self.enabled:
            return
        try:
            self._press(name)
        except Exception:
            return
        try:
            self._release(name)
        except Exception:
            pass

    def release_all(self) -> None:
        for k in sorted(self.held):
            try:
                if self.enabled:
                    self._release(k)
            except Exception:
                pass
        self.held.clear()
        self.current_dir = "stop"
