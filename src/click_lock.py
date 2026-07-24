from tracker import HandTracker


class ClickLock:
    UNLOCKED = "unlocked"
    LOCKED = "locked"

    def __init__(
        self,
        stillness_threshold: float = 0.005,
        stillness_frames: int = 10,
        shake_buffer: int = 5,
        shake_min_changes: int = 3,
    ) -> None:
        self.stillness_threshold = stillness_threshold
        self.stillness_frames = stillness_frames
        self.shake_buffer = shake_buffer
        self.shake_min_changes = shake_min_changes

        self.state = self.UNLOCKED
        self._still_count = 0
        self._prev_pos = None
        self._dir_buffer = []

    def update(self, landmarks) -> str | None:
        index = landmarks[HandTracker.INDEX_TIP]
        x, y = index[0], index[1]

        if self.state == self.UNLOCKED:
            if self._prev_pos is not None:
                dx = x - self._prev_pos[0]
                dy = y - self._prev_pos[1]
                dist = (dx * dx + dy * dy) ** 0.5

                if dist < self.stillness_threshold:
                    self._still_count += 1
                    if self._still_count >= self.stillness_frames:
                        self.state = self.LOCKED
                        self._still_count = 0
                        self._prev_pos = (x, y)
                        return "lock"
                else:
                    self._still_count = 0
            self._prev_pos = (x, y)

        else:
            if self._prev_pos is not None:
                dx = x - self._prev_pos[0]
                direction = 0
                if dx > 0.001:
                    direction = 1
                elif dx < -0.001:
                    direction = -1

                self._dir_buffer.append(direction)
                if len(self._dir_buffer) > self.shake_buffer:
                    self._dir_buffer.pop(0)

                changes = 0
                for i in range(1, len(self._dir_buffer)):
                    a = self._dir_buffer[i - 1]
                    b = self._dir_buffer[i]
                    if a != 0 and b != 0 and a != b:
                        changes += 1

                if changes >= self.shake_min_changes:
                    self.state = self.UNLOCKED
                    self._dir_buffer.clear()
                    self._still_count = 0
                    self._prev_pos = (x, y)
                    return "unlock"
            self._prev_pos = (x, y)

        return None

    def is_locked(self) -> bool:
        return self.state == self.LOCKED

    def reset(self) -> None:
        self.state = self.UNLOCKED
        self._still_count = 0
        self._prev_pos = None
        self._dir_buffer.clear()
