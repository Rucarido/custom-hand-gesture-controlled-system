"""Send mouse events to Windows."""

from pynput.mouse import Button, Controller


class MouseActions:
    def __init__(self) -> None:
        self.mouse = Controller()
        self.enabled = True  # toggle with 'p' key in main loop

    def move(self, x: float, y: float) -> None:
        if self.enabled:
            self.mouse.position = (int(x), int(y))

    def click(self) -> None:
        if self.enabled:
            self.mouse.click(Button.left, 1)
