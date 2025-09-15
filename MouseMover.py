import numpy as np
from pynput.mouse import Button,Controller
from config import Min_speed, Max_speed, Speed_mod, Threshold
from time import time

class Mouse:
    def __init__(self, positional_modifier=5, speed_mod=Speed_mod,
                 min_speed=Min_speed, max_speed=Max_speed, threshold=Threshold,
                 base_acceleration=1.0, max_acceleration=5.0, accel_step=0.1):
        """
        base_acceleration: starting multiplier when movement begins
        max_acceleration: maximum allowed multiplier
        accel_step: increment per consistent frame
        """
        self.mouse = Controller()
        self.positional_modifier = positional_modifier
        self.speed_mod = speed_mod
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.threshold = threshold

        # Acceleration parameters
        self.base_acceleration = base_acceleration
        self.max_acceleration = max_acceleration
        self.accel_step = accel_step
        self.current_acceleration = base_acceleration

        # Store previous movement vector
        self._prev_vector = None

        #Save if over Threshhold
        self.moving = False

        #Click Parameters
        #self.time_since_last_click = 0
        self.button_pressed = False

    def move(self, delta):
        """
        Move the mouse cursor with progressive directional acceleration.

        Parameters:
            delta: np.array([dx, dy]) wrist delta in normalized coordinates
        """
        if delta is None:
            self._prev_vector = None
            self.current_acceleration = self.base_acceleration
            return

        size = np.linalg.norm(delta)
        if size < self.threshold:
            self._prev_vector = None
            self.current_acceleration = self.base_acceleration
            self.moving = False
            return
        self.moving = True

        # Normalize delta
        delta_norm = delta / size

        # Scale delta by positional modifier and threshold
        dx, dy = delta_norm * self.positional_modifier * (size - self.threshold)

        # Map speed_mod (0-1) into [min_speed, max_speed]
        speed = self.min_speed + (self.max_speed - self.min_speed) * np.clip(self.speed_mod, 0, 1)

        # Directional acceleration
        if self._prev_vector is not None and np.linalg.norm(self._prev_vector) > 0:
            prev_norm = self._prev_vector / np.linalg.norm(self._prev_vector)
            cos_sim = np.dot(prev_norm, delta_norm)

            if cos_sim > 0.8:  # aligned enough to accelerate
                # increase acceleration over time
                self.current_acceleration = min(self.current_acceleration + self.accel_step, self.max_acceleration)
            else:
                # reset acceleration if direction changed
                self.current_acceleration = self.base_acceleration
        else:
            self.current_acceleration = self.base_acceleration

        # Apply speed scaling and acceleration
        dx *= speed * self.current_acceleration
        dy *= speed * self.current_acceleration

        # Move mouse
        self.mouse.move(dx, dy)

        # Save current vector
        self._prev_vector = np.array([dx, dy])

    def click(self,clicking):
        if not self.button_pressed and clicking:
            self.mouse.press(Button.left)
            self.button_pressed = True
        elif self.button_pressed and not clicking:
            self.mouse.release(Button.left)
            self.button_pressed = False