import pyautogui
import numpy as np

def move_mouse(delta, positional_modifier=1000, speed_mod=0.5, min_speed=0.2, max_speed=3.0):
    """
    Move the mouse cursor based on wrist delta + roll-based speed scaling.

    Parameters:
        delta: np.array([dx, dy]) wrist delta in normalized coords
        positional_modifier: scales raw delta into screen movement
        speed_mod: normalized (0-1), usually inferred from roll
        min_speed: minimum speed multiplier
        max_speed: maximum speed multiplier
    """
    if delta is None:
        return

    # Scale delta by position modifier
    dx, dy = delta * positional_modifier

    # Map speed_mod (0-1) into [min_speed, max_speed]
    speed = min_speed + (max_speed - min_speed) * np.clip(speed_mod, 0, 1)

    # Apply speed scaling
    dx *= speed
    dy *= speed

    # Move mouse relative to current position
    pyautogui.moveRel(dx, dy, duration=0)