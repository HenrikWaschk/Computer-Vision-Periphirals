import pyautogui
import numpy as np
from config import Min_speed,Max_speed,Speed_mod

def move_mouse(delta, positional_modifier=1000, speed_mod=Speed_mod, min_speed=Min_speed, max_speed=Max_speed):
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