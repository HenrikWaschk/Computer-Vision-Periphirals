import pyautogui
from Xlib import display, X
import time

print("Starting pyautogui test...")

# Get current mouse position
x, y = pyautogui.position()
print(f"Current mouse position: ({x}, {y})")

# Wait 2 seconds so you can switch to desktop
time.sleep(2)

# Move mouse 100 pixels right and 100 pixels down
print("Moving mouse +100,+100")
pyautogui.moveTo(100, 100)

time.sleep(1)

# Move mouse back to original position
print("Moving mouse back to original position")
pyautogui.moveTo(x, y, duration=0.5)

print("Test completed.")
