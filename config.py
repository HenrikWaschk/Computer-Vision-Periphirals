import pyautogui
VideoHeight = 1080
VideoWidth = 1920

#HandTracking
Model_complexity = 1
Handcount = 2
Buffer_Size = 5
Alpha = 0.8
EMA_Smoothing = False

#CursorMovement
Speed_mod=1
Min_speed=0
Max_speed=5.0
Threshold=0.6
pyautogui.FAILSAFE = False

#Utils
#0-1
Alpha_ema_fps = 0.11