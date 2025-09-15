import cv2
from Handtracking import HandTracker
from config import VideoWidth, VideoHeight
import time
from Framerate import fps_tracker


print("OpenCV version:", cv2.__version__)
print("CUDA support:", cv2.cuda.getCudaEnabledDeviceCount() > 0)

tracker = HandTracker()

gst_pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg, width=1920, height=1080, framerate=30/1 ! "
    "jpegdec ! videoconvert ! "
    "appsink max-buffers=1 drop=true sync=false"
)


cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
'''
cap.set(cv2.CAP_PROP_FRAME_WIDTH, VideoWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VideoHeight)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

cap.set(cv2.CAP_PROP_FPS, 30)
'''

# Frame skipping parameters
frame_skip = 1  # process every frame; increase to 2 or 3 to skip frames
frame_count = 0

while True:
    fps_tracker.update()
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # skip this frame to reduce load
    cv2.putText(frame, f'{fps_tracker.get_fps()}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    results, annotated_frame = tracker.process_frame(frame)
    cv2.imshow("Debug Window", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

