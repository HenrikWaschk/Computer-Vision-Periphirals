import cv2
from Handtracking import HandTracker
from config import VideoWidth, VideoHeight
from utils import calculate_FPS_with_EMA,framerate
import time

print("OpenCV version:", cv2.__version__)
print("CUDA support:", cv2.cuda.getCudaEnabledDeviceCount() > 0)

time_processing_frame = 0
start_time_processing_frame = 0
end_time_processing_frame= 0


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
    start_time_processing_frame = time.time()
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # skip this frame to reduce load
    print(time_processing_frame)
    framerate = calculate_FPS_with_EMA(time_processing_frame)
    cv2.putText(frame, f'{framerate}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    results, annotated_frame = tracker.process_frame(frame)
    cv2.imshow("Debug Window", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    end_time_processing_frame = time.time()
    time_processing_frame = end_time_processing_frame-start_time_processing_frame

cap.release()
cv2.destroyAllWindows()

