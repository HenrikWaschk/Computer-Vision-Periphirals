import cv2
from Handtracking import HandTracker
from config import VideoWidth, VideoHeight

tracker = HandTracker()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, VideoWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VideoHeight)

# Frame skipping parameters
frame_skip = 2  # process every frame; increase to 2 or 3 to skip frames
frame_count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # skip this frame to reduce load

    results, annotated_frame = tracker.process_frame(frame)

    cv2.imshow("Debug Window", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

