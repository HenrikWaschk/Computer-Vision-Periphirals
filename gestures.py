import numpy as np
from utils import move_mouse
import mediapipe as mp
from MouseMover import Mouse

# MCP joints we’ll use to define the hand’s forward direction
_MCP_IDXS = [5, 9, 13, 17]

mouse_controller = Mouse()

#Hand Arithmatic
def calculate_landmark_delta_with_z(landmark_1,landmark_2):
    dx = landmark_1.x - landmark_2.x
    dy = landmark_1.y - landmark_2.y
    dz = landmark_1.z - landmark_2.z
    delta = np.array([dx, dy, dz])
    return delta

def calculate_abs_landmark_delta_with_z(landmark_1,landmark_2):
    dx = abs(landmark_1.x - landmark_2.x)
    dy = abs(landmark_1.y - landmark_2.y)
    dz = abs(landmark_1.z - landmark_2.z)
    delta = np.array([dx, dy,dz])
    return delta

def calculate_palmsize(landmarks):
    #In this function the size is calculated by averaging the vector length of every MCP to wrist and
    # the index finger MCP to the pinky MCP
    pairs_height = [[0,5],[0,9],[0,13],[0,17]]
    pairs_width = [[5,17]]
    lengths_height = []
    lengths_width = []
    for pair in pairs_height:
        lengths_height.append(calculate_abs_landmark_delta_with_z(landmarks.landmark[pair[0]],landmarks.landmark[pair[1]]))
    for pair in pairs_width:
        lengths_width.append(calculate_abs_landmark_delta_with_z(landmarks.landmark[pair[0]],landmarks.landmark[pair[1]]))
    size_length = 0
    size_width = 0
    for length in lengths_height:
        size_length += np.linalg.norm(length)
    for length in lengths_width:
        size_width += np.linalg.norm(length)
    size_length = size_length/len(lengths_height)
    size_width = size_width/len(lengths_width)
    if size_length == 0:
        size = 0.15
    if size_width == 0:
        size = 0.15
    return size_length * (len(pairs_height)/(len(pairs_height) + len(pairs_width))) + size_width * (len(pairs_width)/(len(pairs_height) + len(pairs_width)))


#fingerpositions utils
def calculate_average_index_vector(landmarks):
    pairs = [[5,6],[5,7],[5,8]]
    deltas = []
    for pair in pairs:
        deltas.append(calculate_landmark_delta_with_z(landmarks.landmark[pair[0]], landmarks.landmark[pair[1]]))
    average_delta = [0,0]
    counter = 1
    for delta in deltas:
        average_delta[0] += delta[0] * len(pairs)/counter
        average_delta[1] += delta[1] * len(pairs)/counter
        counter += 1
    average_delta[0] = average_delta[0] / 3
    average_delta[1] = average_delta[1] / 3
    return average_delta

def _avg_forward_vec_xy(landmarks):
    """
    Average 2D (x,y) vector from wrist to MCPs. This approximates the hand's
    forward direction in the image plane.
    """
    wrist = np.array([landmarks[0].x, landmarks[0].y], dtype=float)
    vecs = [np.array([landmarks[i].x, landmarks[i].y], dtype=float) - wrist for i in _MCP_IDXS]
    v = np.mean(vecs, axis=0)
    # guard against degenerate very small vectors
    if np.linalg.norm(v) < 1e-8:
        return np.array([0.0, 0.0])
    return v

'''def _palm_orientation_sign(landmarks, handedness):
    """
    Return +1 if palm faces the camera, -1 if it faces away.
    Uses thumb (CMC = 1) and pinky (MCP = 17) relative X positions.

    For Left hand:
      - Palm facing camera: thumb.x > pinky.x
      - Palm away:          thumb.x < pinky.x

    For Right hand:
      - Palm facing camera: thumb.x < pinky.x
      - Palm away:          thumb.x > pinky.x
    """
    thumb_x = landmarks[1].x
    pinky_x = landmarks[17].x

    if handedness == "Left":
        return 1.0 if thumb_x > pinky_x else -1.0
    elif handedness == "Right":
        return 1.0 if thumb_x < pinky_x else -1.0
    else:
        # fallback if handedness is unknown
        return 1.0
'''

def _palm_orientation_sign(landmarks,handedness):
    """
    Return +1 if palm faces the camera, -1 if it faces away.
    In MediaPipe, more negative z is closer to the camera.
    If MCPs (palm area) are closer than the wrist, we assume the palm faces camera.
    """
    wrist_z = float(landmarks[0].z)
    mcp_z_mean = np.mean([float(landmarks[i].z) for i in _MCP_IDXS])
    return 1.0 if mcp_z_mean < wrist_z else -1.0

def _compute_right_yaw(landmarks,handedness):
    """
    Yaw for the right hand. Positive/negative depends on rotation around Z.
    We correct for palm facing direction so that turning the hand the same way
    yields consistent sign whether the palm faces the camera or not.
    """
    v = _avg_forward_vec_xy(landmarks)              # wrist -> MCPs (x,y)
    yaw_deg = np.degrees(np.arctan2(v[1], v[0]))    # atan2(dy, dx)
    yaw_deg *= _palm_orientation_sign(landmarks,handedness)    # front/back correction
    return yaw_deg

def _compute_left_yaw(landmarks,handedness):
    """
    Yaw for the left hand. Because left/right are mirrored in image space,
    mirror X so the yaw sign convention matches the right hand.
    Also correct for palm facing direction.
    """
    v = _avg_forward_vec_xy(landmarks)
    v[0] = -v[0]                                    # mirror X for left hand
    yaw_deg = np.degrees(np.arctan2(v[1], v[0]))
    yaw_deg *= _palm_orientation_sign(landmarks,handedness)    # front/back correction
    return yaw_deg

def compute_hand_yaw(landmarks, handedness):
    """
    Public API (kept the same): compute yaw in degrees for the given hand.
    - landmarks: list of MediaPipe landmarks (with .x, .y, .z)
    - handedness: 'Left' or 'Right'
    Returns: yaw_deg in [-180, 180]
    """
    if handedness == "Left":
        return _compute_left_yaw(landmarks,handedness)
    elif handedness == "Right":
        return _compute_right_yaw(landmarks,handedness)
    else:
        # If the label is unexpected, fall back to right-hand logic (or raise)
        return _compute_right_yaw(landmarks,handedness)

def compute_hand_roll(landmarks, handedness):
    """
    Compute roll angle (rotation around forearm axis) for left/right hand.
    Roll stays consistent whether palm faces camera or away.

    Parameters:
        landmarks: list of 21 MediaPipe landmarks
        handedness: 'Left' or 'Right'

    Returns:
        roll_deg: roll angle in degrees (-180 to 180)
    """
    # Index MCP (5) and pinky MCP (17)
    index_mcp = np.array([landmarks[5].x, landmarks[5].y])
    pinky_mcp = np.array([landmarks[17].x, landmarks[17].y])

    # Vector across palm
    palm_vector = pinky_mcp - index_mcp

    # Angle of this vector relative to horizontal axis
    roll_rad = np.arctan2(palm_vector[1], palm_vector[0])
    roll_deg = np.degrees(roll_rad)

    # Normalize to [-180, 180]
    if roll_deg > 180:
        roll_deg -= 360
    elif roll_deg < -180:
        roll_deg += 360

    # Flip logic depending on hand
    if handedness == "Left":
        roll_deg = -roll_deg  # mirror correction

    # 🔑 Correct for palm orientation (use the helper we wrote)
    orientation = _palm_orientation_sign(landmarks, handedness)
    roll_deg *= orientation

    return roll_deg

def count_pinch_fingers(landmarks, pinch_threshold=0.05):
    """
    Count how many fingers are "pinched" with the thumb.

    Parameters:
        landmarks: list of 21 MediaPipe landmarks
        pinch_threshold: normalized distance threshold for detecting pinch (0-1, relative to image)

    Returns:
        pinch_count: int, number of fingers pinched with thumb
    """
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    fingers_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]  # index, middle, ring, pinky

    pinch_count = 0
    for tip in fingers_tips:
        finger_tip = np.array([tip.x, tip.y])
        distance = np.linalg.norm(finger_tip - thumb_tip)
        if distance < pinch_threshold:
            pinch_count += 1

    return pinch_count

def speed_from_fingers(num_fingers, max_fingers=4):
    """
    Map finger count (0..4) to speed_mod (0..1)
    """
    return np.clip(num_fingers / max_fingers, 0, 1)

def speed_from_fingers(num_fingers, max_fingers=4):
    """
    Map finger count (0..4) to speed_mod (0..1)
    """
    return np.clip(num_fingers / max_fingers, 0, 1)

_prev_wrist = None

def wrist_movement(landmarks, handedness):
    """
    Compute wrist delta (dx, dy) between frames for LEFT hand.
    """
    global _prev_wrist
    label = handedness.classification[0].label
    if label != "Left":
        return None  # Only use left hand for mouse movement

    # Current wrist position in normalized coords (0-1 range)
    wrist = np.array([landmarks[0].x, landmarks[0].y])

    delta = None
    if _prev_wrist is not None:
        delta = wrist - _prev_wrist  # difference between frames

    _prev_wrist = wrist
    return delta

def control_mouse_with_left_hand(landmarks, handedness):
    """
    Use left-hand movement to control mouse:
      - Wrist delta = cursor movement
      - Hand roll = speed scaling

    Parameters:
        landmarks: current list of 21 MediaPipe landmarks
        prev_landmarks: previous frame's landmarks (for delta calculation)
    """
    label = handedness.classification[0].label
    if label != "Left":
        return
        # --- 1. Compute wrist movement delta ---
    delta = wrist_movement(landmarks, handedness)  # returns np.array([dx, dy]) normalized

    if delta is None:
        return

    # --- 2. Compute roll angle and normalize ---
    speed_mod = speed_from_fingers(count_pinch_fingers(landmarks))

    # --- 3. Call move_mouse ---
    move_mouse(delta, positional_modifier=1000, speed_mod=speed_mod,
               min_speed=0.2, max_speed=3.0)

def control_mouse_with_right_hand(landmarks, handedness,handsize = 0):
    label = handedness.classification[0].label
    if label != "Right":
        return
    if handsize == 0:
        handsize = calculate_palmsize(landmarks)
    size = handsize
    #print(handsize)
    index_vector = calculate_average_index_vector(landmarks)
    index_vector[0] = -index_vector[0] / size
    index_vector[1] = -index_vector[1] / size * (9/16) - 0.1
    mouse_controller.move(np.array(index_vector))
    return

def process_click_right_hand(landmarks,handedness,handsize):
    label = handedness.classification[0].label
    if label != "Right":
        return
    if landmarks.landmark[4].y < landmarks.landmark[5].y:
        mouse_controller.click(True)
        return
    mouse_controller.click(False)
def process_gestures(landmarks,handedness,handsize):
    #control_mouse_with_right_hand(landmarks,handedness,handsize)
    process_click_right_hand(landmarks,handedness,handsize)
    return


