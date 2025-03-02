import cv2
import mediapipe as mp
import numpy as np
import math as m
import time as time

# Load MediaPipe Pose Landmarker
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

# Initialize Pose Landmarker
pose = mp_pose.Pose(
    static_image_mode=False,  # False for video input
    model_complexity=1,  # Higher means more accurate but slower
    enable_segmentation=True,  # Get segmentation mask (optional)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Calculates the angle of the b vertex
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = m.atan2(c[1] - b[1], c[0] - b[0]) - m.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Detects hips being held in a stable position for 5 seconds
def calibrate_catch_or_finish(frame, landmarks, isCatch, calibration_start_time, previous_hip_x):
    hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
    
    if previous_hip_x is None:
        previous_hip_x = hip_x
    

    difference_threshold = 0.02
    if calibration_start_time is not None:
        cv2.putText(frame, f'{(time.time() - calibration_start_time):.2f}', (550, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, f'0.0', (550, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
    if abs(hip_x - previous_hip_x) < difference_threshold:
        if calibration_start_time is None:
            calibration_start_time = time.time()
        elif time.time() - calibration_start_time >= 5:
            print("Calibrated!")
            if isCatch:
                return True, hip_x, None, calibration_start_time, previous_hip_x
            else:
                return True, None, hip_x, calibration_start_time, previous_hip_x
    else:
        calibration_start_time = None
    previous_hip_x = hip_x
    return False, None, None, calibration_start_time, previous_hip_x

# Prints the current calibration state
def print_calibration_state(frame, message):
    if(message == "SIT AT THE CATCH" or message == "SIT AT THE FINISH" or message == "FINISH CALIBRATED" or message == "CATCH CALIBRATED"):
        cv2.putText(frame, 'CALIBRATION IN PROCESS', (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f'{message}', (375, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

def is_calibration_delayed(calibration_time):
    if calibration_time is not None and (time.time() - calibration_time) >= 3:
        return False
    return True

def seat_calibration(frame, calibration_start_time, previous_hip_x, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position, completed_calibration_time, message, calibration_in_process):
    # Convert BGR to RGB (MediaPipe requires RGB format)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Pose Landmarker
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark
        
        if not catch_calibrated:
            catch_calibrated, catch_x_position, _, calibration_start_time, previous_hip_x = calibrate_catch_or_finish(frame, landmarks, True, calibration_start_time, previous_hip_x)
            if catch_calibrated:
                completed_calibration_time = time.time()
                message = "CATCH CALIBRATED"
                calibration_start_time = None
        elif not finish_calibrated and not is_calibration_delayed(completed_calibration_time):
            message = "SIT AT THE FINISH"
            finish_calibrated, _, finish_x_position, calibration_start_time, previous_hip_x = calibrate_catch_or_finish(frame, landmarks, False, calibration_start_time, previous_hip_x)
            if finish_calibrated:
                completed_calibration_time = time.time()
                message = "FINISH CALIBRATED"
                calibration_start_time = None
        elif message == "ROW!" and not is_calibration_delayed(completed_calibration_time):
            message = ""
            completed_calibration_time = time.time()
            calibration_start_time = None
            calibration_in_process = False
        elif message == "SITTING READY..." and not is_calibration_delayed(completed_calibration_time):
            message = "ROW!"
            print("ROW!")
            completed_calibration_time = time.time()
            calibration_start_time = None
        elif finish_calibrated and not is_calibration_delayed(completed_calibration_time) and message == "FINISH CALIBRATED": 
            message = "SITTING READY..."
            completed_calibration_time = time.time()
            calibration_start_time = None
            
        print_calibration_state(frame, message)
    
    return frame, calibration_start_time, previous_hip_x, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position, completed_calibration_time, message, calibration_in_process

def main():
    calibration_start_time = None
    previous_hip_x = None
    catch_calibrated = False
    finish_calibrated = False
    catch_x_position = None
    finish_x_position = None
    completed_calibration_time = None
    message = "SIT AT THE CATCH"
    calibration_in_process = True

    # Open video capture (0 for webcam, or replace with 'video.mp4' for file input)
    cap = cv2.VideoCapture(0)

    # Force the webcam to use a higher resolution (change values as needed)
    cam_width = 1280  # Try 1920 for Full HD
    cam_height = 720   # Try 1080 for Full HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # CALIBRATES ROWER AND BEGINS ROWING SEQUENCE
        frame, calibration_start_time, previous_hip_x, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position, completed_calibration_time, message, calibration_in_process = seat_calibration(
            frame, calibration_start_time, previous_hip_x, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position, completed_calibration_time, message, calibration_in_process
        )
        
        # Display the output
        cv2.imshow('Pose Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()