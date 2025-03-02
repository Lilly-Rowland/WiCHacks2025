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

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = m.atan2(c[1] - b[1], c[0] - b[0]) - m.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def calibrate_catch_or_finish(landmarks, isCatch, calibration_start_time, previous_hip_x):
    hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
    
    if previous_hip_x is None:
        previous_hip_x = hip_x

    difference_threshold = 0.02

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

def seat_calibration(frame, calibration_start_time, previous_hip_x, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position):
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

        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        # Print and display hip coordinates
        hip_x, hip_y = hip
        cv2.putText(frame, f'Hip: x={hip_x:.2f}, y={hip_y:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        if not catch_calibrated:
            catch_calibrated, catch_x_position, _, calibration_start_time, previous_hip_x = calibrate_catch_or_finish(landmarks, True, calibration_start_time, previous_hip_x)
        elif abs(hip_x - catch_x_position) > 0.02 and not finish_calibrated:
            finish_calibrated, _, finish_x_position, calibration_start_time, previous_hip_x = calibrate_catch_or_finish(landmarks, False, calibration_start_time, previous_hip_x)
        
        if catch_calibrated:
            cv2.putText(frame, f'CATCH CALIBRATED: {catch_x_position:.2f}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'CATCH NOT CALIBRATED', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if finish_calibrated:
            cv2.putText(frame, f'FINISH CALIBRATED: {finish_x_position:.2f}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'FINISH NOT CALIBRATED', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    return frame, calibration_start_time, previous_hip_x, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position

def main():
    calibration_start_time = None
    previous_hip_x = None
    catch_calibrated = False
    finish_calibrated = False
    catch_x_position = None
    finish_x_position = None

    # Open video capture (0 for webcam, or replace with 'video.mp4' for file input)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame, calibration_start_time, previous_hip_x, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position = seat_calibration(
            frame, calibration_start_time, previous_hip_x, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position
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
