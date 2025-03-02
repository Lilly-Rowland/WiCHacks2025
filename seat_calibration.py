import cv2
import mediapipe as mp
import numpy as np
import math as m
import time as time

# Load MediaPipe Pose Landmarker
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

# Calibration Variables
calibration_start_time = None
previous_hip_x = None
catch_calibrated = False
finish_calibrated = False
catch_x_position = None
finish_x_position = None


catch_calibration_time = None

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

# Open video capture (0 for webcam, or replace with 'video.mp4' for file input)
cap = cv2.VideoCapture(0)

def calibrate_catch_or_finish(landmarks, isCatch):
    global previous_hip_x, calibration_start_time, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position
    hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
    
    if previous_hip_x is None:
        previous_hip_x = hip_x

    difference_threshold = 0.02
    if not calibration_start_time == None:
      cv2.putText(frame, f'{(time.time() - calibration_start_time):.2f}', (550, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, f'TIME: 0.0', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if abs(hip_x - previous_hip_x) < difference_threshold:
        if calibration_start_time is None:
            calibration_start_time = time.time()
        elif time.time() - calibration_start_time >= 5:
            print("Calibrated!")
            if(isCatch):
              catch_calibrated = True
              catch_x_position = hip_x
              calibration_start_time = None
            else:
                finish_calibrated = True
                finish_x_position = hip_x
            return time.time()
    else:
        calibration_start_time = None
    previous_hip_x = hip_x
   

def print_calibration_state(catch_calibrated, finish_calibrated, is_delayed):
  cv2.putText(frame, 'CALIBRATION IN PROCESS', (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
  print(finish_calibrated)
  if(catch_calibrated and is_delayed):
     cv2.putText(frame, 'CATCH CALIBRATED', (375, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
  elif(not catch_calibrated):
     cv2.putText(frame, 'SIT AT CATCH', (425, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
  elif(catch_calibrated and not is_delayed and not finish_calibrated):
     cv2.putText(frame, 'SIT AT FINISH', (425, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
  elif(finish_calibrated):
     cv2.putText(frame, f'FINISH CALIBRATED', (375, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)  

def get_catch_calibration_delay(catch_calibrated, catch_calibration_time):
   # calculate a 3 second delay after catch_calibrated is set to true
   if catch_calibrated:
      if catch_calibration_time is not None and (time.time() - catch_calibration_time) >= 3:
         return False
      return True
   return False

    
# Force the webcam to use a higher resolution (change values as needed)
cam_width = 1280  # Try 1920 for Full HD
cam_height = 720   # Try 1080 for Full HD
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

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
        
        print_calibration_state(catch_calibrated, finish_calibrated, get_catch_calibration_delay(catch_calibrated, catch_calibration_time))


        # Print and display hip coordinates
        hip_x, hip_y = hip
        cv2.putText(frame, f'Hip: x={hip_x:.2f}, y={hip_y:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        if(not catch_calibrated):
            catch_calibration_time = calibrate_catch_or_finish(landmarks, True)
            is_delayed = get_catch_calibration_delay(catch_calibrated, catch_calibration_time)
        elif(not finish_calibrated and not get_catch_calibration_delay(catch_calibrated, catch_calibration_time)):
            calibrate_catch_or_finish(landmarks, False)

        
    # Display the output
    cv2.imshow('Pose Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
