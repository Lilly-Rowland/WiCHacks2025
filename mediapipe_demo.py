import cv2
import mediapipe as mp
import numpy as np
import math as m

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

def check_elbows(elbow_angle):
 # print(elbow_angle)
  if elbow_angle >= 35 and elbow_angle <= 55:
    cv2.putText(frame, f'Good elbow angle, {elbow_angle:.2f} degrees', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  else:
     cv2.putText(frame, f'Bad elbow angle, {elbow_angle:.2f} degrees', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

def check_armpits(armpit_angle):
  if armpit_angle >= 80 and armpit_angle <= 120:
    cv2.putText(frame, f'Good armpit angle, {armpit_angle:.2f} degrees', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  else:
     cv2.putText(frame, f'Bad armpit angle, {armpit_angle:.2f} degrees', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


def check_hips(hip_angle):
  if hip_angle >= 130 and hip_angle <= 150:
    cv2.putText(frame, f'Good hip angle, {hip_angle:.2f} degrees', (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  else:
     cv2.putText(frame, f'Bad hip angle, {hip_angle:.2f} degrees', (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


def check_knees(knee_angle):
  if knee_angle >= 150 and knee_angle <= 175:
    cv2.putText(frame, f'Good knee angle, {knee_angle:.2f} degrees', (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  else:
     cv2.putText(frame, f'Bad knee angle, {knee_angle:.2f} degrees', (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


def check_ankles(ankle_angle):
  if ankle_angle >= 130 and ankle_angle <= 155:
    cv2.putText(frame, f'Good ankle angle, {ankle_angle:.2f} degrees', (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  else:
     cv2.putText(frame, f'Bad ankle angle, {ankle_angle:.2f} degrees', (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


def check_feet(feet_angle):
  if feet_angle >= 30 and feet_angle <= 40:
    cv2.putText(frame, f'Good feet angle, {feet_angle:.2f} degrees', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  else:
     cv2.putText(frame, f'Bad feet angle, {feet_angle:.2f} degrees', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)



# Open video capture (0 for webcam, or replace with 'video.mp4' for file input)
cap = cv2.VideoCapture(0)

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

        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
        toe = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
        

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        armpit_angle = calculate_angle(shoulder, elbow, hip)
        hip_angle = calculate_angle(knee, hip, shoulder)
        knee_angle = calculate_angle(hip, knee, ankle)
        ankle_angle = calculate_angle(knee, ankle, heel)
        foot_angle = calculate_angle(ankle, heel, toe)

        check_elbows(elbow_angle)
        check_armpits(armpit_angle)
        check_hips(hip_angle)
        check_knees(knee_angle)
        check_ankles(ankle_angle)
        #check_feet(foot_angle)

       # angles = {"elbow angle": elbow_angle, "armpit angle": armpit_angle, "hip angle": hip_angle, "knee angle": knee_angle, "ankle angle": ankle_angle, "foot angle": foot_angle}
 
 #       y_offset = 100
  #      for angle_name in angles:
   #         angle = angles[angle_name]
    #        cv2.putText(frame, f'{angle_name}_Angle: {angle:.2f} degrees', 
     #           (50, y_offset), 
      #          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
       #         )
        #    y_offset += 30



    # Display the output
    cv2.imshow('Pose Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
