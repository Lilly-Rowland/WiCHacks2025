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

def find_bad_catch_angles(angles):
    bad_angles = []
    if angles['elbow angle'] > 180 or angles['elbow angle'] < 154: #elbow bad
        bad_angles.append('elbow angle')
    elif angles['armpit angle'] > 109 or angles['armpit angle'] < 70: #armpit bad
        bad_angles.append('armpit angle')
    elif angles['hip angle'] > 35 or angles['hip angle'] < 20:
        bad_angles.append('hip angle')
    elif angles['knee angle'] > 65 or angles['knee angle'] < 40:
        bad_angles.append('knee angle')
    elif angles['ankle angle'] > 185 or angles['ankle angle'] < 145:
        bad_angles.append('ankle angle')
    # elif angles['foot angle'] > 70 or angles['foot angle'] < 23:
    #     bad_angles.append('foot angle')
    return bad_angles

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

        angles = {"elbow angle": elbow_angle, "armpit angle": armpit_angle, "hip angle": hip_angle, "knee angle": knee_angle, "ankle angle": ankle_angle, "foot angle": foot_angle}
 
        y_offset = 100
        for angle_name in angles:
            angle = angles[angle_name]
            cv2.putText(frame, f'{angle_name}_Angle: {angle:.2f} degrees', 
                (50, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
                )
            y_offset += 30


        # Print and display hip coordinates
        hip_x, hip_y = hip
        cv2.putText(frame, f'Hip: x={hip_x:.2f}, y={hip_y:.2f}', 
                (50, 700), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA
                )
        
   
        # Set angle ranges for good catch
        bad_angles = find_bad_catch_angles(angles)
        if bad_angles == []:
            cv2.putText(frame, 'Good Posture', 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                        )
        else:
            cv2.putText(frame, 'Bad Posture', 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                        )
            y_offset2 = 800
            for angle_name in bad_angles:
                cv2.putText(frame, f'{angle_name} INCORRECT', 
                    (50, y_offset2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
                    )
                y_offset2 += 30
    # Display the output
    cv2.imshow('Pose Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
