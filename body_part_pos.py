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

def find_bad_angles(angles, phase):
    bad_angles = []
    if phase == 'catch':
        angle_ranges = {
            'elbow angle': (154, 180),
            'armpit angle': (80, 109),
            'hip angle': (23, 35),
            'knee angle': (40, 65),
            'ankle angle': (145, 185)
        }
    elif phase == 'finish':
        angle_ranges = {
            'elbow angle': (35, 55),
            'armpit angle': (80, 120),
            'hip angle': (130, 150),
            'knee angle': (150, 175),
            'ankle angle': (145, 155)
        }
    else:
        return bad_angles

    for angle_name, (min_angle, max_angle) in angle_ranges.items():
        if angles[angle_name] < min_angle or angles[angle_name] > max_angle:
            bad_angles.append(angle_name)
    
    return bad_angles

def compare_pos(l1, l2):
    if l1[0] < l2[0]:
        return  " is in front of " 
    elif l1[0] > l2[0]:
        return " is behind " 
    else:
        return " is in the same positon as "

def posture_detection(catch_range_max, finish_range_min):
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

            # part = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "toe"]
            part = [("hip", hip), ("shoulder", shoulder), ("wrist", wrist), ("knee", knee), ]

        # compare the landmarks' positions for hips, shouders, wrists, and knees, and print it on the video.

        '''NOTES:
        start_pos = wrist is in front of knee, the rest is behind
        end_pos = hip is in front of shoulder and wrist, rest is behind'''
        cnt = 0
        for p in range(len(part) - 1):
            for a in range(p+1, len(part)):
                pos = compare_pos(part[p][1], part[a][1])
                if pos == " is in front of ":
                    cv2.putText(frame, str(part[p][0]) + pos + str(part[a][0]), (50,50 + cnt*25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                else:
                    cv2.putText(frame, str(part[p][0]) + pos + str(part[a][0]), (50,50 + cnt*25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cnt += 1
        # cv2.putText(frame, "wrist: " + str(wrist[0]), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        # cv2.putText(frame, "knee: " + str(knee[0]), (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        angles = {"elbow angle": elbow_angle, "armpit angle": armpit_angle, "hip angle": hip_angle, "knee angle": knee_angle, "ankle angle": ankle_angle, "foot angle": foot_angle}
        
        # Display the output
        cv2.imshow('Pose Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    catch_range_max = 0.6
    finish_range_min = 0.8
    posture_detection(catch_range_max, finish_range_min)
