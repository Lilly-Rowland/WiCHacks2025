import cv2
import mediapipe as mp
import numpy as np
import math as m
from seat_calibration2 import seat_calibration

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
            'hip angle': (27, 40),#(23, 33),
            'knee angle': (40,60), #(45, 65)
            'ankle angle': (145, 185)
        }
    elif phase == 'finish':
        angle_ranges = {
            'elbow angle': (35, 60),
            'armpit angle': (80, 120),
            'hip angle': (110, 160),
            'knee angle': (140, 180),
            #'ankle angle': (145, 155)
        }
    else:
        return bad_angles

    for angle_name, (min_angle, max_angle) in angle_ranges.items():
        if angles[angle_name] < min_angle or angles[angle_name] > max_angle:
            bad_angles.append(angle_name)
    
    return bad_angles

def compare_pos(l1, l2):
    if l1[0] <= l2[0]:
        return True
    elif l1[0] > l2[0]:
        return False 
    
def check_leg_position(knee_angle, isDrive):
    if(isDrive):
        if knee_angle >= 150 and knee_angle <= 175:
            return True
    else:
        if knee_angle >= 40 and knee_angle <= 60:
            return True
    return False

def shoulders_behind_hips(shoulder, hip, isDrive):
    if isDrive:
        return compare_pos(shoulder, hip)
    
    
    
def detect_drive(checklist, knee_angle, shoulder, hip, wrist):
    if(checklist[0] is False):
        print("DRIVE LOOKING FOR LEGS")
        legs_straight =  check_leg_position(knee_angle, True)
        if legs_straight:
            checklist[0] = True
    elif(checklist[1] is False):
        print("DRIVE LOOKING FOR BACK")
        shoulder_in_front_of_hips = compare_pos(shoulder, hip)
        if not shoulder_in_front_of_hips:
            checklist[1] = True
    elif(checklist[2] is False):
        print("DRIVE LOOKING FOR ARMS")
        wrist_in_front_of_hips = compare_pos(wrist, hip)
        if not wrist_in_front_of_hips:
            checklist[2] = True
    return checklist

def detect_catch(checklist, knee_angle, shoulder, hip, wrist):
    if checklist[0] is False:
        print("CATCH LOOKING FOR ARMS")
        wrist_in_front_of_hips = compare_pos(wrist, hip)
        if wrist_in_front_of_hips:
            checklist[0] = True
    elif checklist[1] is False:
        print("CATCH LOOKING FOR SHOULDERS")
        shoulder_in_front_of_hips = compare_pos(shoulder, hip)
        print(shoulder_in_front_of_hips)
        if shoulder_in_front_of_hips:
            checklist[1] = True
    elif checklist[2] is False:
        print("CATCH LOOKING FOR LEGS")
        legs_bent = check_leg_position(knee_angle, False)
        if legs_bent:
            checklist[2] = True
    return checklist

def verify_checklist(checklist):
    if(checklist[0] is True and checklist[1] is True and checklist[2] is True):
        return True
    return False
  

def process_frame(frame, catch_range_max, finish_range_min, phase, checklist):
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
        bad_angles = []

        if phase is None:
            phase = 'drive'
        
        # Print and display hip coordinates
        hip_x, hip_y = hip
        cv2.putText(frame, f'Hip: x={hip_x:.2f}, y={hip_y:.2f}', 
                (50, 700), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA
                )
        
        if phase == 'drive': #hip_x <= finish_range_min and hip_x >= catch_range_max:
            cv2.putText(frame, 'DRIVE ;)', 
                (frame.shape[1] // 2 - 300, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 8, cv2.LINE_AA, False
                )
            if verify_checklist(checklist):
                phase = 'finish'
                bad_angles = find_bad_angles(angles, phase)
            else:
                checklist = detect_drive(checklist, knee_angle, shoulder, hip, wrist)
        elif phase == "recovery":
            cv2.putText(frame, 'RECOVERY ;)', 
                (frame.shape[1] // 2 - 300, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 8, cv2.LINE_AA, False
                )
            if verify_checklist(checklist):
                phase = 'catch'
                bad_angles = find_bad_angles(angles, phase)
            else:
                checklist = detect_catch(checklist, knee_angle, shoulder, hip, wrist)

            return frame, phase, checklist
        
        #Determine if posture is good or not
        if (phase == 'catch' or phase == 'finish'):
          bad_angles = find_bad_angles(angles, phase)
        if (phase == 'catch' or phase == 'finish') and bad_angles == []:
            cv2.putText(frame, 'Good Posture', 
                        (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 8, cv2.LINE_AA
                        )
        elif (phase == 'catch' or phase == 'finish'):
            cv2.putText(frame, 'Bad Posture', 
                        (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 225), 8, cv2.LINE_AA
                        )
            y_offset = 200
            for angle_name in bad_angles:
                cv2.putText(frame, f'{angle_name} INCORRECT', 
                    (50, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 225), 4, cv2.LINE_AA
                    )
                y_offset += 70
    # if you are at the finish
    # check if arms come over the hips then good to switch
    # Determine if you are out of catch or finish
    if(phase == 'catch' and hip_x > catch_range_max + .05):
      phase = 'drive'
      checklist = [False, False, False]
    elif(phase == 'finish' and compare_pos(wrist, hip)):
      phase = 'recovery'
      checklist = [False, False, False]           

    return frame, phase, checklist

if __name__ == "__main__":

    calibration_start_time = None
    previous_hip_x = None
    catch_calibrated = False
    finish_calibrated = False
    catch_x_position = None
    finish_x_position = None
    completed_calibration_time = None
    message = "SIT AT THE CATCH"
    phase = None
    checklist = [False, False, False]
    # catch_range_max = 0.6
    # finish_range_min = 0.7
    cap = cv2.VideoCapture(0)

    # Force the webcam to use a higher resolution (change values as needed)
    cam_width = 1280  # Try 1920 for Full HD
    cam_height = 720   # Try 1080 for Full HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    calibration_in_process = True
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        if not (catch_calibrated and finish_calibrated) or calibration_in_process:
            frame, calibration_start_time, previous_hip_x, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position, completed_calibration_time, message, calibration_in_process = seat_calibration(
            frame, calibration_start_time, previous_hip_x, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position, completed_calibration_time, message, calibration_in_process
        )
        
        else:
            stroke_length = finish_x_position - catch_x_position
            frame, phase, checklist = process_frame(frame, (catch_x_position + stroke_length*.2), (finish_x_position-stroke_length*.2), phase, checklist)
            
        
        # Display the output
        cv2.imshow('Pose Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
