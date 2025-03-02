import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detect_pose import process_frame
from seat_calibration2 import seat_calibration

def main():
    calibration_start_time = None
    previous_hip_x = None
    catch_calibrated = False
    finish_calibrated = False
    catch_x_position = None
    finish_x_position = None
    completed_calibration_time = None
    message = "SIT AT THE CATCH"
    cap = cv2.VideoCapture(0)

    cam_width = 1280  # Try 1920 for Full HD
    cam_height = 720   # Try 1080 for Full HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    st.title("Posture Detection")
    st.write("Press 'Stop' to exit the video stream.")

    stframe = st.empty()

    stop_button = st.button("Stop")
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
            frame = process_frame(frame, (catch_x_position + stroke_length*.2), (finish_x_position-stroke_length*.2))
        

        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        stframe.image(img, channels="RGB")
        #cv2.imshow('Pose Detection', frame)
        # Exit if 'Stop' button is pressed
        if stop_button:
            break

    cap.release()
    cv2.destroyAllWindows()
    st.stop()

if __name__ == "__main__":
    main()