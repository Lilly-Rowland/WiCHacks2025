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
    catch_calibration_time = None
    cap = cv2.VideoCapture(0)
    calibrated = False

    st.title("Posture Detection")
    st.write("Press 'Stop' to exit the video stream.")

    stframe = st.empty()

    stop_button = st.button("Stop")

    while cap.isOpened():
        
        success, frame = cap.read()
        if not success:
            break

        if not (catch_calibrated and finish_calibrated):
            frame, calibration_start_time, previous_hip_x, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position, catch_calibration_time = seat_calibration(
            frame, calibration_start_time, previous_hip_x, catch_calibrated, finish_calibrated, catch_x_position, finish_x_position, catch_calibration_time
            )
        else:
            stroke_length = finish_x_position - catch_x_position
            frame = process_frame(frame, (catch_x_position + stroke_length*.1), (finish_x_position-stroke_length*.1))
       
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