import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detect_pose import process_frame

def main():
    st.title("Posture Detection")
    st.write("Press 'Stop' to exit the video stream.")

    catch_range_max = 0.6
    finish_range_min = 0.8

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    stop_button = st.button("Stop")

    while cap.isOpened():
        
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame, catch_range_max, finish_range_min)

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