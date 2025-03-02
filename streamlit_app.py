import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detect_pose import process_frame
from seat_calibration2 import seat_calibration

def main():
    st.title("Welcome to MyErgBuddy")
    st.header("Purpose")
    st.write("""
    Our app uses computer vision to track joint angles and key landmarks, providing real-time 
             feedback to improve rowing form, boost efficiency, and reduce injury risk—expert 
             coaching anytime, anywhere.
    """)
    rowing_tracked = False
    if "calibration_started" not in st.session_state:
        st.session_state.calibration_started = False

    if st.button("Calibrate and Start", key="start_button"):
        st.session_state.calibration_started = True

    if st.session_state.calibration_started:
        calibration_start_time = None
        previous_hip_x = None
        catch_calibrated = False
        finish_calibrated = False
        catch_x_position = None
        finish_x_position = None
        completed_calibration_time = None
        message = "SIT AT THE CATCH"
        calibration_in_process = True

        cap = cv2.VideoCapture(0)

        cam_width = 1280  # Try 1920 for Full HD
        cam_height = 720   # Try 1080 for Full HD
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

        stframe = st.empty()
        stop_button = st.button("Stop", key="stop_button")

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
                frame = process_frame(frame, (catch_x_position + stroke_length * 0.2), (finish_x_position - stroke_length * 0.2))

            # Convert the frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            stframe.image(img, channels="RGB")

            # Exit if 'Stop' button is pressed
            if stop_button:
                rowing_tracked = True
                st.session_state.calibration_started = False
                break

        cap.release()
        cv2.destroyAllWindows()

    st.header("Form Summary")
    if not rowing_tracked:
        st.write("Please complete calibration and row to get your feedback.")
    else: 
        st.write("Here is your form summary:")
        st.write("Placeholder for form summary")
        # Placeholder for form summary
        percent_correct_catch = 85  # Example value, replace with actual calculation
        percent_correct_finish = 90  # Example value, replace with actual calculation
        
        # Placeholder for common mistakes, will be specified from summary report
        common_mistakes = ["Leaning too far back at the finish", "Not engaging core properly", "Incorrect hand positioning"]

        st.write(f"Percent Correct Catch: {percent_correct_catch}%")
        st.write(f"Percent Correct Finish: {percent_correct_finish}%")
        st.write("Most Common Mistakes:")
        for mistake in common_mistakes:
            st.write(f"- {mistake}")
        

    st.header("What is the correct form?")
    st.write("""
    Don't know where to start? Here are some tips and images to help you get started with the correct form:
    """)
    col1, col2 = st.columns(2)

    with col1:
        st.image("data/catch.jpg", caption="Catch Position")

    with col2:
        st.image("data/finish.jpg", caption="Finish Position")

    st.header("Helpful Rowing Tips")
    tips = [
        "Maintain a straight back throughout the stroke.",
        "Engage your core to support your lower back.",
        "Drive with your legs first, then lean back and pull with your arms.",
        "Keep your grip relaxed to avoid unnecessary tension.",
        "Ensure smooth and controlled movements to maximize efficiency.",
        "Focus on consistent breathing patterns."
    ]
    for tip in tips:
        st.write(f"- {tip}")

    if st.button("Learn How to Row", key="learn_button"):
        st.write("This is a placeholder button where we can implement a follow along tutorial for how to row.")
    st.stop()

if __name__ == "__main__":
    main()