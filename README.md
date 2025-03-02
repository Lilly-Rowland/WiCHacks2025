# WiCHacks2025

## Authors
**Claire O’Farrel, Lilly Rowland, Aparnaa Senthilnathan**

## Demo Video
[Watch the demo video](https://rit.zoom.us/clips/share/A2F3MRZPVzdPZTJLSFQ3NlBUd2VETnJxYzJRAQ)

## Inspiration
Proper rowing form is essential for efficiency and injury prevention, but expert coaching isn't always accessible. We wanted to create a tool that provides real-time feedback, helping users refine their technique.

## What it does
MyErgBuddy uses computer vision to analyze a rower’s posture, tracking joint angles and key landmarks to assess form. It provides real-time corrections and insights, notifying users when their form is incorrect and how to fix it.

## How we built it
We used a pose estimation model to detect key body landmarks, applying machine learning algorithms to evaluate form accuracy. The system runs on a user-friendly application for accessibility. It measures joint angles and horizontal positions to determine correct posture during different rowing phases. Additionally, it calibrates based on the user's height and erg position, ensuring usability for individuals of all sizes and any erg setup.

## Challenges we ran into
Tuning the algorithm to accurately detect optimal rowing angles and positions was challenging. We had to analyze our own form to verify accuracy. Ensuring a user-friendly experience with clear, timely directions—such as signaling when to ‘catch’ or ‘finish’—was another hurdle. We also worked on refining movement timing to maintain the correct sequence of rowing motions.

## Accomplishments that we're proud of
We successfully developed a working prototype that accurately detects form errors and provides actionable feedback. Our model runs efficiently, making it accessible for everyday users. Currently, it assesses finish and catch positions, offering valuable feedback. We're proud that we created a functional application that aligns with our vision and demonstrates its real-world usability.

## What we learned
We gained experience optimizing pose estimation models, refining real-time feedback systems, and enhancing user interaction with AI-powered applications. Additionally, we deepened our understanding of rowing mechanics to apply that to our project.

## What's next for MyErgBuddy
We have many ideas to improve the project. We plan to enhance accuracy and provide better timing guidance. A key feature we want to add is voice feedback to notify users when their form is incorrect. We also aim to analyze rowing sessions over time, offering insights on the percentage of correct vs. incorrect form and identifying specific mistakes. Future updates could include built-in rowing tips and a “Learn to Row” mode, guiding users through proper form in real time. We also plan to refine the UI for an even better user experience.
