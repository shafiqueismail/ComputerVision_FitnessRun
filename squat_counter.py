import cv2
import mediapipe as mp
import numpy as np
import time
import os
import signal

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


def find_working_camera(max_cameras=10):
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Using camera index: {i}")
                return cap 
        cap.release()
    print("No working camera found.")
    return None

def squat_detector(frame_queue, squat_queue, stop_event):
    cap = find_working_camera()
    if cap == None:
        frame_queue.put(None)
        return
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    squat_score = 0
    prev_stage = ""
    scored_stage = ""
    transition_threshold = 5 # Frames needed to confirm transition
    frame_count = 0

    start_time = time.time()
    countdown_time = 5  # Reduced to 5 seconds

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # Mirror effect
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(image)

        text_coords = (170, 20)

        elapsed_time = time.time() - start_time
        if not results.pose_landmarks: 
            cv2.putText(image, 'Get Farther From Camera!', text_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)    
        # elif elapsed_time < countdown_time: # count down before allowing squat detector to work
        #     cv2.putText(image, f"Starting in {int(countdown_time - elapsed_time)} sec", text_coords, 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            try:

                landmarks = results.pose_landmarks.landmark
                
                # LEFT
                # Get points for angle calculations
                left_hip_value = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                left_hip = [left_hip_value.x, left_hip_value.y, left_hip_value.z]
                left_knee_value = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                left_knee = [left_knee_value.x, left_knee_value.y, left_knee_value.z]
                left_ankle_value = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                left_ankle = [left_ankle_value.x, left_ankle_value.y, left_ankle_value.z]

                # Calculate knee angle
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                # RIGHT
                # Get points for angle calculations
                right_hip_value = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                right_hip = [right_hip_value.x, right_hip_value.y, right_hip_value.z]
                right_knee_value = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                right_knee = [right_knee_value.x, right_knee_value.y, right_knee_value.z]
                right_ankle_value = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                right_ankle = [right_ankle_value.x, right_ankle_value.y, right_ankle_value.z]

                # Calculate knee angle
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                detected_stage = ""

                # We chose 50 degrees for the squat when playing the game because the model is not always accurate when the user faces the camera 
                if left_knee_angle <= 50 and right_knee_angle <= 50: #the model is not super accurate
                    detected_stage = "Squat"
                
                # Ensure stable transition by confirming over multiple frames
                if detected_stage == prev_stage:
                    frame_count += 1
                else:
                    frame_count = 0  

                if frame_count >= transition_threshold:
                    if detected_stage != scored_stage:
                        if detected_stage == "Squat":
                            squat_score += 1
                            squat_queue.put(True)
                        scored_stage = detected_stage
                    frame_count = 0

                prev_stage = detected_stage

                # Debugging
                # nose_value = landmarks[mp_pose.PoseLandmark.NOSE.value]
                # cv2.putText(image, f'Squat Score: {squat_score}', text_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.putText(image, f'Left Angle {left_knee_angle}', (text_coords[0], text_coords[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.putText(image, f'Right Angle {right_knee_angle}', (text_coords[0], text_coords[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.putText(image, f'Visibility right ankle, left ankle, nose', (text_coords[0], text_coords[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.putText(image, f'{right_ankle_value.visibility}', (text_coords[0], text_coords[1] + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.putText(image, f'{left_ankle_value.visibility}', (text_coords[0], text_coords[1] + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.putText(image, f'{nose_value.visibility}', (text_coords[0], text_coords[1] + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            except Exception as e:
                print("Error:", e)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        frame_queue.put(image)  # Send frame to Pygame

    cap.release()
    os.kill(os.getpid(), signal.SIGTERM) # for some reason open cv doesnt close unless you force it