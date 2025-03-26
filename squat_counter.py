import cv2
import mediapipe as mp
import numpy as np
import time

class SquatCounter():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    squat_score = 0
    prev_stage = ""
    scored_stage = ""
    transition_threshold = 2  # Frames needed to confirm transition
    frame_count = 0

    start_time = time.time()
    countdown_time = 5  # Reduced to 5 seconds

    def __init__(self, cap):
        self.cap = cap

    
    def get_cv_output(self): # returns image and jump or not
        with SquatCounter.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            is_jump = False
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                return (None, False)

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            elapsed_time = time.time() - SquatCounter.start_time
            if elapsed_time < SquatCounter.countdown_time:
                cv2.putText(image, f"Starting in {int(SquatCounter.countdown_time - elapsed_time)} sec", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get points for angle calculations
                    hip = [landmarks[SquatCounter.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[SquatCounter.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[SquatCounter.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[SquatCounter.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[SquatCounter.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[SquatCounter.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # Calculate knee angle
                    knee_angle = SquatCounter.calculate_angle(hip, knee, ankle)

                    detected_stage = ""

                    # Super Lenient Squat Detection: Knee angle ≤ 150°
                    if knee_angle <= 150:
                        detected_stage = "Squat"
                    
                    # Ensure stable transition by confirming over multiple frames
                    if detected_stage == SquatCounter.prev_stage:
                        SquatCounter.frame_count += 1
                    else:
                        SquatCounter.frame_count = 0  

                    if SquatCounter.frame_count >= SquatCounter.transition_threshold:
                        if detected_stage != SquatCounter.scored_stage:
                            if detected_stage == "Squat":
                                SquatCounter.squat_score += 1
                                is_jump = True
                            SquatCounter.scored_stage = detected_stage
                        SquatCounter.frame_count = 0

                    SquatCounter.prev_stage = detected_stage

                    cv2.putText(image, f'Squat Score: {SquatCounter.squat_score}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                except Exception as e:
                    print("Error:", e)

            SquatCounter.mp_drawing.draw_landmarks(image, results.pose_landmarks, SquatCounter.mp_pose.POSE_CONNECTIONS,
                                        SquatCounter.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        SquatCounter.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return (image, is_jump)

    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array(a)  # First point
        b = np.array(b)  # Middle joint (reference point)
        c = np.array(c)  # Last point

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid NaN errors

        return np.degrees(angle)
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    cv_squat_counter = SquatCounter(cap)

    while cv_squat_counter.cap.isOpened():
        
        image, is_jump =  cv_squat_counter.get_cv_output()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow("Mediapipe Feed", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cv_squat_counter.cap.release()
    cv2.destroyAllWindows()