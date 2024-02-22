import cv2
import mediapipe as mp
from tqdm import tqdm
import time


def get_center_y(frames):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    center_y_list = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        time.sleep(0.5)
        for image in tqdm(frames):
            # Pose estimation
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                # Calculate the center y-coordinate
                left_shoulder = results.pose_landmarks.landmark[11].y
                right_shoulder = results.pose_landmarks.landmark[12].y

                center_y = (left_shoulder + right_shoulder) / 2
                center_y_list.append(center_y)
            else:
                center_y_list.append(None)  # Add None if no landmarks are detected

    return center_y_list
