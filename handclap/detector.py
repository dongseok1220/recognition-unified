import time

import cv2
from tqdm import tqdm
import mediapipe as mp

from . import metric


class Checker:
    @staticmethod
    def distances(window, t_dist):
        if not len(window):
            return False
        moving_avg = sum(window) / len(window)
        difference = abs(window[-1] - moving_avg)
        return difference > t_dist

    @staticmethod
    def included_angles(window, t_angle):
        if not len(window):
            return False
        max_angle = max(window)
        return max_angle > t_angle
    

def pattern(frames, alpha, beta):
    mp_hands = mp.solutions.hands

    distances = []
    included_angles = []
    with mp_hands.Hands(min_detection_confidence=alpha, min_tracking_confidence=beta) as hands:
        time.sleep(0.5)
        for frame in tqdm(frames):
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                left_hand_landmark_0 = None
                right_hand_landmark_0 = None
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label
                    if handedness == 'Left':
                        left_hand_landmark_0 = hand_landmarks.landmark[0]
                        left_normal_vector = metric.normal_vector(hand_landmarks.landmark)

                    elif handedness == 'Right':
                        right_hand_landmark_0 = hand_landmarks.landmark[0]
                        right_normal_vector = (-1) * metric.normal_vector(hand_landmarks.landmark)

                if left_hand_landmark_0 and right_hand_landmark_0:
                    distance = metric.distance(left_hand_landmark_0, right_hand_landmark_0)
                    distances.append(distance)
                    included_angle = metric.included_angle(left_normal_vector, right_normal_vector)
                    included_angles.append(included_angle)
                
                if not right_hand_landmark_0:
                    distances.append(0)
                    included_angles.append(0)
                
                if not left_hand_landmark_0:
                    distances.append(0)
                    included_angles.append(0)

            else:
                distances.append(0)
                included_angles.append(0)

    return distances, included_angles


def window(start, end, target):
    return target[max(0, start): min(len(target)-1, end)]


def outliers(distances, included_angles, t_dist, t_angle):
    outliers = []
    for i in range(0, len(distances), 10):
        distances_window = window(i, i+30, distances)
        angles_window = window(i-30, i+30, included_angles)
        if Checker.distances(distances_window, t_dist) and Checker.included_angles(angles_window, t_angle):
            outliers.append(i)
    return outliers