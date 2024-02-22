import numpy as np
from typing import Tuple

import cv2
from tqdm import tqdm
import mediapipe as mp


def get_frames(path):
    """
    Extract frames from a video file.

    Args:
    - path (str): The file path to the video.

    Returns:
    - frames (list): A list of frames extracted from the video.
    """
    frames = []
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames) as pbar:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                pbar.update(1)
                frames.append(frame)
            else:
                break
    cap.release()
    return frames, fps


def regularize(data, threshold: float) -> Tuple[int, int]:
    """
    Regularize the data based on a threshold.

    Args:
        data (list): The list of data points to be regularized.
        threshold (float): The threshold value used to determine the range of valid data.

    Returns:
        Tuple[float, float]: A tuple containing the minimum and maximum values after regularization.

    This function sorts the input data and calculates the minimum and maximum values based on a threshold percentage
    of the data length, discarding extreme values.

    """
    data.sort()
    maxlen = len(data)
    alpha = (np.mean(data) - min(data)) / (max(data) - min(data))
    beta = 1 - alpha

    t1 = int(threshold * maxlen * alpha)
    t2 = int(threshold * maxlen * beta)

    minimum = data[t1]
    maximum = data[maxlen-t2-1]
    return minimum, maximum


def find_crop_area(path: str, sampling_interval: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Find the area to crop in the video based on hand landmarks.

    Args:
        path (str): The path to the input video file.
        sampling_interval (int): The interval for sampling frames in the video.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: A tuple containing the coordinates of the top-left and bottom-right points
            defining the cropping area.

    This function processes the input video frame by frame, detects hand landmarks using the MediaPipe library, and
    determines the area to crop based on the detected hand landmarks.

    """
    print(f"STEP1. Find Video Crop Area")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x_data = []
    y_data = []

    frame_count = 0
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)
            frame_count += 1
            if frame_count % sampling_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                                x_data.append(int(lm.x * width))
                                y_data.append(int(lm.y * height))
    
    # Crop Algorithm with Regularization
    x1, x2 = regularize(x_data, 0.03)
    _, y2 = regularize(y_data, 0.05)

    # Representing rectangle area by two points
    left_top = (x1, 0)
    right_bottom = (x2, y2)
    return left_top, right_bottom


def get_crop_frames(path: str, left_top: tuple, right_bottom: tuple, save=True) -> list:
    """
    Crop the input video to the specified area.

    Args:
        path (str): The path to the input video file.
        left_top (tuple): The coordinates of the top-left point of the cropping area.
        right_bottom (tuple): The coordinates of the bottom-right point of the cropping area.

    Returns:
        None

    This function crops the input video to the specified area defined by the top-left and bottom-right points,
    and saves the cropped video as a new file.

    """
    # Load an origin video
    cap = cv2.VideoCapture(path)
    x1, y1 = left_top
    x2, y2 = right_bottom
    
    # For concatenate
    cropped_frames = []

    # Cropped video data setting
    cropped_width  = x2 - x1
    cropped_height = y2 - y1
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer
    if save:
        output_video_path = '.' + path.split('.')[1] + '_cropped.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (cropped_width, cropped_height))

    # Cropping process
    msg = f"STEP2. Video Cropping : {frame_width} x {frame_height} -> {cropped_width} x {cropped_height}"
    print(msg)
    with tqdm(total=total_frames) as pbar:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                pbar.update(1)
                cropped_frame = frame[y1:y2, x1:x2]
                cropped_frames.append(cropped_frame)
                if save:
                    out.write(cropped_frame)
            else:
                break
    cap.release()
    if save:
        out.release()
        print(f"Cropped video saved at {output_video_path}")
    return cropped_frames, fps