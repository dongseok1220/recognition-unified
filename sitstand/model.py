from .tracking import get_center_y
from .detect_action import detect_action 
from .get_frames import get_action_frames

minmax_threshold = 0.2

def model(frames, fps):
   
    # STEP1: center_y extraction from video: tracking.py로 center_y(t)를 얻습니다.
    center_y = get_center_y(frames)
    # STEP2: center_y(t)에 sliding window를 통해 action(일어서기, 앉기)이 언제 있었는지 detect한다. Window의 max[]-min[] > minmax_threshold 이면 TRUE를 return한다.
    action_detected = detect_action(center_y, fps, minmax_threshold)  # Call the function
    # STEP3: action이 detect된 frame들에게 1를 return한다. 각 프레임은 1 or 0 값을 갖는다.
    action_frames=get_action_frames(action_detected, fps)
    # action_frames에서 값이 1인 인덱스를 추출하는 코드
    indices_of_ones = [index for index, value in enumerate(action_frames) if value == 1]

    return indices_of_ones