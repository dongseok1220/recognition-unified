import cv2
import numpy as np
import mediapipe as mp
import time
from .config import *
from .utils import *
from tqdm import tqdm
import time


# Initialize MediaPipe FaceMesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_styles = mp.solutions.drawing_styles

def model(frames):
    normal_vector_prev = None # 이전 법선 벡터
    direction_change_count = 0 # 방향 변화 횟수 
    last_movement_direction = 0 # 마지막 이동 방향 
    last_direction_change_time = time.time() # 방향이 바뀌었을 때 시간 

    detect = [] # headbanging을 탐지했을 때 시간 
    idx = 0 # frames 기준 정답 인덱스 

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        time.sleep(0.5)
        for image in tqdm(frames):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])

                    left_eye = landmarks[133] # 왼쪽 눈 안쪽
                    right_eye = landmarks[362] # 오른쪽 눈 안쪽 
                    left_lip = landmarks[61] # 왼쪽 입술 끝
                    right_lip = landmarks[291] # 오른쪽 입술 끝 
                    
                    center_forehead = (left_eye + right_eye) / 2 # 왼쪽 눈과 오른쪽 눈의 중앙 
                    vector1 = left_lip - center_forehead # 왼쪽 입술 끝과 center_forhead를 잇는 벡터 
                    vector2 = right_lip - center_forehead # 오른쪽 입술 끝과 center_forhead를 잇는 벡터

                    normal_vector = np.cross(vector1, vector2) # 2개의 벡터로 부터 법선벡터 

                    if np.linalg.norm(normal_vector) != 0:
                        normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector) # 정규화 

                        if normal_vector_prev is not None:
                            angle_change = calculate_angle_change(normal_vector_prev, normal_vector_normalized) # 현재 법선벡터와 이전 법선벡터 사이의 각도 계산
                            if angle_change >= angle_threshold: # angle_threshold를 넘은경우 
                                current_time = time.time() # 현재 시간 측정 
                                if current_time - last_direction_change_time > time_interval: # 이전 탐지 시간과 현재 시간차이가 time_interval만큼 날 경우 카운트 초기화 
                                    direction_change_count = 0
                                
                                current_movement_direction = detect_primary_movement_direction(normal_vector_prev, normal_vector_normalized) # 이전 법선벡터를 기준으로 현재 이동 방향 

                                if last_movement_direction != current_movement_direction: # 이전과 다른 방향이면 방향이 전환되었다고 판단 
                                    direction_change_count += 1 # 카운트 증가
                                    last_movement_direction = current_movement_direction # 마지막 이동 방향 업데이트 
                                    last_direction_change_time = current_time # 시간 업데이트 
                                else:
                                    direction_change_count = direction_change_count # 동일하다면 이전 값 유지  

                                if direction_change_count >= headbanging_threshold: # 카운트가 headbanging_threshold 이상이면 headbanging이라고 생각 
                                    detect.append(idx) # 현재 idx 
                                    direction_change_count = 0 # 정답 판단 후 초기화 
                        
                        normal_vector_prev = normal_vector_normalized # 이전 벡터 업데이트 
                    idx += 1 # 다음 인덱스 
    return detect
