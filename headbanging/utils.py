import numpy as np
from .config import direction_mapping

def detect_primary_movement_direction(normal_vector_prev, normal_vector_current):    
    # 벡터 변화량 계산
    vector_change = normal_vector_current - normal_vector_prev

    # 각 방향에 대한 변화량 계산
    horizontal_change = vector_change[0]
    vertical_change = vector_change[1]

    # 변화량이 가장 큰 방향을 결정
    if abs(horizontal_change) > abs(vertical_change):
        # 수평 방향 변화가 더 큼
        if horizontal_change > 0:
            primary_direction = "right"
        else:
            primary_direction = "left"
    else:
        # 수직 방향 변화가 더 큼
        if vertical_change > 0:
            primary_direction = "up"
        else:
            primary_direction = "down"
    
    # 변화량이 없는 경우
    if vector_change[0] == 0 and vector_change[1] == 0:
        primary_direction = "no_change"

    # 사전에서 매핑된 값을 반환
    return direction_mapping[primary_direction]

def calculate_angle_change(normal_vector_prev, normal_vector_current):
    unit_vector_prev = normal_vector_prev 
    unit_vector_current = normal_vector_current
    dot_product = np.dot(unit_vector_prev, unit_vector_current)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)