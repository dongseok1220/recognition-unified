import math
import numpy as np


def normal_vector(landmark) -> np.array:
    """
    [params]
    hand_landmarks.landmark
    
    [return]
    normal_vector : The vector of 3-dimensional Cartesian coordinate system 
                    implicits the unit normal vector of plane containing a, b, c
    """
    p0 = landmark[0]
    p1 = landmark[5]
    p2 = landmark[17]

    a = np.array((p0.x, p0.y, p0.z))
    b = np.array((p1.x, p1.y, p1.z))
    c = np.array((p2.x, p2.y, p2.z))

    ab = np.array(b) - np.array(a)
    ac = np.array(c) - np.array(a)
    normal_vector = np.cross(ab, ac)

    magnitude = np.linalg.norm(normal_vector)
    if magnitude != 0:
        normal_vector = normal_vector / magnitude
    
    return normal_vector


def included_angle(vector1, vector2):
    unit_vector1 = np.array(vector1) / np.linalg.norm(vector1)
    unit_vector2 = np.array(vector2) / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    included_angle = np.arccos(dot_product) * 180 / np.pi
    return included_angle


def distance(left, right):
    x1, y1, z1 = left.x, left.y, left.z
    x2, y2, z2 = right.x, right.y, right.z
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)