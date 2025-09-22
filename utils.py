
import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points: a, b, and c.
    - a, b, c are 2D points represented as (x, y) tuples or lists.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle