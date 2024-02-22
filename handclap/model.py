from . import detector 


def model(frames):
    distances, included_angles = detector.pattern(frames, alpha=0.5, beta=0.7)
    outliers = detector.outliers(distances, included_angles, t_dist=0.2, t_angle=125)
    return outliers