def get_action_frames(action_detected, fps):
    # Initialize an empty list for storing action frames
    action_frames = [0] * len(action_detected) * fps

    # Calculate the step size based on the overlap ratio
    step_size = int(fps)

    # Assign True to the action frames
    for i in range(0, len(action_detected)):
        if action_detected[i]:
            for j in range(i*step_size, i*step_size+fps):
                if j < len(action_frames):  # Make sure not to exceed the length of action_frames
                    action_frames[j] = 1

    return action_frames
