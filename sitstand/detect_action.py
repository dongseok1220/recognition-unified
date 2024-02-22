def detect_action(center_y_list, fps, threshold):
    # Initializations
    window_size = fps  # window size = 1 second
    action_detected = []
    step_size = int(window_size)  # calculate step size based on overlap ratio

    for i in range(0, len(center_y_list) - window_size + 1, step_size):
        window = [y for y in center_y_list[i : i+window_size] if y is not None]  # Exclude None values

        # Check if all values in the window are None
        if all(y is None for y in window):
            action_detected.append(0)
        else:
            # Exclude None values for calculating the difference
            valid_window = [y for y in window if y is not None]
            max_min = max(valid_window) - min(valid_window)

            # Check if there's a significant difference in y-coordinates
            if max_min > threshold:
                action_detected.append(1)
            else:
                action_detected.append(0)

    return action_detected