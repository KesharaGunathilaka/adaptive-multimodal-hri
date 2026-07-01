import copy
import itertools
import numpy as np

def calc_landmark_list(image, landmarks):
    """Convert MediaPipe normalized coordinates to pixel coordinates."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    """Perform relative translation to wrist and scale normalization."""
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    
    # Translate
    for index, _ in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
        
    # Flatten
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Scale normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value == 0:
        return temp_landmark_list
    return [n / max_value for n in temp_landmark_list]

def hand_y_norm(lm):
    """Average y of all landmarks. Lower y = hand is higher in the frame."""
    return np.mean([l.y for l in lm])
