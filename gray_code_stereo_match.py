# -*- coding: utf-8 -*-
"""
格雷码双目匹配
"""
from collections import defaultdict
import numpy as np
import cv2 as cv


def stereo_match(pattern_image_left: np.array, pattern_image_right: np.array):
    if not pattern_image_left.shape == pattern_image_right.shape:
        print('左右pattern大小不一致')
    rows, columns, channels = pattern_image_left.shape

    disparity = np.zeros((rows, columns), np.float32)

    code_cam_left_map = defaultdict(list)
    code_cam_right_map = defaultdict(list)

    for h in range(rows):
        for w in range(columns):
            if not np.isnan(pattern_image_left[h, w, 0]) and not np.isnan(pattern_image_left[h, w, 1]):
                index = (int(pattern_image_left[h, w, 0]), int(pattern_image_left[h, w, 1]))
                code_cam_left_map[index].append((w, h))
            if not np.isnan(pattern_image_right[h, w, 0]) and not np.isnan(pattern_image_right[h, w, 1]):
                index = (int(pattern_image_right[h, w, 0]), int(pattern_image_right[h, w, 1]))
                code_cam_right_map[index].append((w, h))

    for key, value in code_cam_left_map.items():
        # print(key, value)
        # print(code_cam_right_map[key])
        match_point = code_cam_right_map[key]
        if len(match_point) > 0:
            sum_x = 0
            for point in match_point:
                sum_x += point[0]
            # print(sum_x)
            for left_point in value:
                disparity[left_point[1], left_point[0]] = left_point[0] - sum_x / len(match_point)
    disparity = cv.medianBlur(disparity, 5)
    return disparity


def colorize_disparity(disparity):
    rows, columns = disparity.shape
    image = np.zeros((rows, columns, 3), np.uint8)

    max_value, min_value = np.max(disparity), np.min(disparity)
    print('disparity max value: {}'.format(max_value))
    print('disparity min value: {}'.format(min_value))
    n = 4.0
    dt = 255.0 / n

    for h in range(rows):
        for w in range(columns):
            if disparity[h, w] == 0:
                image[h, w] = [0, 0, 0]
                continue
            # display
            t = disparity[h, w] * 255.0 / max_value
            c1, c2, c3 = 0.0, 0.0, 0.0
            if t <= 1.0 * dt:
                # black -> red
                c = n * (t - 0.0 * dt)
                c1 = c  # 0-255
                c2 = 0.0  # 0
                c3 = 0.0  # 0
            elif t <= 2.0 * dt:
                # red -> red,green
                c = n * (t - 1.0 * dt)
                c1 = 255.0  # 255
                c2 = c  # 0-255
                c3 = 0.0  # 0
            elif t <= 3.0 * dt:
                # red,green -> green
                c = n * (t - 2.0 * dt)
                c1 = 255.0 - c  # 255-0
                c2 = 255.0  # 255
                c3 = 0.0  # 0
            elif t <= 4.0 * dt:
                # green -> blue
                c = n * (t - 3.0 * dt)
                c1 = 0.0  # 0
                c2 = 255.0 - c  # 255-0
                c3 = c  # 0
            image[h, w] = [int(c3), int(c2), int(c1)]
    return image
