# -*- coding: utf-8 -*-
"""
视差转点云
"""
import numpy as np


def disparity_to_point_cloud(disparity: np.array, baseline, f, cx, cy):
    points = []
    rows, columns = disparity.shape
    for y in range(rows):
        for x in range(columns):
            if disparity[y, x] != 0:
                Z = baseline * f / disparity[y, x]
                X = Z * (x - cx) / f
                Y = Z * (y - cy) / f
                points.append([X, Y, Z])
    return points
