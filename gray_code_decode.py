# -*- coding: utf-8 -*-
"""
格雷码解码，解码方法参考：
    - https://zhuanlan.zhihu.com/p/110801836
    - http://mesh.brown.edu/calibration/
    - http://mesh.brown.edu/calibration/files/Simple,%20Accurate,%20and%20Robust%20Projector-Camera%20Calibration.pdf
"""
import os
import cv2 as cv
import numpy as np


def gray_to_binary(num, num_bits=32):
    """
    格雷码转二进制
    :param num:
    :param num_bits: 格雷码位数
    :return:
    """
    shift = 1
    while shift < num_bits:
        num ^= num >> shift
        shift <<= 1
    return int(num)


def estimate_direct_light(image_list, root_dir, b=0.3):
    """
    估计图像中灰度值由投影仪贡献的直接分量，
    需要保证至少有38张图像，采用第15,16,17,18,35,36,37,38估计
    :param image_list: image filename list
    :param root_dir: 图像根目录
    :param b: 环境光的比例，默认0.3，那么投影光的比例为0.7
    :return:
    """
    total_images = len(image_list)  # 42
    total_patterns = int(total_images / 2 - 1)  # 42/2-1 = 20
    direct_light_count = 4
    direct_light_offset = 4
    if total_patterns < direct_light_count + direct_light_offset:
        print('too few pattern images to estimate direct light')
        return None

    direct_component_images = []  # 15 << 16 << 17 << 18 << 35 << 36 << 37 << 38
    for i in range(direct_light_count):
        index = total_images - total_patterns - direct_light_count - direct_light_offset + i + 1  # 42-20-4-4+0+1=15
        direct_component_images.append(index)
        direct_component_images.append(index + total_patterns)

    images = []
    for i in direct_component_images:
        images.append(cv.imread(os.path.join(root_dir, image_list[i - 1]), cv.IMREAD_GRAYSCALE))

    count = len(images)
    if count < 1:
        # no images
        print('no images to estimate direct light')
        return None
    for i in range(count):
        if len(images[i].shape) != 2:
            print('gray images required to estimate direct light')
            return None
    # initialize direct light image
    print('estimate direct light begin...')
    rows, columns = images[0].shape
    direct_light = np.zeros((rows, columns, 2), dtype=np.uint8)

    b1 = 1.0 / (1.0 - b)
    b2 = 2.0 / (1.0 - b * 1.0 * b)

    for h in range(rows):
        row_list = []
        for i in range(count):
            row_list.append(images[i][h, :])
        for w in range(columns):
            Lmax = row_list[0][w]
            Lmin = row_list[0][w]
            for i in range(count):
                if Lmax < row_list[i][w]:
                    Lmax = row_list[i][w]
                if Lmin > row_list[i][w]:
                    Lmin = row_list[i][w]
            Ld = int(b1 * (Lmax - Lmin) + 0.5)
            Lg = int(b2 * (Lmin - b * Lmax) + 0.5)
            direct_light[h, w, 0] = Ld if Lg > 0 else Lmax
            direct_light[h, w, 1] = Lg if Lg > 0 else 0

    print('estimate direct light done!')
    return direct_light


def decode_pattern(image_list, root_dir,
                   robust: bool = False, direct_light: np.array = None, m: int = 5, threshold: int = 25):
    init = True

    total_images = len(image_list)  # 42
    total_patterns = int(total_images / 2 - 1)  # 42/2-1 = 20
    total_bits = int(total_patterns / 2)  # 10
    if 2 + 4 * total_bits != total_images:  # 黑白、横向、横向inv、纵向、纵向inv
        print('解码图片数量有误！')
        return False

    bit_count = [0, total_bits, total_bits]  # pattern bits
    group_size = [1, total_bits, total_bits]  # number of image pairs
    COUNT = 2 * (group_size[0] + group_size[1] + group_size[2])  # total image count

    # load every image pair and compute the maximum, minimum, and bit code
    print('decode begin...')
    group, current = 0, 0
    pattern_image, min_max_image = None, None
    for t in range(0, COUNT, 2):
        if current == group_size[group]:
            group += 1
            current = 0

        if group == 0:
            # skip
            current += 1
            continue

        bit = bit_count[group] - current - 1  # current bit: from 0 to (bit_count[set]-1)
        channel = group - 1

        gray_image1 = cv.imread(os.path.join(root_dir, image_list[t + 0]), cv.IMREAD_GRAYSCALE)
        gray_image2 = cv.imread(os.path.join(root_dir, image_list[t + 1]), cv.IMREAD_GRAYSCALE)
        print(f'current: {current}, gray_image1: {image_list[t + 0]}')
        print(f'current: {current}, gray_image2: {image_list[t + 1]}')

        if gray_image1.shape != gray_image2.shape:
            print('Initial images have different size')
            return False
        if robust and gray_image1.shape[0] != direct_light.shape[0] and gray_image1.shape[1] != direct_light.shape[1]:
            print('Direct Component image has different size')
            return False

        rows, columns = gray_image1.shape
        if init:
            pattern_image = np.zeros((rows, columns, 2), dtype=np.float32)
            min_max_image = np.zeros((rows, columns, 2), dtype=np.uint8)

        # compare
        for h in range(rows):
            for w in range(columns):
                value1 = gray_image1[h, w]
                value2 = gray_image2[h, w]

                if init or value1 < min_max_image[h, w, 0] or value2 < min_max_image[h, w, 0]:
                    min_max_image[h, w, 0] = value1 if value1 < value2 else value2
                if init or value1 > min_max_image[h, w, 1] or value2 > min_max_image[h, w, 1]:
                    min_max_image[h, w, 1] = value1 if value1 > value2 else value2

                if not robust:
                    # [simple] pattern bit assignment
                    if value1 > value2:
                        # set bit n to 1
                        pattern_image[h, w, channel] += (1 << bit)
                else:
                    if direct_light is not None and (init or not np.isnan(pattern_image[h, w, channel])):
                        p = get_robust_bit(value1, value2, direct_light[h, w, 0], direct_light[h, w, 1], m)
                        if np.isnan(p):
                            pattern_image[h, w, channel] = np.nan
                        else:
                            pattern_image[h, w, channel] += (p << bit)

        init = False

        current += 1

    # threshold
    rows, columns, _ = pattern_image.shape
    for h in range(rows):
        for w in range(columns):
            if min_max_image[h, w, 1] - min_max_image[h, w, 0] < threshold:
                pattern_image[h, w] = [np.nan, np.nan]

    # gray code to binary
    convert_pattern(pattern_image)

    print('decode done!')
    return pattern_image


def get_robust_bit(value1, value2, Ld, Lg, m):
    """
    二值化
    """
    if Ld < m:
        return np.nan
    if Ld > Lg:
        return 1 if value1 > value2 else 0
    if value1 <= Ld and value2 >= Lg:
        return 0
    if value1 >= Lg and value2 <= Ld:
        return 1
    return np.nan


def convert_pattern(pattern_image):
    """
    将解码后的格雷码图像转为二进制
    :param pattern_image:
    :return:
    """
    rows, columns, channels = pattern_image.shape
    for h in range(rows):
        for w in range(columns):
            if not np.isnan(pattern_image[h, w, 0]):
                p = int(pattern_image[h, w, 0])
                code = gray_to_binary(p)
                if code < 0:
                    code = 0
                pattern_image[h, w, 0] = code + (pattern_image[h, w, 0] - p)
            if not np.isnan(pattern_image[h, w, 1]):
                p = int(pattern_image[h, w, 1])
                code = gray_to_binary(p)
                if code < 0:
                    code = 0
                pattern_image[h, w, 1] = code + (pattern_image[h, w, 1] - p)


def colorize_pattern(pattern_image, group):
    rows, columns, _ = pattern_image.shape
    image = np.zeros((rows, columns, 3), np.uint8)

    max_value, min_value = np.nanmax(pattern_image[:, :, group]), np.nanmin(pattern_image[:, :, group])
    print('group: {}, max_value: {}'.format(group, max_value))
    print('group: {}, min_value: {}'.format(group, min_value))
    n = 4.0
    dt = 255.0 / n

    for h in range(rows):
        for w in range(columns):
            if pattern_image[h, w, group] > max_value or np.isnan(pattern_image[h, w, group]):
                image[h, w] = [0, 0, 0]
                continue
            # display
            t = pattern_image[h, w, group] * 255.0 / max_value
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
