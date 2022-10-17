# -*- coding: utf-8 -*-
"""
格雷码图案生成
"""
import os
import numpy as np
import cv2 as cv

current_pattern = 0


def gray_code_pattern_gen(width, height):
    cols, rows = width, height
    vbits, hbits = 1, 1
    # 计算需要生成多少图案
    i = 1 << vbits
    while i < cols:
        vbits += 1
        i = 1 << vbits
    i = 1 << hbits
    while i < rows:
        hbits += 1
        i = 1 << hbits
    pattern_count = max(vbits, hbits)
    print('vbits: {}, hbits: {}, 需要生成 {} 张图案'.format(vbits, hbits, 4 * pattern_count + 2))
    voffset = int(((1 << vbits) - cols) / 2)
    hoffset = int(((1 << hbits) - rows) / 2)
    print('voffset: {}, hoffset: {}'.format(voffset, hoffset))

    global current_pattern
    while current_pattern < 4 * pattern_count + 2:
        make_pattern(width, height, vbits, hbits, pattern_count)
        current_pattern += 1


def make_pattern(width, height, vbits, hbits, pattern_count):
    cols, rows = width, height

    vmask, voffset = 0, int(((1 << vbits) - cols) / 2)
    hmask, hoffset = 0, int(((1 << hbits) - rows) / 2)
    inverted = (current_pattern % 2) == 0

    # patterns
    # -----------
    # 00 white
    # 01 black
    # -----------
    # 02 vertical, bit N-0, normal
    # 03 vertical, bit N-0, inverted
    # 04 vertical, bit N-1, normal
    # 04 vertical, bit N-2, inverted
    # ..
    # XX =  (2*_pattern_count + 2) - 2 vertical, bit N, normal
    # XX =  (2*_pattern_count + 2) - 1 vertical, bit N, inverted
    # -----------
    # 2+N+00 = 2*(_pattern_count + 2) horizontal, bit N-0, normal
    # 2+N+01 horizontal, bit N-0, inverted
    # ..
    # YY =  (4*_pattern_count + 2) - 2 horizontal, bit N, normal
    # YY =  (4*_pattern_count + 2) - 1 horizontal, bit N, inverted

    if current_pattern < 2:
        # white or black
        do_make_pattern(rows, cols, vmask, voffset, hmask, hoffset, inverted)
    elif current_pattern < 2 * pattern_count + 2:
        # vertical
        bit = vbits - int(current_pattern / 2)
        vmask = 1 << bit
        print(f'v# cp: {current_pattern + 1}, bit: {bit}, mask: {vmask}, inverted: {inverted}')
        do_make_pattern(rows, cols, vmask, voffset, hmask, hoffset, not inverted)
    elif current_pattern < 4 * pattern_count + 2:
        # horizontal
        bit = hbits + pattern_count - int(current_pattern / 2)
        hmask = 1 << bit
        print(f'h# cp: {current_pattern + 1}, bit: {bit}, mask: {hmask}, inverted: {inverted}')
        do_make_pattern(rows, cols, vmask, voffset, hmask, hoffset, not inverted)


def do_make_pattern(rows, cols, vmask, voffset, hmask, hoffset, inverted):
    tvalue = 0 if inverted else 255
    fvalue = 255 if inverted else 0

    pattern = np.zeros((rows, cols), dtype=np.uint8)
    for h in range(rows):
        for w in range(cols):
            bit = (binary_to_gray(h + hoffset) & hmask) + (binary_to_gray(w + voffset) & vmask)
            value = tvalue if bit > 0 else fvalue
            pattern[h, w] = value
    cv.imwrite(os.path.join('./pattern', f'{(current_pattern + 1):02}.png'), pattern)


def binary_to_gray(num):
    """
    二进制转格雷码
    """
    return int((num >> 1) ^ num)


if __name__ == '__main__':
    gray_code_pattern_gen(3840, 2160)
