# -*- coding: utf-8 -*-
"""
主函数，将gray_code_decode.py、gray_code_stereo_match.py、disparity_to_pointcloud.py集成
"""
# -*- coding: utf-8 -*-
"""
author:     tianhe
time  :     2022-10-08 16:13
"""
import os
from threading import Thread
import numpy as np
import cv2 as cv
import open3d as o3d
from gray_code_decode import estimate_direct_light, decode_pattern, colorize_pattern
from gray_code_stereo_match import stereo_match, colorize_disparity
from disparity_to_pointcloud import disparity_to_point_cloud


class GrayCodeDecodeThread(Thread):
    pattern_image_left = None
    pattern_image_right = None

    def __init__(self, img_root_dir, use_n_bit_to_decode):
        super().__init__()
        self.img_root_dir = img_root_dir
        self.use_n_bit_to_decode = use_n_bit_to_decode

    def run(self) -> None:
        root_dir = self.img_root_dir
        print('格雷码结构光解码, root_dir: {}...'.format(root_dir))

        all_image_list = sorted(os.listdir(root_dir))
        left_image_list = [img for img in all_image_list if img.find('left') > -1]
        left_image_list = left_image_list[1:]
        half_pattern_size = int((len(left_image_list) - 2) / 2)
        left_image_list = left_image_list[:2 + 2 * self.use_n_bit_to_decode] + \
                          left_image_list[2 + half_pattern_size:2 + half_pattern_size + 2 * self.use_n_bit_to_decode]
        print('格雷码结构光解码, left_image_list: {}...'.format(left_image_list))

        right_image_list = [img for img in all_image_list if img.find('right') > -1]
        right_image_list = right_image_list[1:]
        half_pattern_size = int((len(right_image_list) - 2) / 2)
        right_image_list = right_image_list[:2 + 2 * self.use_n_bit_to_decode] + \
                           right_image_list[
                           2 + half_pattern_size:2 + half_pattern_size + 2 * self.use_n_bit_to_decode]
        print('格雷码结构光解码, right_image_list: {}...'.format(right_image_list))

        left_image_decode_thread = DoGrayCodeDecodeThread(left_image_list, root_dir, 'left')
        right_image_decode_thread = DoGrayCodeDecodeThread(right_image_list, root_dir, 'right')
        left_image_decode_thread.start()
        right_image_decode_thread.start()

        left_image_decode_thread.join()
        right_image_decode_thread.join()
        print('格雷码结构光解码完成')

        left_image1 = colorize_pattern(self.pattern_image_left, 0)
        left_image2 = colorize_pattern(self.pattern_image_left, 1)
        right_image1 = colorize_pattern(self.pattern_image_right, 0)
        right_image2 = colorize_pattern(self.pattern_image_right, 1)

        result_path = os.path.join(root_dir, 'result')
        print('保存解码结果: {}'.format(result_path))
        os.mkdir(result_path)
        cv.imwrite(os.path.join(result_path, 'left_image1.png'), left_image1)
        cv.imwrite(os.path.join(result_path, 'left_image2.png'), left_image2)
        cv.imwrite(os.path.join(result_path, 'right_image1.png'), right_image1)
        cv.imwrite(os.path.join(result_path, 'right_image2.png'), right_image2)

        np.save(os.path.join(result_path, 'pattern_image_left'), self.pattern_image_left)
        np.save(os.path.join(result_path, 'pattern_image_right'), self.pattern_image_right)

        print('显示解码结果...')
        cv.namedWindow("left image1", cv.WINDOW_NORMAL)
        cv.namedWindow("left image2", cv.WINDOW_NORMAL)
        cv.namedWindow("right image1", cv.WINDOW_NORMAL)
        cv.namedWindow("right image2", cv.WINDOW_NORMAL)
        cv.resizeWindow("left image1", 640, 400)
        cv.resizeWindow("left image2", 640, 400)
        cv.resizeWindow("right image1", 640, 400)
        cv.resizeWindow("right image2", 640, 400)
        cv.imshow("left image1", left_image1)
        cv.imshow("left image2", left_image2)
        cv.imshow("right image1", right_image1)
        cv.imshow("right image2", right_image2)

        print('立体匹配，计算视差...')
        disparity = stereo_match(self.pattern_image_left, self.pattern_image_right)
        np.save(os.path.join(result_path, 'disparity'), disparity)
        disp_color = colorize_disparity(disparity)
        cv.imwrite(os.path.join(result_path, 'disparity.png'), disp_color)
        cv.namedWindow("disparity", cv.WINDOW_NORMAL)
        cv.resizeWindow("disparity", 640, 400)
        cv.imshow("disparity", disp_color)

        print('视差转点云...')
        points = disparity_to_point_cloud(disparity, baseline=49.97, f=640.07, cx=642.22, cy=355.13)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join(result_path, 'point_cloud.ply'), point_cloud, write_ascii=True)

        cv.waitKey(0)


class DoGrayCodeDecodeThread(Thread):
    def __init__(self, image_list, root_dir, model):
        super().__init__()
        self.root_dir = root_dir
        self.image_list = image_list
        self.model = model

    def run(self) -> None:
        direct_light = estimate_direct_light(self.image_list, self.root_dir, b=0.3)
        pattern_image = decode_pattern(self.image_list, self.root_dir,
                                       robust=True, direct_light=direct_light, m=5, threshold=25)
        if self.model == 'left':
            GrayCodeDecodeThread.pattern_image_left = pattern_image
        elif self.model == 'right':
            GrayCodeDecodeThread.pattern_image_right = pattern_image


if __name__ == '__main__':
    img_root_dir = './data/1665223106'  # 格雷码图片路径
    use_n_bit_to_decode = 9  # 使用9位格雷码解码
    gray_code_decode_thread = GrayCodeDecodeThread(img_root_dir, use_n_bit_to_decode)
    gray_code_decode_thread.start()
    gray_code_decode_thread.join()
