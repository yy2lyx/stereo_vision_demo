from utils.one_eye_cam_libration import OneEyeCamLibration
from utils.stereo_rectify import undistortimgs,epipolar_geometric,draw_line,getRectifyTransform,rectifyImage
import cv2
import numpy as np

if __name__ == '__main__':
    # 1. 单目相机标定
    cam_config_path = 'data/cal.txt'
    one_cam_libration = OneEyeCamLibration('data/pattern', cam_config_path, w = 9, h = 6)
    # one_cam_libration.libration()

    config = one_cam_libration.readfromfile(cam_config_path)
    mtx = np.array(config['mtx'])
    dist = np.array(config['dist'])
    print("K:", mtx, "\ndist:", dist)

    # 2. 畸变校正
    ipath = "my_data/l_r_imgs/*.jpg"
    # 畸变校正
    imglist, sumofimg, height, width = undistortimgs(ipath, mtx, dist)  # 返回校正好的图像序列，高，宽
    img1 = imglist[1]
    img2 = imglist[0]
    F, R, t = epipolar_geometric(img1, img2, mtx)  # 通过特征点计算基础矩阵,并解出R,t
    print("R:", R, "\nt:", t)
    # 计算校正变换
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, mtx, dist, R, t)
    #
    rectifyed_img1, rectifyed_img2 = rectifyImage(img1, img2, map1x, map1y, map2x, map2y)
    # 绘制等间距平行线，检查立体校正的效果
    line = draw_line(rectifyed_img1, rectifyed_img2)
    cv2.namedWindow("two_compare",cv2.WINDOW_NORMAL)
    cv2.imshow("two_compare",line)
    cv2.waitKey(0)