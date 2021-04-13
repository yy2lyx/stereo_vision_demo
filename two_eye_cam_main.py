from two_eye_utils.disparity_pointcloud import stereoMatchSGBM,DepthColor2Cloud,view_cloud
from two_eye_utils.stereo_rectify import getRectifyTransform,rectifyImage,preprocess,draw_line
from two_eye_utils.stereoconfig import StereoCamera

import cv2


if __name__ == '__main__':
    # 读取MiddleBurry数据集的图片
    iml = cv2.imread('data/Bicycle1-perfect/im0.png')  # 左图
    imr = cv2.imread('data/Bicycle1-perfect/im1.png')  # 右图
    height, width = iml.shape[0:2]

    # 读取相机内参和外参
    config = StereoCamera()

    # 立体校正
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
    cv2.imshow("l",iml_rectified)
    cv2.waitKey(0)
    print(Q)

    # 绘制等间距平行线，检查立体校正的效果
    line = draw_line(iml_rectified, imr_rectified)
    cv2.imwrite('data/stereo_rectify.png', line)

    # 立体匹配
    iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以
    disp, _ = stereoMatchSGBM(iml_, imr_, True)  # 这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
    cv2.imwrite('data/parallax.png', disp)

    # # 展示距离
    # cv2.namedWindow('distance_map', cv2.WINDOW_NORMAL)
    # cv2.imshow("distance_map", iml)
    # cv2.setMouseCallback("distance_map", mouse_callback)
    # cv2.waitKey(0)

    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q)  # 可以使用上文的stereo_config.py给出的参数

    # 构建点云--Point_XYZRGBA格式(需要点云数据是N*4，分别表示x,y,z,RGB ,其中RGB 用一个整数表示颜色)
    pointcloud = DepthColor2Cloud(points_3d, iml)

    # 显示点云
    view_cloud(pointcloud)