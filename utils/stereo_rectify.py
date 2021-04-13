import cv2
import numpy as np
import glob


# 根据标定数据，校正图片
def undistortimgs(imgspath, mtx, dist):  # 配置路径，图片路径
    # 获取目录下所有匹配文件
    pathlist = glob.glob(imgspath)
    imglist = []
    for i, fname in enumerate(pathlist):
        img = cv2.imread(pathlist[i])
        h_1, w_1 = img.shape[:2]
        # retrieve only sensible pixels alpha=0 ,越接近0，，越小
        # keep all the original image pixels if there is valuable information in the corners alpha=1
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_1, h_1), 1)
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w_2, h_2 = roi
        dst = dst[y:y + h_2, x:x + w_2]  # 裁剪
        imglist.append(dst)
        # imglist.append(img)
    n = len(imglist)
    print("畸变校正完毕\n总共", n, "张图片")
    return imglist, n, h_2, w_2  # 返回校正好的图片

# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output

def epipolar_geometric(img1, img2, K):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    # orb = cv2.xfeatures2d.SIFT_create()
    # 获取角点及描述符
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    print("角点数量：", len(kp1), len(kp2))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    print("匹配点对数量：", len(matches))
    good = []
    pts1 = []
    pts2 = []
    # 筛选匹配点对
    for i, m in enumerate(matches):
        # if m.distance < 0.8 * n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        # print(i, kp2[m.trainIdx].pt, kp2[m.trainIdx].pt, "\n")
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    #  !!!特征点匹配精度有待提高
    # 计算基本矩阵
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    E, mask = cv2.findEssentialMat(pts1, pts2, K)
    # E = np.transpose(K, (1, 0)) @ F @ K
    ret, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return F, R, t

# 极线校正(立体校正)
def getRectifyTransform(height, width, mtx, dist, R, t):
    height = int(height)
    width = int(width)
    # alpha越接近0,图片越小
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx, dist, mtx, dist, (width, height),
                                                      R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.7)
    # initUndistortRectifyMap同时考虑畸变和对极几何
    dist = dist / 10;
    map1x, map1y = cv2.initUndistortRectifyMap(mtx, dist, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx, dist, R2, P2, (width, height), cv2.CV_32FC1)
    # 返回map数组为图像坐标的一一对应映射
    return map1x, map1y, map2x, map2y, Q

# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    #                         原图，映射函数，插值方式
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT)
    return rectifyed_img1, rectifyed_img2