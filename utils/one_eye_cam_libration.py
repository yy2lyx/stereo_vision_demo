import cv2
import numpy as np
import glob
import json


class OneEyeCamLibration:
    def __init__(self,pattern_path,save_path,w,h):
        self.pattern_path = pattern_path
        self.save_path = save_path
        self.w = w
        self.h = h

    @staticmethod
    def readfromfile(path):
        with open(path, 'r') as f:
            js = f.read()
            mydict = json.loads(js)
        return mydict

    @staticmethod
    def writetofile(dict, path):
        for index, item in enumerate(dict):
            print(index, ":", item, ':', type(dict[item]), ":", dict[item], ' ok\n')
            dict[item] = np.array(dict[item])
            dict[item] = dict[item].tolist()
        js = json.dumps(dict)
        with open(path, 'w') as f:
            f.write(js)
            print("One Eye Cam Hyperparameter has been saved to files!")

    def libration(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.h * self.w, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.w, 0:self.h].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        imglist = glob.glob(f'{self.pattern_path}/*.jpg')
        print(f"Total Images : {len(imglist)}")
        for i, fname in enumerate(imglist):
            img = cv2.imread(imglist[i])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.w, self.h), None)
            if ret:
                objpoints.append(objp)
                # 提取亚像素焦点
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                # 对角点连接画线加以展示
                cv2.drawChessboardCorners(img, (self.w, self.h), corners2, ret)
                cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                cv2.imshow('img', img)
                cv2.waitKey(0)
            print(i)
        cv2.destroyAllWindows()
        #  保存结果到文件
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        dict = {'ret': ret, 'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
        self.writetofile(dict, self.save_path)










