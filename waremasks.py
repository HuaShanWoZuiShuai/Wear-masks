import numpy as np
from PIL import Image
import cv2
import dlib
import sys

# 计算透视变换系数
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def WearMask(photopath):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('libs/shape_predictor_68_face_landmarks.dat')

    # cv2读取图像
    img = cv2.imread(photopath)
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # pil读取图像
    people = Image.open(photopath)
    width, height = people.size
    # 读取口罩图片
    mask = Image.open('res/mask.png')
    tmask = Image.new(mode='RGBA', size=(width, height))
    tmask.paste(mask, (0, 0), mask)
    
    # 人脸数rects
    rects = detector(img_gray)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
        face = landmarks.tolist()
        #计算透视变换系数
        coeffs = find_coeffs(
        [face[27],face[3],face[8],face[13]],
        [[176, 45], [33, 174], [179, 308], [313, 174]]
        )
        #透视变换mask图像
        mask = tmask.transform((width, height), Image.PERSPECTIVE, coeffs,Image.BICUBIC)
        #粘贴
        people.paste(mask, (0, 0), mask)
    
    return people

if __name__ == '__main__':
    photo = WearMask(sys.argv[1])
    #photo.show()
    photo.save(sys.argv[2])