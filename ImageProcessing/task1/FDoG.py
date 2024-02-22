import cv2
import numpy as np
from matplotlib import pyplot as plt

#读取图像
image = cv2.imread('f:\\R.jpg', cv2.IMREAD_GRAYSCALE)

#将图像转换为浮点数类型
image_float = image.astype(float)

#使用高斯滤波进行平滑处理
sigma = 1.0
smoothed_image = cv2.GaussianBlur(image_float, (0, 0), sigma)

#计算图像的梯度
gradient_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)

#计算梯度的幅值
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

#计算梯度的方向
gradient_direction = np.arctan2(gradient_y, gradient_x)

#FDoG算法
alpha = 0.5
beta = 0.2
fdog_edges = np.exp(-(gradient_magnitude**alpha) / (beta * np.max(gradient_magnitude)))

#将图像灰度值拉伸到0-255范围
fdog_edges = (fdog_edges * 255).astype(np.uint8)

#将图像灰度值反转
inverted_fdog_edges = cv2.bitwise_not(np.uint8(fdog_edges * 255))

#显示原始图像和反转后的边缘检测结果
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
plt.title('原始图片'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(inverted_fdog_edges, cmap='gray')
plt.title('FDoG'), plt.xticks([]), plt.yticks([])

plt.show()
