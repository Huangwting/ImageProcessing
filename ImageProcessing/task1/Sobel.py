import cv2
import numpy as np
from matplotlib import pyplot as plt

#读取图像
image = cv2.imread('f:\\R.jpg', cv2.IMREAD_GRAYSCALE)

#使用Sobel算子进行边缘检测
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

#计算梯度的幅值
sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)

#将图像灰度值反转
inverted_sobel_edges = cv2.bitwise_not(np.uint8(sobel_edges))

#显示原始图像和反转后的边缘检测结果
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
plt.title('原始图片'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(inverted_sobel_edges, cmap='gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])

plt.show()
