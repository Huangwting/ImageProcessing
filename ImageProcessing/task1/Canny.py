import cv2
import numpy as np
from matplotlib import pyplot as plt

#读取图像
image = cv2.imread('f:\\R.jpg', cv2.IMREAD_GRAYSCALE)

#使用Canny算法进行边缘检测
canny_edges = cv2.Canny(image, 50, 150)

#将图像灰度值反转
inverted_canny_edges = cv2.bitwise_not(canny_edges)

#显示原始图像和反转后的边缘检测结果
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
plt.title('原始图片'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(inverted_canny_edges, cmap='gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])

plt.show()
