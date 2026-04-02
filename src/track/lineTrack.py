import cv2
import numpy as np
import math
from src.common.utils import clamp  # 导入我们在 utils.py 中定义的工具函数

class LineTracker:
    def extract_arrows(self, src):
        """
        提取赛道中心点
        返回: (lane_points, debug_mask)
        """
        # ROI 裁剪提速 (若需要可解开注释)
        # roi_offset_y = int(src.shape[0] * 0.6)
        # src_roi = src[roi_offset_y:, :]
        
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

        # HSV 阈值
        lower_blue = np.array([72, 30, 80])
        upper_blue = np.array([110, 230, 230])

        debug_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        debug_mask = cv2.morphologyEx(debug_mask, cv2.MORPH_OPEN, kernel)
        debug_mask = cv2.morphologyEx(debug_mask, cv2.MORPH_CLOSE, kernel)

        # 第二步：初始化“惯性”滑窗参数
        nwindows = 10                       # 滑窗的数量
        window_height = 25                  # 每个滑窗的高度
        margin = debug_mask.shape[1] // 3   # 滑窗的半宽
        minpix = 25                         # 窗口内最少需要多少个白点才算有效

        # 寻找起始点：统计图像下半部分的像素直方图
        h, w = debug_mask.shape
        bottom_half = debug_mask[h - h // 8:h, :]
        
        histogram = np.sum(bottom_half, axis=0)
        current_x = int(np.argmax(histogram)) # 滑窗初始的X坐标

        # 如果底部一点蓝色都没有，给个默认值（画面中间）
        if histogram[current_x] == 0:
            current_x = w // 2

        # “惯性”参数初始化
        dx = 0                # 记录上一帧的横向位移 (斜率趋势)
        consecutive_lost = 0  # 连续丢失赛道的窗口数
        lane_points =[]      # 存储提取出的赛道中心点

        # 第三步：带有惯性预测的向上滑窗寻线
        for w_idx in range(nwindows):
            win_y_low = clamp(h - (w_idx + 1) * window_height - 30, 0, h - 1)
            win_y_high = clamp(h - w_idx * window_height - 30, 0, h - 1)
            win_x_low = clamp(current_x - margin, 0, w - 1)
            win_x_high = clamp(current_x + margin, 0, w - 1)

            # 提取当前窗口内的图像 ROI
            window_roi = debug_mask[win_y_low:win_y_high, win_x_low:win_x_high]

            # 查找窗口内非零像素 (白点)
            nonzero = cv2.findNonZero(window_roi)
            found_pixels = len(nonzero) if nonzero is not None else 0

            if found_pixels > minpix:
                # 情况 A：找到足够的赛道像素，没有被遮挡
                mean_x = int(np.mean(nonzero[:, 0, 0])) + win_x_low
                
                dx = mean_x - current_x
                current_x = mean_x
                consecutive_lost = 0
                
                lane_points.append((current_x, win_y_low + window_height // 2))
            else:
                break
                # # 情况 B：像素不够，启动“惯性”预测
                # current_x = current_x + dx
                # consecutive_lost += 1

                # if consecutive_lost > 1:
                #     break # 直接跳出循环，停止向上滑窗
                
                # pred_x = clamp(current_x, 0, w - 1)
                # lane_points.append((pred_x, win_y_low + window_height // 2))

        return lane_points, debug_mask
