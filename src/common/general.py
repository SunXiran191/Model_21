import cv2
import numpy as np

class General:
    def __init__(self):
        # 原本为 float 类型的 Point 数组，Python中使用 numpy 的 float32 数组
        dst_points = np.array([
            [123.0, 105.0],
            [195.0, 104.0],
            [102.0, 135.0],
            [211.0, 134.0]
        ], dtype=np.float32)

        src_points = np.array([
            [110.0, 40.0],
            [210.0, 40.0],
            [110.0, 140.0],
            [210.0, 140.0]
        ], dtype=np.float32)

        # 计算透视变换矩阵
        self.rotation = cv2.getPerspectiveTransform(src_points, dst_points)
        
        self.change_un_Mat = np.array([
            [-3.614457831325295, -5.466867469879507, 744.9397590361433],
            [-9.964747281289631e-16, -11.48343373493974, 999.0963855421669],
            [-4.864999171266168e-18, -0.03463855421686741, 1]
        ], dtype=float)

        self.Re_change_un_Mat = np.array([
            [0.5571147540983606, -0.4899672131147541, 74.50754098360656],
            [4.174901165517516e-17, -0.08708196721311479, 87.00327868852459],
            [1.719608593143229e-19, -0.003016393442622951, 1]
        ], dtype=float)
        
        self.counter = 0

    def clip(self, x: int, low: int, up: int) -> int:
        return up if x > up else (low if x < low else x)

    def save_picture(self, image, delta: int = 1, prefix: str = ""):
        self.counter += delta
        print(f"image:{self.counter}")
        img_path = "/home/edgeboard/Run_ACCM_2025Demo/image/"
        name = f"{img_path}{self.counter}{prefix}.jpg"
        cv2.imwrite(name, image)

    def factorial(self, x: int) -> int:
        f = 1
        for i in range(1, x + 1):
            f *= i
        return f

    def bezier(self, dt: float, input_points: list) -> list:
        # Bezier 曲线计算
        output = []
        t = 0.0
        while t <= 1:
            # 实现贝塞尔插值逻辑
            t += dt
        return output

    def sigma(self, vec: list, n: int, m: int) -> float:
        pass

    def filter(self, value: float) -> float:
        # 滑动均值滤波实现
        return value

    def pid_realize_a(self, actual: float, set_val: float, _p: float, _d: float) -> float:
        # 位置式 PID 角度外环
        pass

    def transf(self, i: int, j: int):
        # 逆透视变换矩阵乘法实现等同于 C++ 指针的按值返回
        pass

    def reverse_transf(self, i: int, j: int):
        pass
