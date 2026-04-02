from enum import Enum

class CarFlag(Enum):
    CAR_NONE = 0
    CAR_AVOID = 1
    CAR_LOST = 2

CLASS_ID_CAR = 1

class Car:
    def __init__(self, config=None):
        self.flag_car = CarFlag.CAR_NONE

    def check_car(self, predict_result: list):
        is_car_detected = False

        # 1. 遍历检测结果，寻找车辆
        for target in predict_result:
            if getattr(target, 'class_id', -1) == CLASS_ID_CAR:
                is_car_detected = True
                break

        # 2. 状态机流转逻辑
        if is_car_detected:
            # 如果看到了车，直接进入避障/处理状态
            self.flag_car = CarFlag.CAR_AVOID
        else:
            # 如果没看到车
            if self.flag_car == CarFlag.CAR_AVOID:
                # 之前在避障，现在突然看不到了，进入丢失状态
                self.flag_car = CarFlag.CAR_LOST
            elif self.flag_car == CarFlag.CAR_LOST:
                # 连续没看到车，重置状态
                self.flag_car = CarFlag.CAR_NONE

    def run_car(self, src_img, predict_result: list):
        pass
