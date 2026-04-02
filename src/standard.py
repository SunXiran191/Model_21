from detection.gold import Gold, GoldFlag
from detection.car import Car, CarFlag
from detection.human import Human, HumanFlag
from detection.light import Light, LightFlag
import cv2

class TrackState:
    TRACK_MIDDLE = 0
    TRACK_AI_MIDDLE = 1

class Scene:
    Normal = 0
    Gold = 1
    Car = 2
    Human = 3
    Light = 4

class SceneStatus:
    def __init__(self):
        self.GoldScene = False
        self.CarScene = False
        self.HumanScene = False
        self.LightScene = False

    def all(self) -> bool:
        return self.GoldScene or self.CarScene or self.HumanScene or self.LightScene

class Standard:
    def __init__(self, config=None):
        self.trackstate = TrackState.TRACK_MIDDLE
        self.scene = Scene.Normal
        self._config = config
        
        self.gold_end_counter = 0
        self.car_end_counter = 0
        self.human_end_counter = 0
        
        self.gold = Gold()
        self.car = Car()
        self.human = Human()
        self.light = Light()

    def run(self, src_img, predict_result: list, pitch_angle: float):
        same_points = 0
        
        # 巡线类型
        if self.trackstate == TrackState.TRACK_MIDDLE:
            pass
        elif self.trackstate == TrackState.TRACK_AI_MIDDLE:
            pass

        # 元素检测
        enAI = getattr(self._config, 'enAI', False) if self._config else False
        if self.scene == Scene.Normal and enAI:
            # 金币
            if self.gold.flag_gold == GoldFlag.GOLD_NONE:
                self.gold.check_gold(predict_result)
                if self.gold.flag_gold != GoldFlag.GOLD_NONE:
                    pass
            
            # 汽车
            if self.car.flag_car == CarFlag.CAR_NONE:
                self.car.check_car(predict_result)
                if self.car.flag_car != CarFlag.CAR_NONE:
                    pass
                    
            # 行人
            if self.human.flag_human == HumanFlag.HUMAN_NONE:
                self.human.check_human(predict_result)
                if self.human.flag_human != HumanFlag.HUMAN_NONE:
                    pass
            
            # 红绿灯
            if self.light.flag_light == LightFlag.LIGHT_NONE:
                self.light.check_light(predict_result)
                if self.light.flag_light != LightFlag.LIGHT_NONE:
                    pass
            elif self.light.flag_light == LightFlag.LIGHT_LOST:
                pass

        # 元素处理
        self.gold_end_counter += 1
        if self.gold_end_counter > 600:
            self.gold_end_counter = 599
            
        if self.scene == Scene.Gold:
            self.gold.run_gold(src_img, predict_result)
            if self.gold.flag_gold == GoldFlag.GOLD_NONE:
                pass
        elif self.scene == Scene.Car:
            pass
        else:
            pass

        # 返回 task data (定义取决于实际应用，这里占位)
        return None
