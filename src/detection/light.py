from enum import Enum

class LightFlag(Enum):
    LIGHT_NONE = 0
    LIGHT_GREEN_FAR = 1
    LIGHT_GREEN_NEAR = 2
    LIGHT_RED_FAR = 3
    LIGHT_RED_NEAR = 4
    LIGHT_LOST = 5

class Light:
    def __init__(self, config=None):
        self.flag_light = LightFlag.LIGHT_NONE

    def check_light(self, predict_result: list):
        pass

    def run_light(self, src_img, predict_result: list):
        pass
