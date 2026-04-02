from enum import Enum

class GoldFlag(Enum):
    GOLD_NONE = 0
    GOLD_GET = 1
    GOLD_LOST = 2

class Gold:
    def __init__(self, config=None):
        self.flag_gold = GoldFlag.GOLD_NONE

    def check_gold(self, predict_result: list):
        pass

    def run_gold(self, src_img, predict_result: list):
        pass
