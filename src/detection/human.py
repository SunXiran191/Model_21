from enum import Enum

class HumanFlag(Enum):
    HUMAN_NONE = 0
    HUMAN_AVOID = 1
    HUMAN_LOST = 2

class Human:
    def __init__(self, config=None):
        self.flag_human = HumanFlag.HUMAN_NONE

    def check_human(self, predict_result: list):
        pass

    def run_human(self, src_img, predict_result: list):
        pass
