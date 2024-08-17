from enum import Enum

class BettingStage(Enum):
    PREFLOP = 'preflop'
    POSTFLOP = 'postflop'

class ActionType(Enum):
    FOLD = 'fold'
    CALL = 'call'
    BET = 'bet'
    RAISE = 'raise'
    ALL_IN = 'all_in'

# 翻牌前动作空间
PREFLOP_ACTIONS = {
    ActionType.FOLD: [0],
    ActionType.CALL: [1],
    ActionType.RAISE: [2.5, 5, 10, 25],  # 2.5 5 10 25
    ActionType.ALL_IN: [float('inf')]
}

# 翻牌后动作空间
POSTFLOP_ACTIONS = {
    ActionType.FOLD: [0],
    ActionType.CALL: [1],
    ActionType.BET: [0.33, 0.66, 1, 1.5],  # 以底池百分比表示
    ActionType.ALL_IN: [float('inf')]
}

def get_action_space(stage: BettingStage):
    if stage == BettingStage.PREFLOP:
        return PREFLOP_ACTIONS
    elif stage == BettingStage.POSTFLOP:
        return POSTFLOP_ACTIONS
    else:
        raise ValueError("Invalid betting stage")