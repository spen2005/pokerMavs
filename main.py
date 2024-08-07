from agents.mcts import MCTS
from pypokerengine.api.emulator import Emulator
from pypokerengine.players import BasePokerPlayer

class MCTSPlayer(BasePokerPlayer):
    def __init__(self, max_depth=5):
        self.mcts = MCTS(max_depth)

    def declare_action(self, valid_actions, hole_card, round_state):
        # 选择一个随机动作作为示例
        action = valid_actions[np.random.choice(len(valid_actions))]["action"]
        return action, 0  # 第二个参数是动作的值，如跟注的金额

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

if __name__ == "__main__":
    # 初始化游戏
    emulator = Emulator()
    emulator.set_game_rule(nb_player=6, max_round=10, small_blind_amount=10, ante_amount=1)
    
    # 注册玩家
    players = [MCTSPlayer() for _ in range(6)]
    for i, player in enumerate(players):
        emulator.register_player("player{}".format(i), player)
    
    # 模拟游戏
    game_result = emulator.start_game()
    print(game_result)
