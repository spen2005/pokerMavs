from agents.policy import PolicyNetwork
from agents.value_function import ValueFunction
from environment.state import transition, dealer_deals, is_game_end
from data.dataset import Dataset

class MCTS:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.policy_network = PolicyNetwork()
        self.value_function = ValueFunction()
        self.dataset_p = Dataset()
        self.dataset_v = Dataset()

    def mcts_strategy(self, state, player, depth, max_depth):
        if is_game_end(state):
            return self.calculate_payoff(state)
        if depth > max_depth:
            return self.value_function.evaluate(state)

        if player == 0:
            # 庄家的回合
            state_x = transition(state, dealer_deals(state))
            player_x = (player + 1) % 6
            return self.mcts_strategy(state_x, player_x, depth + 1, max_depth)
        else:
            action_set = self.get_possible_actions(state)
            v_a = []
            for a in action_set:
                state_a = transition(state, a)
                player_a = (player + 1) % 6
                v_a.append(self.calculate_cost(a) + self.mcts_strategy(state_a, player_a, depth + 1, max_depth)[player])

            expected_value = sum(v_a) / len(action_set)
            r_a = [v - expected_value for v in v_a]

            sum_positive_r_a = sum([max(0, r) for r in r_a])
            normalized_rewards = [(max(0, r) / sum_positive_r_a if sum_positive_r_a > 0 else 0) for r in r_a]

            self.dataset_p.add((normalized_rewards, self.policy_network.forward(state)))
            self.dataset_v.add(expected_value, self.value_function.evaluate(state))

            return expected_value

    def calculate_payoff(self, state):
        # 计算游戏结束时的每个玩家的收益
        return [0] * 6  # 示例的收益列表

    def get_possible_actions(self, state):
        # 获取可能的行动集合
        return [0, 1, 2]  # 示例的三个可能的行动

    def calculate_cost(self, action):
        # 计算执行某个行动的成本
        return 0  # 示例的成本
