from agents.policy import PolicyNetwork
from agents.value_function import ValueFunction
from data.dataset import Dataset
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card_from_deck
import numpy as np

class MCTS:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.policy_network = PolicyNetwork()
        self.value_function = ValueFunction()
        self.dataset_p = Dataset()
        self.dataset_v = Dataset()
        self.emulator = Emulator()

    def update_networks(self):
        self.value_function.update(self.dataset_v)
        self.policy_network.update(self.dataset_p)

    def mcts_strategy(self, state, player, depth, max_depth):
        if self.is_game_end(state):
            return self.calculate_payoff(state)
        if depth > max_depth:
            return self.value_function.evaluate(state)

        if player == 0:
            # Dealer's turn
            sum_values = 0
            X = 10  # 假设我们生成10个可能的发牌结果
            for _ in range(X):
                state_x = self.dealer_deals(state)
                player_x = (player + 1) % 6
                sum_values += self.mcts_strategy(state_x, player_x, depth + 1, max_depth)
            return sum_values / X
        else:
            action_set = self.get_possible_actions(state)
            v_a = []
            for a in action_set:
                state_a = self.transition(state, a)
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
        game_state = restore_game_state(state)
        # 获取公共牌和每个玩家的手牌
        public_cards = gen_cards(game_state['table']['community_card'])
        player_hands = {player_uuid: gen_cards(info['hole_card']) for player_uuid, info in game_state['seats'].items()}

        # 获取还在游戏中的玩家
        active_players = [player_uuid for player_uuid, info in game_state['seats'].items() if not info['is_folding']]

        # 计算每个玩家的手牌评分
        hand_strengths = {player_uuid: HandEvaluator.eval_hand(player_hands[player_uuid], public_cards) for player_uuid in active_players}

        # 找到最大手牌评分和对应的玩家
        winning_player = max(hand_strengths, key=hand_strengths.get)

        # 获取总底池
        pot = game_state['table']['pot']['main']['amount']

        # 初始化每个玩家的收益
        payoffs = {player_uuid: 0 for player_uuid in game_state['seats'].keys()}

        # 将所有奖金分配给获胜者
        payoffs[winning_player] = pot

        return list(payoffs.values())


    def get_possible_actions(self, state):
        # Implement the logic to determine possible actions from the current state
        return [Action.FOLD, Action.CALL, Action.RAISE]

    def calculate_cost(self, action):
        if action == Action.FOLD:
            return 0  # Folding costs nothing
        elif action == Action.CALL:
            return self.state.current_bet - self.state.player_bets[self.state.current_player]
        elif action == Action.RAISE:
            return self.state.current_bet + self.min_raise - self.state.player_bets[self.state.current_player]
        else:
            raise ValueError("Invalid action")

    def transition(self, state, action):
        # Use PyPokerEngine to process a transition based on the action taken
        game_state = restore_game_state(state)
        actions = self.emulator.generate_possible_actions(game_state)
        # Map your action to PyPokerEngine's action structure
        emulator_action = actions[action]  # Map to CALL, FOLD, or RAISE
        updated_state, _ = self.emulator.apply_action(game_state, emulator_action)
        return updated_state

    def dealer_deals(self, state):
        # Use PyPokerEngine to handle the dealer dealing cards
        game_state = restore_game_state(state)
        # Assuming public_cards and deck are available in the state
        new_state = attach_hole_card_from_deck(game_state, game_state['table']['deck'])
        return new_state

    def is_game_end(self, state):
        # Determine if the game has ended using PyPokerEngine's state utilities
        game_state = restore_game_state(state)
        return game_state['street'] == 'showdown'
