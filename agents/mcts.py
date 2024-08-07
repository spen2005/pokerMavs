from agents.policy import PolicyNetwork
from agents.value_function import ValueFunction
from data.dataset import Dataset
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card_from_deck
from environment.actions import Action
import numpy as np

class MCTS:
    def __init__(self, max_depth, policy_network_params, value_function_params):
        self.max_depth = max_depth
        self.policy_network = PolicyNetwork(**policy_network_params)
        self.value_function = ValueFunction(**value_function_params)
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
            X = 10  # Generate 10 possible dealing outcomes
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
                v_a.append(self.calculate_cost(a) + self.mcts_strategy(state_a, player_a, depth + 1, max_depth))

            expected_value = sum(v_a) / len(action_set)
            r_a = [v - expected_value for v in v_a]

            sum_positive_r_a = sum([max(0, r) for r in r_a])
            normalized_rewards = [(max(0, r) / sum_positive_r_a if sum_positive_r_a > 0 else 0) for r in r_a]

            self.dataset_p.add((normalized_rewards, self.policy_network.forward(state)))
            self.dataset_v.add(expected_value, self.value_function.evaluate(state))

            return expected_value

    def calculate_payoff(self, state):
        game_state = restore_game_state(state)
        public_cards = gen_cards(game_state['table']['community_card'])
        player_hands = {player_uuid: gen_cards(info['hole_card']) for player_uuid, info in game_state['seats'].items()}
        active_players = [player_uuid for player_uuid, info in game_state['seats'].items() if not info['is_folding']]
        hand_strengths = {player_uuid: HandEvaluator.eval_hand(player_hands[player_uuid], public_cards) for player_uuid in active_players}
        winning_player = max(hand_strengths, key=hand_strengths.get)
        pot = game_state['table']['pot']['main']['amount']
        payoffs = {player_uuid: 0 for player_uuid in game_state['seats'].keys()}
        payoffs[winning_player] = pot
        return list(payoffs.values())

    def get_possible_actions(self, state):
        # Determine possible actions based on the state
        return [Action.FOLD, Action.CALL, Action.RAISE]

    def calculate_cost(self, action):
        # Calculate the cost associated with an action
        if action == Action.FOLD:
            return 0
        elif action == Action.CALL:
            return self.state.current_bet - self.state.player_bets[self.state.current_player]
        elif action == Action.RAISE:
            return self.state.current_bet + self.min_raise - self.state.player_bets[self.state.current_player]
        else:
            raise ValueError("Invalid action")

    def transition(self, state, action):
        # Apply action to the state
        game_state = restore_game_state(state)
        actions = self.emulator.generate_possible_actions(game_state)
        emulator_action = actions[action]  # Map to CALL, FOLD, or RAISE
        updated_state, _ = self.emulator.apply_action(game_state, emulator_action)
        return updated_state

    def dealer_deals(self, state):
        # Dealer deals the hole card
        game_state = restore_game_state(state)
        new_state = attach_hole_card_from_deck(game_state, game_state['table']['deck'])
        return new_state

    def is_game_end(self, state):
        # Determine if the game has ended
        game_state = restore_game_state(state)
        return game_state['street'] == 'showdown'
