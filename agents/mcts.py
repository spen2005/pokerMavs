import torch
import numpy as np
from pypokerengine.api.emulator import Emulator
from environment.actions import ActionType, get_action_space, BettingStage
from .PHE import PokerHandEvaluator

class MCTS:
    def __init__(self, policy_network, value_function, num_players=6):
        self.policy_network = policy_network
        self.value_function = value_function
        self.num_players = num_players
        self.phe = PokerHandEvaluator()
        self.emulator = Emulator()

    def act(self, game_state):
        betting_stage = BettingStage.PREFLOP if game_state['street'] == 'preflop' else BettingStage.POSTFLOP
        action_space = get_action_space(betting_stage)

        avg_expected_values, old_policy, new_policy = self.mcts_strategy(game_state)
        
        action_index = np.random.choice(len(new_policy), p=new_policy)
        chosen_action = action_space[action_index]

        return {'action': chosen_action, 'amount': self.get_action_amount(chosen_action, game_state)}

    def mcts_strategy(self, game_state, num_samples=1000):
        if self.is_game_end(game_state):
            return self.calculate_payoff(game_state), None, None

        betting_stage = BettingStage.PREFLOP if game_state['street'] == 'preflop' else BettingStage.POSTFLOP
        action_space = get_action_space(betting_stage)
        num_players = len(game_state['seats'])

        expected_values = np.zeros((len(action_space), num_players))

        for _ in range(num_samples):
            simulated_state = self.emulator.apply_action(game_state, self.act(game_state))
            
            while not self.is_round_end(simulated_state):
                simulated_state = self.emulator.apply_action(simulated_state, self.act(simulated_state))

            value_input = self.prepare_value_input(simulated_state)
            payoffs = self.value_function.evaluate(*value_input)
            
            action_index = action_space.index(simulated_state['action_histories'][betting_stage][0]['action'])
            expected_values[action_index] += np.array(payoffs) / num_samples

        current_player = game_state['next_player']
        current_policy_input = self.prepare_policy_input(game_state, self.calculate_hand_strengths(game_state)[current_player])
        old_policy = self.policy_network(current_policy_input).detach().numpy()

        current_player_expected_values = expected_values[:, current_player]
        avg_expected_value = np.dot(old_policy, current_player_expected_values)
        regrets = current_player_expected_values - avg_expected_value

        positive_regrets = np.maximum(regrets, 0)
        regret_sum = np.sum(positive_regrets)
        if regret_sum > 0:
            new_policy = positive_regrets / regret_sum
        else:
            new_policy = np.ones_like(old_policy) / len(old_policy)

        avg_expected_values = np.dot(old_policy, expected_values)

        return avg_expected_values, old_policy, new_policy

    def calculate_hand_strengths(self, game_state):
        hand_strengths = []
        for player in game_state['seats']:
            hole_cards = player['hole_card']
            community_cards = game_state['community_card']
            known_cards = hole_cards + community_cards
            strength_matrix = self.phe.monte_carlo_simulation(known_cards)
            hand_strengths.append(strength_matrix)
        return hand_strengths

    def calculate_public_strength(self, community_cards):
        return self.phe.monte_carlo_simulation(community_cards)

    def prepare_policy_input(self, game_state, hand_strength):
        round_num = torch.tensor(game_state['round_count'], dtype=torch.float32)
        my_position = torch.tensor(game_state['next_player'], dtype=torch.float32)
        active_players = torch.tensor(sum(1 for p in game_state['seats'] if p['state'] == 'participating'), dtype=torch.float32)
        player_status = torch.tensor([p['state'] == 'participating' for p in game_state['seats']], dtype=torch.float32)
        player_bets = torch.tensor([p['stack'] for p in game_state['seats']], dtype=torch.float32)
        min_bet = torch.tensor(game_state['small_blind_amount'], dtype=torch.float32)
        max_bet = torch.tensor(game_state['small_blind_amount'] * 2, dtype=torch.float32)
        public_strength = torch.tensor(self.calculate_public_strength(game_state['community_card']), dtype=torch.float32)

        return (round_num.unsqueeze(0), 
                torch.tensor(hand_strength, dtype=torch.float32).unsqueeze(0), 
                public_strength.unsqueeze(0), 
                my_position.unsqueeze(0), 
                active_players.unsqueeze(0), 
                player_status.unsqueeze(0), 
                player_bets.unsqueeze(0), 
                min_bet.unsqueeze(0), 
                max_bet.unsqueeze(0))

    def prepare_value_input(self, game_state):
        hand_strengths = self.calculate_hand_strengths(game_state)
        pot = game_state['pot']['main']['amount']
        return torch.tensor(hand_strengths, dtype=torch.float32).unsqueeze(0), torch.tensor([[pot]], dtype=torch.float32)

    def is_round_end(self, game_state):
        return game_state['street'] in ['preflop', 'flop', 'turn', 'river'] and \
               all(player['action_histories'][game_state['street']] for player in game_state['seats'])

    def is_game_end(self, game_state):
        return game_state['street'] == 'showdown'

    def calculate_payoff(self, game_state):
        payoffs = [0] * self.num_players
        for i, player in enumerate(game_state['seats']):
            payoffs[i] = player['stack'] - player['initial_stack']
        return payoffs

    def get_action_amount(self, action, game_state):
        if action in [ActionType.FOLD, ActionType.CALL]:
            return 0
        elif action == ActionType.RAISE:
            return game_state['small_blind_amount'] * 2  # 这里可以根据需要调整加注量
        elif action == ActionType.ALL_IN:
            return game_state['seats'][game_state['next_player']]['stack']
        else:
            raise ValueError(f"Invalid action: {action}")