from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from agents.mcts import MCTS
from agents.policy import PolicyNetwork
from agents.value_function import ValueFunction
from environment.actions import ActionType, get_action_space, BettingStage
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class MCTSPlayer:
    def __init__(self, policy_network, value_function):
        self.mcts = MCTS(policy_network, value_function)
        self.policy_network = policy_network
        self.value_function = value_function

    def declare_action(self, valid_actions, hole_card, round_state):
        game_state = self.convert_round_state_to_game_state(round_state, hole_card)
        action = self.mcts.act(game_state)
        return action['action'], action['amount']

    def receive_game_start_message(self, game_info):
        self.game_info = game_info

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_card = hole_card
        self.seats = seats

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def convert_round_state_to_game_state(self, round_state, hole_card):
        game_state = {
            'next_player': round_state['next_player'],
            'round_count': round_state['round_count'],
            'small_blind_amount': round_state['small_blind_amount'],
            'street': round_state['street'],
            'seats': round_state['seats'],
            'community_card': round_state['community_card'],
            'pot': round_state['pot'],
            'hole_card': hole_card
        }
        return game_state

def train(num_games=1000, update_interval=5):
    num_players = 6
    policy_network = PolicyNetwork(action_dim=len(ActionType), num_players=num_players)
    value_function = ValueFunction(num_players=num_players, num_hand_categories=117)
    
    policy_optimizer = optim.Adam(policy_network.parameters())
    value_optimizer = optim.Adam(value_function.parameters())

    policy_data = []
    value_data = []

    config = setup_config(max_round=1, initial_stack=1000, small_blind_amount=5)
    for i in range(num_players):
        config.register_player(name=f"player_{i}", algorithm=MCTSPlayer(policy_network, value_function))

    for game in range(num_games):
        game_result = start_poker(config, verbose=0)
        
        for round_state in game_result['round_states']:
            betting_stage = BettingStage.PREFLOP if round_state['street'] == 'preflop' else BettingStage.POSTFLOP
            
            for player in round_state['seats']:
                if player['state'] == 'participating':
                    mcts = MCTS(policy_network, value_function)
                    game_state = MCTSPlayer.convert_round_state_to_game_state(None, round_state, player['hole_card'])
                    avg_expected_values, old_policy, new_policy = mcts.mcts_strategy(game_state)
                    
                    policy_input = mcts.prepare_policy_input(game_state, mcts.calculate_hand_strengths(game_state)[player['uuid']])
                    policy_data.append((policy_input, new_policy))
                    
                    if player == round_state['seats'][0]:
                        value_input = mcts.prepare_value_input(game_state)
                        value_data.append((value_input, avg_expected_values))

        if (game + 1) % update_interval == 0:
            update_networks(policy_network, value_function, policy_optimizer, value_optimizer, policy_data, value_data)
            policy_data = []
            value_data = []

        if (game + 1) % 100 == 0:
            print(f"Completed {game + 1} games")

def update_networks(policy_network, value_function, policy_optimizer, value_optimizer, policy_data, value_data):
    policy_optimizer.zero_grad()
    policy_loss = 0
    for inputs, target in policy_data:
        output = policy_network(inputs)
        policy_loss += F.kl_div(output.log(), torch.tensor(target), reduction='batchmean')
    policy_loss /= len(policy_data)
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss = 0
    for inputs, target in value_data:
        output = value_function(*inputs)
        value_loss += F.mse_loss(output, torch.tensor(target))
    value_loss /= len(value_data)
    value_loss.backward()
    value_optimizer.step()

    print(f"Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")

if __name__ == "__main__":
    train()